
Scheduler decides **what to run this step**. ModelRunner decides **how to run it on GPU**. If Scheduler compresses dynamic request queues into `SchedulerOutput`, ModelRunner translates that output into contiguous tensors, KV cache slots, attention metadata, forward context, logits, and sampled tokens.

So yes, ModelRunner is the execution core of inference. It does not own HTTP serving or global queue policy, but once `SchedulerOutput` exists, the model starts running here.

Read this after the [Scheduler]({{< relref "scheduler-request-queue-to-scheduler-output" >}}) post. If the whole path is still fuzzy, start from the [request lifecycle]({{< relref "request-lifecycle-openai-to-forward-pass" >}}) overview.

{{< figure src="/images/posts/model-runner-scheduler-output-to-gpu-forward/model-runner-execution-map.svg" caption="<span class=\"figure-number\">Figure 1: </span>GPUModelRunner sits between SchedulerOutput and the actual model forward. It owns input materialization, attention metadata, KV slot mapping, forward context, logits, and sampling state." width="100%" >}}

## Start With A Small Batch {#toy-batch}

Suppose the scheduler emits this step:

| request | computed tokens | scheduled tokens | phase |
|---|---:|---:|---|
| A | 4 | 1 | decode |
| B | 0 | 3 | prefill chunk |

From the scheduler's perspective, this is a normal mixed batch: request A decodes one token, while request B prefills three tokens. The GPU cannot execute that high-level description directly. ModelRunner lowers it into execution data:

| data | meaning | toy batch shape |
|---|---|---|
| `input_ids` | actual tokens for this step | `[A4, B0, B1, B2]` |
| `positions` | each token's position in its sequence | `[4, 0, 1, 2]` |
| `query_start_loc` | request boundaries in the flattened token batch | `[0, 1, 4]` |
| `seq_lens` | optimistic sequence lengths after this forward | `[5, 3]` |
| `slot_mapping` | physical KV cache slot for each token | computed from the block table |
| `logits_indices` | hidden states that should become logits | usually the last position per request |

That is the core mental model: ModelRunner is not just "calling a PyTorch model." It maintains execution invariants across scheduling, KV cache, attention backends, CUDA graphs, pipeline parallelism, speculative decoding, and sampling.

## Entry Points And Execution Flow {#entry}

This post follows the vLLM V1 path:

```text
vllm/v1/worker/gpu_worker.py
  GPUWorker.execute_model()
    -> self.model_runner.execute_model(...)

vllm/v1/worker/gpu_model_runner.py
  GPUModelRunner.execute_model()
    -> _update_states(...)
    -> _prepare_inputs(...)
    -> _get_slot_mappings(...)
    -> _build_attention_metadata(...)
    -> _preprocess(...)
    -> set_forward_context(...)
    -> _model_forward(...)
    -> compute_logits(...)
    -> sample_tokens(...)
```

`GPUWorker` is the worker-level entry point. It handles pipeline-parallel tensor receive/send and then calls `model_runner.execute_model()` with the current `SchedulerOutput`. In this checkout, `GPUWorker` can choose the V1 runner or the V2 runner. This article uses V1 because its responsibilities are concentrated in one file, which makes the mechanism easier to inspect first.

`GPUModelRunner.execute_model()` is a two-phase path: preprocess, forward, compute logits, store ephemeral state in `ExecuteModelState`, then return `None`; later, `GPUWorker.sample_tokens()` calls `model_runner.sample_tokens()` to sample, update request state, and produce `ModelRunnerOutput`. This split supports async scheduling, pipeline parallelism, speculative decoding, and structured output.

One `execute_model()` step can be compressed into this table:

| phase | what happens | why it matters |
|---|---|---|
| update persistent batch | apply `SchedulerOutput` deltas to runner-owned batch state | avoids rebuilding large tensors from Python objects every step |
| build input tensors | create `req_indices`, `input_ids`, `positions`, `query_start_loc`, `logits_indices` | lowers request-level decisions into token-level tensors |
| choose execution shape | select padding, CUDA graph mode, microbatching, cross-DP token counts | reconciles dynamic traffic with stable GPU shapes |
| build attention metadata | compute block tables, slot mappings, seq lens, prefill/decode/spec state | tells attention backends how to read/write KV cache |
| forward and sample | call the model under `set_forward_context(...)`, compute logits, then sample | produces tokens and state for the next scheduler step |

The actual model call is short:

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

That short call sits on top of all the prior preparation. `set_forward_context(...)` has already installed attention metadata, slot mappings, CUDA graph runtime mode, and microbatch slices. `input_ids`, `positions`, `inputs_embeds`, and pipeline intermediate tensors have been shaped for execution.

The boundary is important: model classes own transformer blocks, MLPs, MoE layers, and logits heads; ModelRunner plus attention backends own the runtime environment in which those layers execute. A high-performance forward pass needs both.

## Why V2 Reworks This Layer {#mrv2}

The vLLM source tree already contains a Model Runner V2 design document. Its existence is itself a signal: ModelRunner is where inference execution complexity accumulates.

| problem | V1 pressure | V2 direction |
|---|---|---|
| persistent batch | state and input tensors are coupled | decouple persistent state from per-step inputs |
| async scheduling | CPU/GPU async copies can race | async-first execution and fewer barriers |
| block table updates | large tensors are expensive to copy every step | staged writes that submit only deltas |
| sampling | Python/torch paths are complex | Triton-native sampler |
| CUDA graphs | capture/launch logic is implicit | explicit CUDA graph manager |
| file structure | V1 `gpu_model_runner.py` is large | split runner logic into focused modules |

Read V1 to understand the mechanism. Read V2 to understand the engineering direction. The hard part is not calling `model.forward`; it is preserving invariants across dynamic requests, KV cache, attention backends, sampling, parallel communication, and CUDA graph execution.

## How vLLM-Omni Extends The Boundary {#omni}

vLLM-Omni does not discard vLLM's ModelRunner boundary. It reuses and extends it for multimodal, multi-stage execution.

In `vllm-omni`, `OmniGPUModelRunner` inherits from vLLM's `GPUModelRunner`. `GPUARModelRunner` targets autoregressive stages. It keeps the two-phase execute/sample flow while returning per-request hidden representations, multimodal outputs, and connector payloads. `GPUGenerationModelRunner` targets non-autoregressive generation stages. It reuses input preparation, multimodal handling, and TP/PP/DP glue, but does not compute logits or run token sampling; instead, it returns generation outputs through output fields.

More generally: vLLM's ModelRunner is the execution core for AR transformer serving; vLLM-Omni places that core inside a larger stage graph. Text tokens and speech tokens can still use the scheduler, KV cache, attention metadata, and model-runner machinery. Diffusion, vocoder, and code2wav stages need specialized runner/output protocols because their outputs are not next-token logits.

## Source-Reading Invariants {#invariants}

ModelRunner is easy to get lost in because the file is long and feature-flag heavy. Start with five invariants:

- `SchedulerOutput` is the input contract: the scheduler decides which requests get token budget in this step.
- `InputBatch` is cross-step state: the runner owns token ids, request indices, sampling metadata, block tables, and related persistent state.
- `slot_mapping` is the KV cache landing zone: every token in this step must map to a physical KV slot.
- `forward context` is the attention runtime environment: attention layers read batch metadata from it.
- sampled tokens feed the next scheduler step: a forward pass is not the endpoint; it produces the next scheduling input.

If you remember one sentence, make it this: **ModelRunner translates "what should run this step" into "which GPU shape, which KV slots, and which attention metadata should be used to run it."**

A practical reading order:

1. `vllm/v1/worker/gpu_worker.py`: how the worker calls the runner and where pipeline parallelism enters.
2. `vllm/v1/worker/gpu_model_runner.py`: focus on `execute_model()`, `_prepare_inputs()`, `_build_attention_metadata()`, and `sample_tokens()`.
3. `vllm/v1/worker/gpu_input_batch.py`: understand how `InputBatch` carries persistent batch state.
4. `vllm/docs/design/model_runner_v2.md`: compare V1 complexity with the V2 design.
5. `vllm-omni/vllm_omni/worker/*model_runner.py`: see how Omni inherits, overrides, and extends the runner boundary.
