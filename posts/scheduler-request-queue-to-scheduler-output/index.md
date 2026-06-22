
In the request lifecycle, Scheduler is the easiest piece to underestimate. The HTTP server admits requests, and ModelRunner executes batches on GPU. Scheduler answers the per-step question in between: **who runs now, how many tokens do they get, and can the KV cache hold the result?**

The three lifecycle posts fit together like this:

| post | question |
|---|---|
| [request lifecycle]({{< relref "request-lifecycle-openai-to-forward-pass" >}}) | how a request reaches EngineCore |
| Scheduler | EngineCore decides **what to run** in each step |
| [ModelRunner]({{< relref "model-runner-scheduler-output-to-gpu-forward" >}}) | `SchedulerOutput` becomes **how to run on GPU** |

Scheduler output is not a vague "batch." It is a concrete `SchedulerOutput`: which requests are new, which are already cached on workers, how many tokens each request gets, which KV blocks were allocated, which requests were preempted, and which finished requests must be cleaned up.

{{< figure src="/images/posts/scheduler-request-queue-to-scheduler-output/scheduler-output-map.svg" caption="<span class=\"figure-number\">Figure 1: </span>Scheduler turns request queues, token budgets, KV cache allocation, prefix-cache hits, and preemption decisions into SchedulerOutput. ModelRunner consumes this object in the next stage." width="100%" >}}

## Entry Points And The Loop {#enginecore-step}

Put Scheduler back into `EngineCore.step()`:

```text
vllm/v1/engine/core.py
  EngineCore.step()
    -> scheduler.schedule(...)
    -> model_executor.execute_model(scheduler_output, ...)
    -> scheduler.update_from_output(scheduler_output, model_runner_output)

vllm/v1/core/sched/scheduler.py
  Scheduler.schedule()

vllm/v1/core/sched/output.py
  SchedulerOutput
```

Scheduler is not a one-time module that runs only when a request arrives. It runs repeatedly in the engine busy loop. Each step creates a `SchedulerOutput`, ModelRunner executes it, and Scheduler consumes ModelRunner output to update its state.

That means Scheduler maintains a dynamic system:

- a request's `num_computed_tokens` changes every step;
- output tokens, speculative tokens, and placeholder tokens change how many tokens remain;
- KV cache blocks may be allocated, reused, preempted, or freed later;
- waiting requests may be blocked by remote KV transfer, structured-output grammar, streaming input, or similar dependencies;
- running requests are not guaranteed to run in every step.

## What One schedule() Step Does {#schedule-mainline}

The comment at the top of `Scheduler.schedule()` is the key: the scheduler does not have a hard-coded "decode phase" or "prefill phase." Each request has `num_computed_tokens` and a target `num_tokens_with_spec`. At each step, Scheduler tries to assign enough tokens for requests to catch up.

That one abstraction covers normal decode, prefill, chunked prefill, prefix caching, and speculative decoding. Small example:

| request | state | computed tokens | target tokens | possible scheduling |
|---|---|---:|---:|---|
| A | running | 99 | 100 | decode 1 token |
| B | running | 0 | 4096 | prefill chunk |
| C | waiting | 0 | 128 | new prefill |

This is not plain FIFO. Scheduler must also check token budget, long-prefill thresholds, `max_num_running_reqs`, KV block availability, prefix-cache hits, and DP prefill balancing. The main path is:

| phase | what happens | key output |
|---|---|---|
| initialize budgets | set `token_budget`, encoder budget, temporary lists and maps | step resource limits |
| schedule running | active requests get first chance to advance | `scheduled_running_reqs`, `num_scheduled_tokens` |
| allocate KV slots | call `kv_cache_manager.allocate_slots(...)` for new tokens | `req_to_new_blocks` |
| preempt if needed | free low-priority running request blocks and move it back to waiting | `preempted_reqs` |
| admit waiting | admit new or preempted requests, handling prefix/remote KV | `scheduled_new_reqs`, `scheduled_resumed_reqs` |
| build output | gather request deltas, block ids, connector metadata | `SchedulerOutput` |

The important point: **KV cache allocation happens during scheduling**. Scheduler does not first form a batch and then hope workers can fit it in memory. It allocates KV slots while deciding the step. If allocation fails, preemption may happen.

`_preempt_request(...)` frees the request's KV blocks and encoder cache, marks it as `PREEMPTED`, resets `num_computed_tokens`, clears speculative tokens, and puts it back at the front of the waiting queue. Scheduling is therefore constrained by KV block availability, not just fairness or FIFO order.

Prefix cache also changes scheduling here. When a waiting request first enters, `kv_cache_manager.get_computed_blocks(request)` checks local prefix-cache hits; KVConnector may add external or remote hits. After a hit, `num_computed_tokens` is no longer zero, so Scheduler only schedules the remaining tokens. Prefix cache changes `num_scheduled_tokens` and KV block allocation, not just a later attention detail.

## What SchedulerOutput Contains {#scheduler-output}

`SchedulerOutput` in `vllm/v1/core/sched/output.py` is the contract between Scheduler and ModelRunner. These fields are the important ones:

| field | role |
|---|---|
| `scheduled_new_reqs` | requests scheduled for the first time; worker does not yet cache full request data |
| `scheduled_cached_reqs` | requests already known to workers; only deltas are sent |
| `num_scheduled_tokens` | core field: `req_id -> token count for this step` |
| `total_num_scheduled_tokens` | total scheduled token count; ModelRunner uses it to decide whether forward is needed |
| `scheduled_spec_decode_tokens` | speculative draft tokens verified or executed in this step |
| `scheduled_encoder_inputs` | multimodal or encoder inputs that need processing now |
| `num_common_prefix_blocks` | common prefix blocks among running requests, usable by cascade attention |
| `finished_req_ids` | requests finished since the previous step, used for worker-side cleanup |
| `preempted_req_ids` | requests preempted in this step, especially relevant to V2 runner paths |
| `kv_connector_metadata` | opaque metadata for KV transfer/load/save |
| `new_block_ids_to_zero` | freshly allocated KV blocks that workers should zero before use |

ModelRunner consumes this object by updating `InputBatch` from `scheduled_new_reqs` and `scheduled_cached_reqs`, preparing a flattened token batch from `num_scheduled_tokens`, and building attention metadata from block ids and slot mappings.

## The Other Half: update_from_output() {#update-from-output}

Reading only `schedule()` is not enough. `scheduler.update_from_output(...)` updates Scheduler state after ModelRunner executes. It handles sampled token ids, accepted or rejected speculative draft tokens, stop conditions, logprobs, pooling outputs, KV connector results, stopped request cleanup, and scheduler stats.

One detail matters: `_update_after_schedule(...)` advances each request's `num_computed_tokens` immediately after scheduling, so the next scheduler step can continue chunked prefills without waiting. If speculative tokens are later rejected, `update_from_output(...)` corrects the computed-token count.

Scheduler is therefore an optimistic state machine. It advances state based on the schedule to keep the engine pipeline moving, then corrects state when real GPU outputs, rejections, stops, errors, or KV transfer results arrive.

## Boundary And Reading Guide {#boundary}

The Scheduler/ModelRunner boundary is:

| module | question answered | typical data |
|---|---|---|
| Scheduler | who runs this step, how many tokens, and whether KV cache can fit | `SchedulerOutput`, `num_scheduled_tokens`, block ids |
| ModelRunner | how this step runs on GPU | `InputBatch`, `input_ids`, `positions`, `slot_mapping`, attention metadata |

When reading Scheduler, hold onto these invariants:

- one engine step maps to one scheduling decision;
- running requests are considered before waiting requests;
- scheduled tokens must not exceed `max_num_scheduled_tokens`;
- KV slot allocation is part of scheduling;
- prefix-cache hits reduce this step's forward work;
- GPU execution reads `SchedulerOutput`, not waiting/running queues.

If you remember one sentence, make it this: **Scheduler compresses dynamic request queues and KV cache constraints into an executable `SchedulerOutput` for the current step.** ModelRunner then turns that output into a real GPU forward.

A practical reading order:

1. `vllm/v1/engine/core.py`: see how `EngineCore.step()` connects schedule, execute, and update.
2. `vllm/v1/core/sched/interface.py`: read the `schedule()` interface comment first.
3. `vllm/v1/core/sched/scheduler.py`: focus on `schedule()`, `_preempt_request()`, `_make_cached_request_data()`, and `update_from_output()`.
4. `vllm/v1/core/sched/output.py`: map `SchedulerOutput` fields to how ModelRunner consumes them.
5. `vllm/v1/core/kv_cache_manager.py`: follow `allocate_slots(...)` to see why scheduling cannot be separated from KV block management.
