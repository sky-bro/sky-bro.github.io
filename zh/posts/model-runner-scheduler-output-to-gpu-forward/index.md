
Scheduler 决定 **what to run this step**；ModelRunner 决定 **how to run it on GPU**。如果说 Scheduler 把动态请求队列压缩成 `SchedulerOutput`，那 ModelRunner 就负责把这份调度结果翻译成连续 tensor、KV cache slot、attention metadata、forward context、logits 和 sampled token。

所以 ModelRunner 确实是推理执行核心。它不负责 HTTP 接入，也不负责全局排队策略；但一旦 scheduler 给出 `SchedulerOutput`，模型真正跑起来的过程就在这里展开。

这篇接在 [Scheduler]({{< relref "scheduler-request-queue-to-scheduler-output" >}}) 后面读；如果还没建立全链路地图，可以先看 [request lifecycle]({{< relref "request-lifecycle-openai-to-forward-pass" >}})。

{{< figure src="/images/posts/model-runner-scheduler-output-to-gpu-forward/model-runner-execution-map.svg" caption="<span class=\"figure-number\">Figure 1: </span>GPUModelRunner sits between SchedulerOutput and the actual model forward. It owns input materialization, attention metadata, KV slot mapping, forward context, logits, and sampling state." width="100%" >}}

## 从一个小 batch 开始 {#toy-batch}

假设这一轮 scheduler 给了两个 request：

| request | 已算 token | 本轮调度 token | 阶段 |
|---|---:|---:|---|
| A | 4 | 1 | decode |
| B | 0 | 3 | prefill chunk |

从 scheduler 看，这是一个很自然的 mixed batch：A 继续 decode 一个 token，B 开始 prefill 三个 token。但 GPU 不能直接执行“request A 一个，request B 三个”这种高层描述。ModelRunner 要把它变成更低层的数据：

| 数据 | 含义 | toy batch 里的形状 |
|---|---|---|
| `input_ids` | 本轮真正送进模型的 token | `[A4, B0, B1, B2]` |
| `positions` | 每个 token 在原序列里的位置 | `[4, 0, 1, 2]` |
| `query_start_loc` | 每个 request 在扁平 token batch 里的边界 | `[0, 1, 4]` |
| `seq_lens` | 本轮 forward 后的乐观长度 | `[5, 3]` |
| `slot_mapping` | 每个 token 的 KV 写入/读取物理位置 | 由 block table 计算 |
| `logits_indices` | 哪些 hidden state 需要算 logits | 通常每个 request 最后一个位置 |

这就是读 ModelRunner 的关键：它不是“调用 PyTorch model”这么简单，而是在调度、KV cache、attention backend、CUDA graph、pipeline parallel、speculative decoding 和 sampling 之间维护一组执行不变量。

## 源码入口和执行流程 {#entry}

本文看 vLLM V1 主线：

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

`GPUWorker` 是 worker 层入口。它处理 pipeline-parallel rank 之间的 tensor 收发，然后把本轮 `SchedulerOutput` 交给 `model_runner.execute_model()`。当前源码里，`GPUWorker` 可以按配置选择 V1 runner 或 V2 runner；本文用 V1 讲机制，因为它把职责集中在一个文件里，更适合第一次源码阅读。

`GPUModelRunner.execute_model()` 本身是两段式的：先 preprocess、forward、compute logits，把临时状态塞进 `ExecuteModelState`，然后返回 `None`；后续 `GPUWorker.sample_tokens()` 再调用 `model_runner.sample_tokens()` 完成采样、更新 request 状态，并产生 `ModelRunnerOutput`。这个拆分服务于 async scheduling、pipeline parallel、speculative decoding 和 structured output。

一次 `execute_model()` 可以压成这张表：

| 阶段 | 做什么 | 为什么重要 |
|---|---|---|
| 更新 persistent batch | 把 `SchedulerOutput` 增量同步到 runner 的 batch 状态 | 避免每步从 Python 对象重建大 tensor |
| 准备 input tensor | 生成 `req_indices`、`input_ids`、`positions`、`query_start_loc`、`logits_indices` | 把 request-level 决策压成 token-level tensor |
| 决定执行形状 | 选择 padding、CUDA graph mode、microbatch、跨 DP token 数 | 在动态请求和 GPU 稳定形状之间折中 |
| 构造 attention metadata | 计算 block table、slot mapping、seq lens、prefill/decode/spec 状态 | attention backend 靠这些信息读写 KV cache |
| forward 与采样 | 在 `set_forward_context(...)` 下调用模型，计算 logits，再 sample | 产生下一轮 scheduler 需要的 token 和状态 |

真正的模型调用很短：

```python
return self.model(
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    **model_kwargs,
)
```

但这段短代码站在大量准备工作之上。`set_forward_context(...)` 已经设置好 attention metadata、slot mapping、CUDA graph runtime mode、microbatch slices 等信息；`input_ids`、`positions`、`inputs_embeds` 和 pipeline intermediate tensors 也已经按执行形状准备好。

这个边界很重要：具体模型类负责 transformer block、MLP、MoE、logits head 等结构；ModelRunner 和 attention backend 负责告诉这些层“这次 batch 的 KV cache 和 attention 运行环境是什么”。二者合起来才是一次高性能 forward。

## 为什么 V2 要重写这层 {#mrv2}

vLLM 源码里已经有 Model Runner V2 设计文档。它的存在本身说明：ModelRunner 是推理执行中最容易堆复杂度的位置。

| 问题 | V1 的压力 | V2 的方向 |
|---|---|---|
| persistent batch | state 和 input tensor 耦合，重排复杂 | 持久状态与 per-step input 解耦 |
| async scheduling | CPU/GPU 异步拷贝容易出现 race | async-first，减少同步屏障 |
| block table 更新 | 大 tensor 每轮复制成本高 | staged write，只提交增量 |
| sampling | Python/torch 组合路径复杂 | Triton-native sampler |
| CUDA graph | capture/launch 逻辑隐式 | 显式 CUDA graph manager |
| 文件结构 | V1 `gpu_model_runner.py` 巨大 | 拆成更模块化的 runner 组件 |

读 V1 是为了理解机制，读 V2 是为了理解工程演进方向。推理框架难的地方，不是“能不能调用 model.forward”，而是能不能在动态请求流、KV cache、attention backend、采样、并行通信和 CUDA graph 之间保持一致的不变量。

## vLLM-Omni 如何扩展这个边界 {#omni}

vLLM-Omni 没有把 vLLM ModelRunner 丢掉。它复用这个边界，再往上扩展多模态和多阶段执行。

在 `vllm-omni` 源码里，`OmniGPUModelRunner` 继承自 vLLM 的 `GPUModelRunner`。`GPUARModelRunner` 面向自回归 stage，保留两阶段 execute/sample 语义，同时把 per-request hidden representations、多模态输出、connector payload 等信息带回上层。`GPUGenerationModelRunner` 面向非自回归 generation stage，复用输入准备、多模态处理和 TP/PP/DP glue，但它不计算 logits，也不执行 token sampling，而是把生成过程的结果通过输出字段返回。

更一般地说：vLLM 的 ModelRunner 是 AR transformer serving 的执行核心；vLLM-Omni 把这个执行核心放进更大的 stage graph。文本 token、语音 token 这类自回归阶段仍然可以落回 scheduler、KV cache、attention metadata、model runner 体系；diffusion、vocoder、code2wav 这类非 AR 阶段则需要专门的 runner/output 协议。

## 读源码时抓住这些不变量 {#invariants}

读 ModelRunner 很容易迷路，因为文件长、分支多、feature flag 多。先抓住五个不变量：

- `SchedulerOutput` 是输入契约：scheduler 决定本轮哪些 request 得到多少 token budget。
- `InputBatch` 是跨 step 状态：runner 维护 token ids、request index、sampling metadata、block table 等持久状态。
- `slot_mapping` 是 KV cache 的落点：每个本轮 token 必须映射到可读写的物理 KV slot。
- `forward context` 是 attention 的运行时环境：attention layer 从这里拿 batch metadata。
- sampled token 会回流到下一轮 scheduler：一次 forward 的结果不是终点，而是下一次调度的输入。

如果只记一句话：**ModelRunner 把“这一轮要算什么”翻译成“GPU 上按什么形状、读写哪些 KV slot、用哪些 attention metadata 来算”。**

建议继续读：

1. `vllm/v1/worker/gpu_worker.py`：看 worker 如何调用 runner，以及 pipeline parallel 的边界。
2. `vllm/v1/worker/gpu_model_runner.py`：重点看 `execute_model()`、`_prepare_inputs()`、`_build_attention_metadata()`、`sample_tokens()`。
3. `vllm/v1/worker/gpu_input_batch.py`：理解 `InputBatch` 如何承载 persistent batch。
4. `vllm/docs/design/model_runner_v2.md`：对照 V1 的复杂度，理解 V2 为什么要重构。
5. `vllm-omni/vllm_omni/worker/*model_runner.py`：看 Omni 如何继承、覆写和扩展 runner 边界。
