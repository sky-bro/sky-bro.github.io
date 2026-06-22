
在 request lifecycle 里，Scheduler 是最容易被低估的一段。HTTP server 负责接入请求，ModelRunner 负责把 batch 跑到 GPU 上；Scheduler 每一轮要回答的是：**这一步到底让谁跑，跑几个 token，KV cache 放不放得下？**

可以把三篇文章的边界先记成一句话：

| 文章 | 关注的问题 |
|---|---|
| [request lifecycle]({{< relref "request-lifecycle-openai-to-forward-pass" >}}) | 请求怎样从 OpenAI API 走到 EngineCore |
| Scheduler | EngineCore 每一步决定 **what to run** |
| [ModelRunner]({{< relref "model-runner-scheduler-output-to-gpu-forward" >}}) | 把 `SchedulerOutput` 变成 **how to run on GPU** |

Scheduler 的产物不是一个模糊的 “batch”，而是一份具体的 `SchedulerOutput`：哪些 request 是新进来的，哪些已经在 worker 侧缓存，谁本轮算几个 token，哪些 KV block 新分配，哪些 request 被抢占，哪些 finished request 需要清理。

{{< figure src="/images/posts/scheduler-request-queue-to-scheduler-output/scheduler-output-map.svg" caption="<span class=\"figure-number\">Figure 1: </span>Scheduler turns request queues, token budgets, KV cache allocation, prefix-cache hits, and preemption decisions into SchedulerOutput. ModelRunner consumes this object in the next stage." width="100%" >}}

## 源码入口和闭环 {#enginecore-step}

先把 Scheduler 放回 `EngineCore.step()`：

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

Scheduler 不是请求开始时运行一次的模块。它在 engine busy loop 里反复运行：每一步先生成 `SchedulerOutput`，交给 ModelRunner 执行，再用 ModelRunner 返回的 sampled token、logprobs、pooling output、KV connector output 等结果更新内部状态。

这意味着 Scheduler 维护的是一个动态系统：

- request 的 `num_computed_tokens` 每一步都在变；
- output token、spec token、placeholder token 会改变下一步还差多少 token；
- KV cache block 可能被新分配、复用、抢占、延迟释放；
- waiting queue 里的 request 可能因为 remote KV、structured output grammar、streaming input 等原因暂时不可调度；
- running queue 里的 request 也不一定每一步都能跑。

## 一步 schedule() 到底做什么 {#schedule-mainline}

`Scheduler.schedule()` 的注释里有一句关键话：scheduler 里没有固定的 “decoding phase” 或 “prefill phase”。每个 request 只有当前已计算的 `num_computed_tokens` 和目标侧的 `num_tokens_with_spec`。每一步，scheduler 尝试给 request 分配一些 token，让 `num_computed_tokens` 追上目标。

这套抽象同时覆盖普通 decode、prefill、chunked prefill、prefix caching 和 speculative decoding。一个最小例子：

| request | 状态 | 已算 token | 目标 token | 本轮可能调度 |
|---|---|---:|---:|---|
| A | running | 99 | 100 | decode 1 token |
| B | running | 0 | 4096 | prefill chunk |
| C | waiting | 0 | 128 | new prefill |

Scheduler 不是简单 FIFO。它还要看 token budget、长 prefill 阈值、`max_num_running_reqs`、KV block 是否足够、prefix cache 命中、DP prefill balancing 等约束。主线可以压成这张表：

| 阶段 | 做什么 | 关键输出 |
|---|---|---|
| 初始化预算 | 设置 `token_budget`、encoder budget、临时列表和字典 | 本轮资源上限 |
| 先看 running | 已在运行的 request 优先推进 | `scheduled_running_reqs`、`num_scheduled_tokens` |
| 分配 KV slots | 为本轮新增 token 调 `kv_cache_manager.allocate_slots(...)` | `req_to_new_blocks` |
| 不够就抢占 | 释放低优先级 running request 的 KV blocks，放回 waiting | `preempted_reqs` |
| 再看 waiting | 接纳新 request 或恢复 preempted request，处理 prefix/remote KV | `scheduled_new_reqs`、`scheduled_resumed_reqs` |
| 构造输出 | 汇总 request 增量、block ids、connector metadata | `SchedulerOutput` |

最关键的一点是：**KV cache 分配发生在调度时**。Scheduler 不是先决定 batch，再让 worker 祈祷显存放得下；它在调度阶段就尝试分配 KV slots。如果分配失败，就可能触发 preemption。

`_preempt_request(...)` 会释放 request 的 KV blocks 和 encoder cache，把状态改成 `PREEMPTED`，把 `num_computed_tokens` 归零，清掉 spec tokens，然后放回 waiting queue 前端。这说明调度策略不只是 fairness 或 FIFO，它受 KV block 可用性强约束。

Prefix cache 也在这里影响调度。waiting request 首次进入时，scheduler 会用 `kv_cache_manager.get_computed_blocks(request)` 查本地 prefix cache；如果启用 KVConnector，还可能查外部/远端 KV。命中后，`num_computed_tokens` 不再是 0，scheduler 只调度剩余未计算 token。也就是说，prefix cache 改变的是本轮 `num_scheduled_tokens` 和 KV block 分配，而不是后面 attention 层的一个小细节。

## SchedulerOutput 是什么 {#scheduler-output}

`vllm/v1/core/sched/output.py` 里的 `SchedulerOutput` 是 Scheduler 和 ModelRunner 之间的关键接口。先记这些字段就够了：

| 字段 | 作用 |
|---|---|
| `scheduled_new_reqs` | 第一次被调度的 request，worker 侧还没有完整缓存 |
| `scheduled_cached_reqs` | 已经被调度过的 request，只发送增量状态 |
| `num_scheduled_tokens` | 核心字段：`req_id -> 本轮调度 token 数` |
| `total_num_scheduled_tokens` | 本轮总 token 数，ModelRunner 用它判断是否需要 forward |
| `scheduled_spec_decode_tokens` | 本轮一起验证/执行的 speculative draft tokens |
| `scheduled_encoder_inputs` | 多模态/encoder 输入本轮需要处理哪些 |
| `num_common_prefix_blocks` | running requests 的 common prefix blocks，可供 cascade attention 使用 |
| `finished_req_ids` | 上一步到当前 step 之间完成的 request，需要 worker 清理缓存状态 |
| `preempted_req_ids` | 本轮被抢占的 request，V2 runner 特别会用到 |
| `kv_connector_metadata` | KV transfer/load/save 相关的不透明 metadata |
| `new_block_ids_to_zero` | 新分配且需要 worker 清零的 KV block |

ModelRunner 读到这份对象后，会根据 `scheduled_new_reqs` / `scheduled_cached_reqs` 更新自己的 `InputBatch`，根据 `num_scheduled_tokens` 准备扁平 token batch，根据 block ids 和 slot mapping 建 attention metadata。这就和 ModelRunner 那篇接上了。

## 另半个闭环：update_from_output() {#update-from-output}

只看 `schedule()` 还不完整。`scheduler.update_from_output(...)` 会在 ModelRunner 执行后更新 Scheduler 状态，包括 sampled token、accepted/rejected draft tokens、stop condition、logprobs、pooling output、KV connector 结果、stopped request 清理和 scheduler stats。

一个容易误解的细节是：`_update_after_schedule(...)` 会在 schedule 后先把 request 的 `num_computed_tokens` 往前推进，方便下一轮 scheduler 立即继续调度 chunked prefill；如果 speculative tokens 后来被拒绝，`update_from_output(...)` 再把 computed token 数修正回来。

所以 Scheduler 是一个 optimistic state machine：先根据调度结果推进状态，让 engine pipeline 持续流动；等 GPU 输出回来后，再根据真实采样、拒绝、停止、错误和 KV transfer 结果修正状态。

## 边界和读法 {#boundary}

Scheduler 和 ModelRunner 的边界可以这样看：

| 模块 | 负责的问题 | 典型数据 |
|---|---|---|
| Scheduler | 本轮谁能跑、跑几个 token、KV cache 是否放得下 | `SchedulerOutput`, `num_scheduled_tokens`, block ids |
| ModelRunner | 本轮如何在 GPU 上执行 | `InputBatch`, `input_ids`, `positions`, `slot_mapping`, attention metadata |

读 Scheduler 时抓住六个不变量：

- 每个 engine step 对应一次调度决策；
- running 先于 waiting；
- scheduled tokens 总和不能超过 `max_num_scheduled_tokens`；
- KV slot 分配是 schedule 决策的一部分；
- prefix cache 命中会减少本轮 forward token；
- GPU 执行不再看 waiting/running queue，而是看 `SchedulerOutput`。

如果只记一句话：**Scheduler 把动态请求队列和 KV cache 约束，压缩成一份本轮可执行的 `SchedulerOutput`。** ModelRunner 接过这份输出，才开始准备真正的 GPU forward。

建议继续读：

1. `vllm/v1/engine/core.py`：看 `EngineCore.step()` 如何把 schedule、execute、update 串成闭环。
2. `vllm/v1/core/sched/interface.py`：先读 `schedule()` 的接口注释。
3. `vllm/v1/core/sched/scheduler.py`：重点读 `schedule()`、`_preempt_request()`、`_make_cached_request_data()`、`update_from_output()`。
4. `vllm/v1/core/sched/output.py`：把 `SchedulerOutput` 字段和 ModelRunner 使用方式对应起来。
5. `vllm/v1/core/kv_cache_manager.py`：继续追 `allocate_slots(...)`，理解调度为什么绕不开 KV block 管理。
