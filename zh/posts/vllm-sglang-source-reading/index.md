
这条系列专门放源码阅读和工程落地文章。目标不是逐文件翻译源码，而是把推理引擎里的关键机制定位到真实代码路径，并用 benchmark 或小实验验证它们的行为。

## 阅读顺序 {#reading-order}

后续文章会按请求生命周期展开：

1. [请求生命周期：OpenAI API 到一次 forward]({{< relref "request-lifecycle-openai-to-forward-pass" >}})
2. [推理采样：temperature、top-p、top-k]({{< relref "llm-sampling-temperature-top-p-top-k" >}})
3. Scheduler loop：waiting queue、running queue、token budget 和 decode 优先
4. vLLM Block Manager：从逻辑 block 到物理 KV block
5. SGLang Radix Cache：为什么前缀复用要用树
6. Prefix cache 命中一次到底省了什么
7. Chunked prefill 的参数、调度分支和 benchmark
8. Structured output / FSM decoding 为什么是 SGLang 的强项

## 每篇文章的固定格式 {#format}

每篇源码阅读都应该回答四个问题：

- 这个机制解决什么生产问题？
- 代码入口在哪里？
- 关键数据结构如何变化？
- 用什么指标证明它确实影响 TTFT、TPOT、throughput 或显存？

这样读源码才会服务于岗位要求里的 profiling、瓶颈分析和工程落地，而不是停在“看过某个类”的层面。
