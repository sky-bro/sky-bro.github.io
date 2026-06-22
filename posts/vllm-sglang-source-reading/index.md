
This series is for source reading and engineering follow-through. The goal is not to translate files line by line, but to locate core inference-engine mechanisms in real code paths and verify their behavior with benchmarks or small experiments.

## Reading Order {#reading-order}

Start with the three core posts in request path -> scheduling decision -> GPU execution order:

1. [Request lifecycle: from OpenAI API to one forward pass]({{< relref "request-lifecycle-openai-to-forward-pass" >}})
2. [vLLM Scheduler: How Request Queues Become SchedulerOutput]({{< relref "scheduler-request-queue-to-scheduler-output" >}})
3. [vLLM ModelRunner: How SchedulerOutput Becomes a GPU Forward]({{< relref "model-runner-scheduler-output-to-gpu-forward" >}})
4. [Inference sampling: temperature, top-p, and top-k]({{< relref "llm-sampling-temperature-top-p-top-k" >}})
5. vLLM architecture map: from core serving engine to vLLM-Omni (draft)
6. vLLM Block Manager: from logical blocks to physical KV blocks
7. SGLang Radix Cache: why prefix reuse wants a tree
8. What a prefix cache hit actually saves
9. Chunked prefill parameters, scheduling branches, and benchmarks
10. Why structured output / FSM decoding is a strong SGLang use case

## Standard Format {#format}

Each source-reading post should answer four questions:

- What production problem does this mechanism solve?
- Where is the code entry point?
- How do the key data structures change?
- Which metric proves the behavior affects TTFT, TPOT, throughput, or memory?

That keeps source reading tied to the job requirements: profiling, bottleneck analysis, and engineering delivery, not just recognizing class names.
