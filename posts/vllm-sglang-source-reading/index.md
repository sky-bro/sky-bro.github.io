
This series is for source reading and engineering follow-through. The goal is not to translate files line by line, but to locate core inference-engine mechanisms in real code paths and verify their behavior with benchmarks or small experiments.

## Reading Order {#reading-order}

Planned posts will follow the request lifecycle:

1. Request lifecycle: from OpenAI API to one forward pass
2. Scheduler loop: waiting queue, running queue, token budget, and decode priority
3. vLLM Block Manager: from logical blocks to physical KV blocks
4. SGLang Radix Cache: why prefix reuse wants a tree
5. What a prefix cache hit actually saves
6. Chunked prefill parameters, scheduling branches, and benchmarks
7. Why structured output / FSM decoding is a strong SGLang use case

## Standard Format {#format}

Each source-reading post should answer four questions:

- What production problem does this mechanism solve?
- Where is the code entry point?
- How do the key data structures change?
- Which metric proves the behavior affects TTFT, TPOT, throughput, or memory?

That keeps source reading tied to the job requirements: profiling, bottleneck analysis, and engineering delivery, not just recognizing class names.
