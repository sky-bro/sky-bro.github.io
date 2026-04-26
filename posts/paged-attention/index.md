
## the fragmentation problem {#fragmentation}

the [previous post]({{< relref "kv-cache" >}}) in this series explained *why* we cache key and value vectors during autoregressive decoding. by the end of that post, the KV cache was saving us enormous amounts of recomputation — but it quietly introduced a new problem: **where do you put all that memory?**

the naive answer is: allocate a contiguous block of GPU memory for each request, big enough to hold its maximum possible output length. but in practice, you don't know the output length in advance. so you guess — reserving space for the worst case.

this leads to two kinds of waste:

**internal fragmentation.** every request gets pre-allocated memory for the maximum output length, but most requests finish much earlier. if you reserve 2048 tokens of KV cache but the request ends at token 300, the remaining 1748 token slots sit empty for the entire duration.

**external fragmentation.** as requests finish at different times, they release their reserved blocks back to the free pool. soon the free memory is scattered in many disconnected chunks. a new long request needs a large *contiguous* block, but even if the total free memory is sufficient, no single contiguous region may be large enough to satisfy the allocation.

```
GPU memory (naive allocation):

[Request A: ████████░░░░░░░░░░░░░░░]  ← 1/3 used, 2/3 wasted
[Request B: ████████████░░░░░░░░░░░]  ← 1/2 used, 1/2 wasted
[Request C: ██░░░░░░░░░░░░░░░░░░░░░]  ← early finish, lots of waste
[          free (fragmented)         ]
[Request D: ????  can't fit! total   ]
[           free is enough, but no   ]
[           contiguous block exists  ]
```

{{< figure src="/images/posts/paged-attention/memory-fragmentation.svg" caption="<span class=\"figure-number\">Figure 1: </span>naive contiguous allocation (left) wastes most GPU memory through internal and external fragmentation; paged attention (right) scatters fixed-size blocks through a shared pool, achieving ~96% utilization." width="100%" >}}

the vLLM paper (Kwon et al., 2023) measured that naive implementations achieve only **20–38%** GPU memory utilization. the rest is wasted on fragmentation.

## borrowing from the OS {#os-analogy}

operating systems solved exactly this problem decades ago for process memory. the solution: **paging**.

instead of giving each process a contiguous chunk of physical memory, the OS divides physical memory into fixed-size *frames* and gives processes a virtual address space divided into same-size *pages*. a *page table* maps each virtual page to its physical frame. frames don't need to be contiguous — scattered free frames are assembled into a coherent virtual address space through the mapping.

paged attention applies this idea to KV cache management:

| OS concept | paged attention equivalent |
|---|---|
| physical page frame | **KV block** — a fixed-size contiguous chunk of GPU memory |
| virtual page | **logical block** — the i-th segment of a request's KV sequence |
| page table | **block table** — per-request mapping from logical to physical blocks |
| page fault | requesting a new block from the block manager |

instead of allocating one large contiguous region per request, we carve the entire available GPU memory into a pool of fixed-size KV blocks (typically holding 16 tokens each). requests are granted blocks one at a time as they need them — and those blocks can come from anywhere in the pool.

## data structures {#data-structures}

### the KV block

each KV block stores the key and value vectors for exactly `block_size` tokens (typically 16), for every transformer layer:

```
KV Block #7  (block_size = 16):
  layer  0 →  K[0..15] shape (16, d_head)
              V[0..15] shape (16, d_head)
  layer  1 →  K[0..15], V[0..15]
  ...
  layer 31 →  K[0..15], V[0..15]
```

the block pool is pre-allocated at startup:

```
Physical block pool (N blocks total):
[ B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7 | ... | B_N ]
  A     A    A    B    free  C    A   free  ...
```

each block is either *free* or assigned to exactly one request (or shared, as we'll see below).

### the block table

every sequence maintains a block table: a simple list mapping logical block index to physical block number.

```
Request A (38 tokens generated, block_size = 16):

  logical block 0  →  physical block  1   (full: 16 tokens)
  logical block 1  →  physical block  5   (full: 16 tokens)
  logical block 2  →  physical block  3   (6 tokens, partially filled)
```

to find where token \\(t\\) lives: `physical_block = block_table[t // block_size]`, offset within block = `t % block_size`. physical blocks need not be contiguous; the block table is what assembles them into a coherent sequence.

{{< figure src="/images/posts/paged-attention/block-table-mapping.svg" caption="<span class=\"figure-number\">Figure 2: </span>logical blocks 0, 1, 2 map via the block table to scattered physical blocks anywhere in the pool. the attention kernel dereferences this indirection at runtime." width="100%" >}}

### the block manager

a centralized block manager tracks which blocks are free, maintains reference counts for shared blocks, and handles allocation and deallocation:

```python
class BlockManager:
    def __init__(self, num_gpu_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_gpu_blocks))
        self.ref_count = [0] * num_gpu_blocks

    def allocate(self, seq) -> bool:
        needed = ceil(seq.prompt_len / self.block_size)
        if len(self.free_blocks) < needed:
            return False  # scheduler must preempt another request
        seq.block_table = []
        for _ in range(needed):
            bid = self.free_blocks.pop()
            self.ref_count[bid] = 1
            seq.block_table.append(bid)
        return True

    def append_slot(self, seq):
        last_token = seq.num_tokens - 1
        last_logical = last_token // self.block_size
        if last_logical >= len(seq.block_table):
            # current block is full, need a new one
            bid = self.free_blocks.pop()
            self.ref_count[bid] = 1
            seq.block_table.append(bid)
        else:
            phys = seq.block_table[last_logical]
            if self.ref_count[phys] > 1:
                self._copy_on_write(seq, last_logical)

    def free(self, seq):
        for bid in seq.block_table:
            self.ref_count[bid] -= 1
            if self.ref_count[bid] == 0:
                self.free_blocks.append(bid)
```

## attention over non-contiguous blocks {#paged-attention-kernel}

standard attention assumes \\(K\\) and \\(V\\) are contiguous tensors. paged attention modifies this assumption: KV pairs live in scattered blocks, and the attention kernel must gather them.

the computation is equivalent to splitting the key-value sequence into blocks and running *online softmax* to merge results incrementally:

```
m = -inf  (running max)
l = 0     (running normalizer)
o = 0     (running output)

for each logical block j:
    K_j, V_j = load_block(block_table[j])       # gather from physical memory
    a_j = (q @ K_j.T) / sqrt(d_k)               # shape: (1, block_size)

    m_j = max(a_j)
    l_j = sum(exp(a_j - m_j))
    o_j = exp(a_j - m_j) @ V_j

    # merge with running statistics (log-sum-exp trick)
    m_new = max(m, m_j)
    l = exp(m - m_new) * l + exp(m_j - m_new) * l_j
    o = exp(m - m_new) * o + exp(m_j - m_new) * o_j
    m = m_new

output = o / l
```

this online softmax formulation lets each block be fetched independently. there is no need to concatenate KV before running attention — the CUDA kernel loops over blocks (indexed via the block table) and accumulates the softmax-weighted output in registers.

mathematically, this produces exactly the same result as standard attention over a concatenated sequence. the key identity is:

\begin{equation}
\text{softmax}([a_0, a_1, \ldots]) \cdot [V_0; V_1; \ldots] = \text{OnlineSoftmax}(\{(a_j, V_j)\}_j)
\end{equation}

where `OnlineSoftmax` denotes the incremental merge above.

## copy-on-write for parallel sampling {#cow}

parallel sampling generates multiple independent outputs from the same prompt (e.g., beam search, or sampling with \\(n > 1\\)). without paged attention, each candidate would need its own full copy of the prompt's KV cache — wasting memory proportional to the number of candidates.

with block-level sharing, all candidates *share* the same physical blocks for the prompt, identified via reference counting:

```
Parallel sampling (3 candidates from the same prompt):

  logical block 0  →  physical block 2  (ref_count = 3, shared)
  logical block 1  →  physical block 6  (ref_count = 3, shared)

  Candidate 1 →  block 9   (ref_count = 1, exclusive)
  Candidate 2 →  block 11  (ref_count = 1, exclusive)
  Candidate 3 →  block 14  (ref_count = 1, exclusive)
```

when a candidate needs to *write* to a shared block (because the new token falls in a slot that is still shared), it triggers **copy-on-write**:

1. allocate a new free block
2. `gpu_memcpy` the shared block's content into the new block
3. decrement `ref_count` on the original block
4. update the sequence's block table to point to the new block
5. write the new token into the new block (now exclusively owned)

{{< figure src="/images/posts/paged-attention/cow-sharing.svg" caption="<span class=\"figure-number\">Figure 3: </span>three sampling candidates share prompt blocks (ref_count = 3) at zero memory cost. when candidate 2 writes a new token into a shared block, copy-on-write allocates a fresh exclusive block before writing." width="100%" >}}

this is exactly how Linux handles copy-on-write after `fork()`.

## prefix caching falls out naturally {#prefix-caching}

a beautiful property of block-level KV management is that *cross-request sharing* becomes straightforward. if two requests share the same prefix (e.g., the same system prompt), their blocks for that prefix will contain identical content — so they can literally share the same physical blocks.

the implementation: compute a hash of each block's token contents. use this hash as a key in a global block cache. on a new request, walk its prompt block-by-block: if a hash hits the cache, reuse the cached physical block (increment ref count); if it misses, compute the KV normally and populate the cache.

```python
def allocate_with_prefix_cache(self, seq):
    for i, token_block in enumerate(seq.prompt_blocks()):
        h = hash_block(token_block)
        if h in self.prefix_cache:
            phys = self.prefix_cache[h]
            self.ref_count[phys] += 1
            seq.block_table.append(phys)
            seq.cached_prefix_len += self.block_size
        else:
            phys = self.free_blocks.pop()
            self.ref_count[phys] = 1
            seq.block_table.append(phys)
            # will be populated during prefill
            self.prefix_cache[h] = phys
```

cached blocks skip prefill entirely — the attention kernel reads their KV pairs just like any other block, but no forward pass was needed to fill them. for a 4096-token system prompt with `block_size = 16`, that's 256 blocks that can be reused across all requests sharing that prompt. [a full post on prefix caching]({{< relref "prefix-caching" >}}) covers the eviction policies and radix-tree structure that make this practical at scale.

## memory utilization: 30% → 96% {#memory-utilization}

| implementation | GPU utilization | reason for waste |
|---|---|---|
| naive contiguous allocation | 20–38% | max-length reservation + external fragmentation |
| paged attention | ~96% | only the last (partially filled) block per request has internal fragmentation |

the worst case internal fragmentation with paged attention is \\(\text{block\_size} / 2\\) tokens per request (on average the last block is half full). with `block_size = 16`, that's 8 tokens of wasted space — negligible compared to the hundreds or thousands of tokens that naive allocation wastes.

this improvement is not just an efficiency win: it directly translates to **higher concurrency**. if GPU memory utilization goes from 30% to 96%, roughly **3× more requests** can be in flight simultaneously on the same hardware.

## connection to continuous batching {#continuous-batching-connection}

paged attention doesn't exist in isolation. it pairs naturally with [continuous batching]({{< relref "continuous-batching" >}}), which schedules new requests at every decoding iteration rather than waiting for an entire batch to finish.

continuous batching creates a dynamic workload: requests are constantly arriving and completing, KV cache blocks are being allocated and freed. the block manager handles this gracefully — freed blocks go back to the pool immediately, and new requests draw from the pool without any compaction or reshuffling. the block table indirection absorbs all the movement.

when the block pool runs out entirely, the scheduler can **preempt** low-priority requests by either swapping their KV blocks to CPU memory (expensive: ~32 GB/s PCIe bandwidth) or simply discarding them and marking them for re-prefill later (simpler, usually faster for short prompts).

## summary {#summary}

paged attention's core insight is one sentence: **KV cache doesn't need to be contiguous — it just needs to be addressable**.

by borrowing virtual memory from the OS:

- **block table** indirection absorbs physical non-contiguity
- **online softmax** lets the attention kernel gather blocks at scatter addresses
- **reference counting** enables zero-copy sharing among parallel candidates and across requests with common prefixes
- **fine-grained allocation** eliminates external fragmentation; only ≤ half a block of internal fragmentation remains per request

the payoff: GPU utilization climbs from ~30% to ~96%, enabling ~3× more concurrent requests on the same hardware. this is the foundational memory management technique behind vLLM and essentially every modern LLM serving engine.

with KV cache memory under control, the next question is how to keep the GPU busy while requests arrive and finish at different times — that's the problem [continuous batching]({{< relref "continuous-batching" >}}) solves.
