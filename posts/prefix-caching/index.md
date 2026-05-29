
## the repeated-prefix problem {#problem}

the [KV cache]({{< relref "kv-cache" >}}) eliminates redundant computation *within* a single request. but production LLM serving has a second kind of redundancy: **many different requests begin with the same tokens**.

three workload shapes make this especially common:

**system prompts.** every request to a code assistant, agent, or customer-facing chatbot may begin with the same multi-kilotoken instruction block. without reuse, the server re-runs prefill over those identical tokens on every request, then throws the resulting KV cache away when the request ends.

**few-shot examples and RAG context.** retrieval-augmented generation often prepends retrieved documents before the user's question. when retrieval is deterministic or many users ask about the same hot document, those context tokens are recomputed repeatedly.

**multi-turn conversations.** each turn re-processes all previous turns. in a five-turn conversation, turn 1 is processed five times, turn 2 four times, and so on.

**prefix caching** — also called *KV cache reuse* — addresses all three cases with one mechanism: compute the KV vectors for a prefix once, store them, and let future requests that begin with the same tokens reuse those vectors directly.

the important boundary is this: prefix caching saves **prefill** work for tokens that already have valid KV vectors. it does not remove the need for later tokens to attend to that prefix, and it does not make decode cheaper.

## block-level reuse mechanism {#mechanism}

### chained block hashing {#hashing}

[paged attention]({{< relref "paged-attention" >}}) divides the KV cache into fixed-size blocks, typically 16 tokens. prefix caching uses the same granularity: each logical block gets a cache key, and a global table maps that key to a physical KV block.

the hash for block \\(i\\) is computed over the *full prefix up to and including* that block, not only over the block's own tokens:

```python
key_0 = hash(tokens[0 : B])
key_1 = hash(key_0 || tokens[B : 2B])
key_2 = hash(key_1 || tokens[2B : 3B])
...
```

where \\(B\\) is the block size and `||` denotes concatenation. this chained hash ensures that two sequences that differ in an early block produce different keys for all later blocks, even if a later block contains identical token ids.

{{< alert theme="info" >}}

why chain the hash? consider two prompts: *"The capital of France is Paris. What is 2+2?"* and *"The capital of Spain is Paris. What is 2+2?"* the final block `"What is 2+2?"` is identical, but its KV vectors differ because the tokens attended to different earlier context. the chained hash correctly distinguishes them.

{{< /alert >}}

### lookup and allocation {#lookup}

when a new request arrives, the scheduler walks its prompt block by block:

```text
new request prompt: [sys_prompt_block_0 | sys_prompt_block_1 | user_query_block]

  block 0: compute key_0 = hash(tokens[0:16])
           -> cache HIT  -> reuse physical block #3 (ref_count++)
  block 1: compute key_1 = hash(key_0 || tokens[16:32])
           -> cache HIT  -> reuse physical block #7 (ref_count++)
  block 2: compute key_2 = hash(key_1 || tokens[32:48])
           -> cache MISS -> allocate new physical block, run prefill for these tokens
```

cached blocks are inserted directly into the request's block table. the attention kernel reads them like any other block, so no KV data is copied and **the projection work for cached prefix tokens is skipped entirely**.

{{< figure src="/images/posts/prefix-caching/block-hash-lookup.svg" caption="<span class=\"figure-number\">Figure 1: </span>chained hash lookup for a 3-block prompt. blocks 0 and 1 (system prompt) hit the cache and are reused; block 2 (unique user query) misses and triggers prefill for 16 tokens only. the block table maps all three into a coherent KV sequence." width="100%" >}}

after the uncached suffix finishes prefill, its physical blocks can be inserted into the prefix cache for future reuse.

### why paged attention makes this cheap {#connection}

prefix caching is a natural extension of paged attention's block table:

```text
Request A (completed):
  block table: [Block #3: sys_p0] -> [Block #7: sys_p1] -> [Block #12: turn1] -> [Block #9: turn2]

Request B (new, same system prompt):
  block table: [Block #3: sys_p0] -> [Block #7: sys_p1] -> [Block #18: new_query]
               (shared, ref_count=2) (shared, ref_count=2) (new allocation)
```

shared blocks are protected by reference counting: they cannot be evicted while any live request holds a reference. multiple requests literally read from the same physical GPU memory locations, which is why prefix caching can be implemented as metadata reuse instead of a copying scheme.

## performance and cache policy {#performance-and-policy}

### compute savings {#benefits}

let \\(n_s\\) be the shared system-prefix length, \\(n_q\\) the user-query length, and \\(R\\) the number of requests.

**without prefix caching**, every request pays full prefill cost. the quadratic term dominates for long prompts:

$$
C_{\text{no cache}} = R \cdot O\bigl((n_s + n_q)^2 \cdot d\bigr)
$$

**with prefix caching** and a 100% hit rate, the system prompt KV is built once and only the user-query suffix is prefilled per request:

$$
C_{\text{cached}} = \underbrace{O(n_s^2 \cdot d)}_{\text{build cache once}} + R \cdot O\bigl(n_q^2 \cdot d + n_q \cdot n_s \cdot d\bigr)
$$

the \\(n_q \cdot n_s \cdot d\\) term remains because the query tokens still attend to the cached system-prompt keys. prefix caching skips computing K/V for the prefix again; it does not remove attention from the suffix to the prefix.

for a typical RAG or agent setup with \\(n_s = 4096\\), \\(n_q = 128\\), and many requests:

$$
\text{prefill compute saved} \approx 1 - \frac{n_q}{n_s + n_q} = 1 - \frac{128}{4224} \approx 97\%
$$

TTFT drops by a similar factor because the request moves from "prefill 4224 tokens" to "prefill 128 tokens plus attend to cached prefix keys."

### eviction and pinning {#eviction}

GPU memory is finite. when the prefix cache is full, blocks must be evicted, and evicting a hot long prefix can force an expensive full prefill on the next request.

**LRU (least recently used)** is the usual default. vLLM maintains a free-block LRU queue: blocks whose reference count drops to zero enter the queue tail, and the allocator takes from the queue head when it needs memory. blocks still referenced by live requests are immune to eviction.

**pinning high-frequency prefixes** is a common production tuning. operators identify the top-k system prompts by hit count and mark their blocks as non-evictable, preventing cache thrashing when a high-traffic prompt is briefly inactive.

**minimum hold time** handles a subtle LRU corner case. if a long system prompt fills a large fraction of the cache but requests for it are infrequent, a flood of unrelated prefixes can evict it before reuse. keeping newly computed blocks resident for at least \\(T\\) seconds gives expensive prefixes a chance to be hit.

## partial matches and scheduler composition {#partial-matches-and-composition}

### radix tree lookup {#radix-tree}

SGLang replaces the flat hash table with a **radix tree** (also called a trie or prefix tree) over token sequences. the tree structure makes partial prefix matching fast and natural:

```text
Root
├── [sys_block_0, sys_block_1]           <- system prompt blocks
│   ├── [query_A_block]  -> Req A KV
│   ├── [query_B_block]  -> Req B KV
│   └── [turn1_block, turn2_block]       <- multi-turn conversation
│       └── [turn3_block]                -> Req C KV (3 turns)
└── [other_prefix_block]                 -> different system prompt
```

to look up a new request, walk the tree from the root and match blocks until the tokens diverge. the traversed path is the **longest cached prefix**; those blocks are hits, and the remaining suffix needs prefill.

the radix tree has two advantages over a flat hash table:

- **single traversal** finds the longest matching prefix, instead of hashing and probing each block independently
- **explicit structural sharing** makes common prefixes visible in the data structure itself

### interaction with chunked prefill {#interaction}

prefix caching and [chunked prefill]({{< relref "chunked-prefill" >}}) compose cleanly. cached blocks are skipped before the chunked schedule begins, so the effective prompt length that chunked prefill must process is only the uncached suffix:

```text
prompt: [1024 cached tokens][512 uncached tokens]

chunked prefill sees only: 512 tokens
  iter 1: [tokens 0-511 prefill]  (only 1 chunk needed)
  iter 2: [decode]
```

with both optimizations active:

- prefix caching eliminates prefill for the cached portion
- chunked prefill interleaves the remaining prefill with decode
- the result is minimal TTFT and minimal TPOT interference

## when prefix caching helps {#when-it-helps}

prefix caching is highly effective when requests share long prefixes and much less useful when traffic has no prefix locality.

**high-benefit scenarios:**

- long system prompts shared across many requests, as in RAG, agents, and code assistants
- multi-turn conversations where the history grows each turn
- batch inference with a shared prompt template

**low or zero benefit:**

- every request has a unique prefix, such as random user documents used as context
- the shared prefix is shorter than one block, so there is nothing useful to cache at block granularity
- the request rate is high but spread across many different system prompts, causing cache thrashing
- the bottleneck is decode, not prefill, so prefix caching cannot improve TPOT

the last point is the key operational caveat: prefix caching only saves *prefill* work. once the request enters decode, it generates tokens one at a time and reads the KV cache each step. that decode cost is identical whether the prompt KV came from cache or from live computation.

## summary {#summary}

prefix caching exploits a structural property of production LLM workloads: many requests share long prefixes. it turns that structure into a systems optimization:

- **chained block hashes** identify cached KV blocks by the entire prefix state, not just local token ids
- **zero-copy sharing** reuses physical KV blocks through paged attention's reference-counted block table
- **cache policy** keeps hot prefixes resident through LRU, pinning, and hold-time rules
- **radix trees** make longest-prefix matching efficient when partial reuse matters

the payoff is large when prefixes are long and repeated. for a 4096-token system prompt with 128-token queries, about 97% of prefill projection work can be eliminated, which directly reduces TTFT for cache hits.

even with paged attention, continuous batching, chunked prefill, and prefix caching, prefill and decode still compete for the same GPU. the next step is to separate them entirely. that is the idea behind [disaggregated prefill]({{< relref "disaggregated-prefill" >}}).
