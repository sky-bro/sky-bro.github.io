
## the repeated-prefix problem {#problem}

the [KV cache]({{< relref "kv-cache" >}}) eliminates redundant computation *within* a single request. but in production, a different kind of redundancy is pervasive: **many requests share the same prefix**.

three scenarios account for the majority of production traffic:

**system prompts.** every request to a code assistant, agent, or customer-facing chatbot begins with the same multi-kilotoken system prompt. on each new request, the server re-runs prefill over those identical tokens — then throws away the resulting KV cache when the request ends.

**few-shot examples and RAG context.** retrieval-augmented generation prepends retrieved documents before the user's question. if the top-retrieved documents are the same across requests (common when retrieval is deterministic), their KV pairs are recomputed from scratch every time.

**multi-turn conversations.** each turn re-processes all previous turns. a five-turn conversation processes turn 1 five times, turn 2 four times, and so on.

**prefix caching** — also called *KV cache reuse* — addresses all three cases with a single mechanism: compute the KV vectors for a prefix once, store them, and serve them to any future request that begins with the same tokens.

## the mechanism {#mechanism}

### block-level hashing {#hashing}

[paged attention]({{< relref "paged-attention" >}}) divides the KV cache into fixed-size blocks (typically 16 tokens). prefix caching operates at exactly this granularity: each block gets a hash key, and a global table maps hash → physical block.

the hash for block \\(i\\) is computed over the *full prefix up to and including* that block — not just the block's own tokens:

```python
key_0 = hash(tokens[0 : B])
key_1 = hash(key_0 || tokens[B : 2B])
key_2 = hash(key_1 || tokens[2B : 3B])
...
```

where \\(B\\) is block size and `||` denotes concatenation. this chained hash ensures that two sequences that differ in an early block produce different hash keys for all subsequent blocks, even if those later blocks happen to contain the same tokens.

{{< alert theme="info" >}}

why chain the hash? consider two prompts: *"The capital of France is Paris. What is 2+2?"* and *"The capital of Spain is Paris. What is 2+2?"* — the final block `"What is 2+2?"` is identical, but its KV vectors will differ because they attended to different prior context. the chained hash correctly distinguishes them.

{{< /alert >}}

### lookup and allocation {#lookup}

when a new request arrives, the scheduler walks its prompt block-by-block:

```
new request prompt: [sys_prompt_block_0 | sys_prompt_block_1 | user_query_block]

  block 0: compute key_0 = hash(tokens[0:16])
           → cache HIT  → reuse physical block #3 (ref_count++)
  block 1: compute key_1 = hash(key_0 || tokens[16:32])
           → cache HIT  → reuse physical block #7 (ref_count++)
  block 2: compute key_2 = hash(key_1 || tokens[32:48])
           → cache MISS → allocate new physical block, run prefill for these tokens
```

cached blocks are inserted directly into the request's block table — the attention kernel reads them just like any other block, at no extra cost. **the prefill computation for cached blocks is skipped entirely.**

{{< figure src="/images/posts/prefix-caching/block-hash-lookup.svg" caption="<span class=\"figure-number\">Figure 1: </span>chained hash lookup for a 3-block prompt. blocks 0 and 1 (system prompt) hit the cache and are reused; block 2 (unique user query) misses and triggers prefill for 16 tokens only. the block table maps all three into a coherent KV sequence." width="100%" >}}

after the new (uncached) suffix finishes prefill, its physical blocks are written to the prefix cache for future reuse.

### connection to paged attention {#connection}

prefix caching is trivially composable with paged attention's block table:

```
Request A (completed):
  block table: [Block #3: sys_p0] → [Block #7: sys_p1] → [Block #12: turn1] → [Block #9: turn2]

Request B (new, same system prompt):
  block table: [Block #3: sys_p0] → [Block #7: sys_p1] → [Block #18: new_query]
               (shared, ref_count=2) (shared, ref_count=2) (new allocation)
```

shared blocks are protected by reference counting — they won't be evicted while any live request holds a reference. no data is copied; multiple requests literally read from the same physical GPU memory locations.

## quantifying the benefit {#benefits}

let \\(n_s\\) be the system prompt length (in tokens), \\(n_q\\) the user query length, and \\(R\\) the number of requests.

**without prefix caching**, each request pays full prefill cost. the quadratic term dominates for long prompts:

\begin{equation}
C_{\text{no cache}} = R \cdot O\bigl((n_s + n_q)^2 \cdot d\bigr)
\end{equation}

**with prefix caching** (100% hit rate), only the user query is prefilled per request. the system prompt KV is computed once:

\begin{equation}
C_{\text{cached}} = \underbrace{O(n_s^2 \cdot d)}_{\text{build cache once}} + R \cdot O\bigl(n_q^2 \cdot d + n_q \cdot n_s \cdot d\bigr)
\end{equation}

the second term has \\(n_q \cdot n_s \cdot d\\) because the user query's queries must still attend to the (cached) system prompt keys — that attention operation still runs. only the *projection* of system prompt tokens is skipped.

for typical RAG deployments with \\(n_s = 4096\\), \\(n_q = 128\\), and many requests:

\begin{equation}
\text{prefill compute saved} \approx 1 - \frac{n_q}{n_s + n_q} = 1 - \frac{128}{4224} \approx 97\%
\end{equation}

TTFT drops by a similar factor — from "prefill 4224 tokens" to "prefill 128 tokens."

## eviction policy {#eviction}

GPU memory is finite. when the cache is full, blocks must be evicted. the challenge: an evicted block that gets hit on the next request forces a full prefill — expensive for long system prompts.

**LRU (least recently used)** is the standard default. vLLM maintains a free-block LRU queue: blocks whose reference count drops to zero enter the queue tail. when the allocator needs a block, it takes from the queue head (oldest). blocks still referenced by live requests are immune to eviction.

**pinning high-frequency prefixes** is a common production tuning: identify the top-k system prompts by hit count and mark their blocks as non-evictable. this prevents cache thrashing when a high-traffic system prompt is briefly unpopular.

**the LRU corner case.** if a long system prompt fills the entire cache but requests are infrequent, a flood of requests with different prefixes can evict the system prompt entirely. the fix: implement a *minimum hold time* — newly computed blocks stay in cache for at least \\(T\\) seconds before becoming eligible for eviction.

## radix tree: efficient partial-match lookup {#radix-tree}

SGLang replaces the flat hash table with a **radix tree** (also called a trie or prefix tree) over token sequences. the tree structure makes partial prefix matching fast and natural:

```
Root
├── [sys_block_0, sys_block_1]           ← system prompt blocks
│   ├── [query_A_block]  → Req A KV
│   ├── [query_B_block]  → Req B KV
│   └── [turn1_block, turn2_block]       ← multi-turn conversation
│       └── [turn3_block]               → Req C KV (3 turns)
└── [other_prefix_block]                → different system prompt
```

to look up a new request: walk the tree from the root, matching blocks as long as they match the request's tokens. when the walk terminates (either at a leaf or at a diverging branch), the path traversed is the **longest cached prefix**. all blocks on that path are cache hits; the remaining blocks need prefill.

the radix tree has two advantages over a flat hash table:
1. **single traversal** finds the longest matching prefix — no need to hash each block independently
2. **structural sharing** is explicit — two requests sharing a common prefix will literally share a subtree

## when prefix caching doesn't help {#limitations}

prefix caching is highly effective in specific workload shapes and irrelevant in others.

**high-benefit scenarios:**
- long system prompts shared across many requests (RAG, agents, code assistants)
- multi-turn conversations where the history grows each turn
- batch inference with a shared prompt template

**low or zero benefit:**
- every request has a unique prefix (random user documents as context)
- the shared prefix is shorter than one block (< 16 tokens) — nothing to cache at block granularity
- very high request rate with many different system prompts — cache thrashing evicts blocks before they can be reused
- the bottleneck is decode, not prefill — prefix caching has no effect on TPOT

the second point is subtle but important: prefix caching only saves *prefill* work. once the request enters decode, it generates tokens one at a time, reading the KV cache each step — that cost is identical whether the prompt KV came from cache or from live computation.

## interaction with chunked prefill {#interaction}

prefix caching and [chunked prefill]({{< relref "chunked-prefill" >}}) compose cleanly. cached blocks are skipped before the chunked schedule even begins, so the effective "prompt length" that chunked prefill must process is only the uncached suffix:

```
prompt: [1024 cached tokens][512 uncached tokens]

chunked prefill sees only: 512 tokens
  iter 1: [tokens 0–511 prefill]  (only 1 chunk needed)
  iter 2: [decode]
```

with both optimizations active:
- prefix caching eliminates prefill for the cached portion
- chunked prefill interleaves the remaining prefill with decode
- the result: minimal TTFT *and* minimal TPOT interference

## summary {#summary}

prefix caching exploits a structural property of production LLM workloads — many requests share long prefixes — and turns that structure into a performance advantage:

- **block-level hashing** (chained, to capture prefix identity) maps token sequences to cached KV blocks
- **zero-copy sharing** via paged attention's reference-counted block table — multiple requests read the same physical GPU memory
- **radix tree** (SGLang) enables single-pass longest-prefix-match lookup across the entire cache
- **LRU eviction** with pinning for high-frequency prefixes keeps the cache warm

the payoff: for a 4096-token system prompt with 128-token queries, ~97% of prefill computation is eliminated, and TTFT drops by a corresponding factor. for long-running agent systems or RAG pipelines, this is one of the highest-leverage optimizations available.

even with all these optimizations — paged attention, continuous batching, chunked prefill, prefix caching — there is still a fundamental limit: prefill and decode compete for the same GPU. the next step is to separate them entirely. that's what [disaggregated prefill]({{< relref "disaggregated-prefill" >}}) does.
