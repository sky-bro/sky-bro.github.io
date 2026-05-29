
## 重复前缀问题 {#problem}

[KV cache]({{< relref "kv-cache" >}}) 解决的是**同一个请求内部**的重复计算：解码第 \\(t\\) 个 token 时，不需要重新计算前面 \\(t-1\\) 个 token 的 K、V。可是生产环境里还有另一种更大规模的重复：**不同请求经常以完全相同的一段 token 开头**。

最常见的是三类流量。

**system prompt。** 代码助手、agent、客服机器人往往都有一段几千 token 的固定系统提示词。没有跨请求复用时，每个新请求都要重新对这段提示词做 prefill，请求结束后又把算出来的 KV cache 丢掉。

**few-shot examples 和 RAG context。** RAG 会把检索到的文档放在用户问题前面。如果检索是确定性的，或者许多用户都在问同一个热点文档，那么这些上下文 token 会在不同请求里反复被计算。

**多轮对话。** 每一轮都要重新处理之前的历史。一个五轮对话里，第一轮会被处理五次，第二轮会被处理四次，以此类推。

**prefix caching**，也叫 *KV cache reuse*，用一个机制覆盖这些场景：某个前缀的 KV 向量只计算一次并存起来，之后任何以同一段 token 开头的请求都可以直接复用这批向量。

这里有一个重要边界：prefix caching 节省的是已有前缀 token 的 **prefill** 工作。后续 token 仍然需要 attend 到这段前缀，它也不会让 decode 阶段本身变便宜。

## 块级复用机制 {#mechanism}

### 链式块哈希 {#hashing}

[paged attention]({{< relref "paged-attention" >}}) 把 KV cache 切成固定大小的块，常见块大小是 16 个 token。prefix caching 直接沿用这个粒度：每个逻辑块生成一个 cache key，全局表把这个 key 映射到一个物理 KV block。

第 \\(i\\) 个块的哈希不是只看这个块自己的 token，而是覆盖**从序列开头到这个块为止的完整前缀**：

```python
key_0 = hash(tokens[0 : B])
key_1 = hash(key_0 || tokens[B : 2B])
key_2 = hash(key_1 || tokens[2B : 3B])
...
```

其中 \\(B\\) 是 block size，`||` 表示拼接。这个链式哈希保证：只要早期某个块不同，后面所有块的 key 都会不同，即使后面的 token id 恰好完全一样。

{{< alert theme="info" >}}

为什么必须链式哈希？考虑两个 prompt：*"The capital of France is Paris. What is 2+2?"* 和 *"The capital of Spain is Paris. What is 2+2?"*。最后的 `"What is 2+2?"` 这一块 token 完全相同，但它的 KV 向量不同，因为它 attend 过的前文不同。链式哈希可以正确区分这两种上下文。

{{< /alert >}}

### 查找与分配 {#lookup}

当新请求到达时，scheduler 会按块扫描 prompt：

```text
new request prompt: [sys_prompt_block_0 | sys_prompt_block_1 | user_query_block]

  block 0: compute key_0 = hash(tokens[0:16])
           -> cache HIT  -> reuse physical block #3 (ref_count++)
  block 1: compute key_1 = hash(key_0 || tokens[16:32])
           -> cache HIT  -> reuse physical block #7 (ref_count++)
  block 2: compute key_2 = hash(key_1 || tokens[32:48])
           -> cache MISS -> allocate new physical block, run prefill for these tokens
```

命中的块会被直接插入这个请求的 block table。attention kernel 读取它们的方式和读取普通块一样，因此不需要复制 KV 数据，并且**命中前缀 token 的投影计算会被完全跳过**。

{{< figure src="/images/posts/prefix-caching/block-hash-lookup.svg" caption="<span class=\"figure-number\">Figure 1: </span>一个 3-block prompt 的链式哈希查找。block 0 和 block 1 是 system prompt，命中 cache 后被复用；block 2 是当前用户问题，miss 后只需要对这 16 个 token 做 prefill。block table 会把三者组织成连续的 KV 序列。" width="100%" >}}

未命中的后缀完成 prefill 之后，它的物理 block 也可以写入 prefix cache，供后续请求复用。

### 为什么 paged attention 让复用变便宜 {#connection}

prefix caching 和 paged attention 的 block table 天然配合：

```text
Request A (completed):
  block table: [Block #3: sys_p0] -> [Block #7: sys_p1] -> [Block #12: turn1] -> [Block #9: turn2]

Request B (new, same system prompt):
  block table: [Block #3: sys_p0] -> [Block #7: sys_p1] -> [Block #18: new_query]
               (shared, ref_count=2) (shared, ref_count=2) (new allocation)
```

共享块通过引用计数保护：只要还有活跃请求引用某个物理块，它就不能被淘汰。多个请求实际上读取的是同一块 GPU 物理显存，所以 prefix caching 更像是元数据复用，而不是复制一份缓存。

## 性能收益与缓存策略 {#performance-and-policy}

### 计算节省 {#benefits}

设 \\(n_s\\) 是共享 system prefix 的长度，\\(n_q\\) 是用户 query 的长度，\\(R\\) 是请求数。

**没有 prefix caching** 时，每个请求都要支付完整 prefill 成本。长 prompt 下二次项会主导：

$$
C_{\text{no cache}} = R \cdot O\bigl((n_s + n_q)^2 \cdot d\bigr)
$$

**有 prefix caching** 且命中率为 100% 时，system prompt 的 KV 只构建一次；每个请求只需要 prefill 用户 query 这段后缀：

$$
C_{\text{cached}} = \underbrace{O(n_s^2 \cdot d)}_{\text{build cache once}} + R \cdot O\bigl(n_q^2 \cdot d + n_q \cdot n_s \cdot d\bigr)
$$

其中 \\(n_q \cdot n_s \cdot d\\) 仍然存在，因为 query token 还是要 attend 到缓存里的 system-prompt keys。prefix caching 跳过的是再次为前缀计算 K/V 的工作，而不是取消后缀对前缀的注意力。

对一个典型 RAG 或 agent 场景，假设 \\(n_s = 4096\\)、\\(n_q = 128\\)，并且请求数很多：

$$
\text{prefill compute saved} \approx 1 - \frac{n_q}{n_s + n_q} = 1 - \frac{128}{4224} \approx 97\%
$$

TTFT 会接近同比例下降，因为请求从“prefill 4224 个 token”变成“prefill 128 个 token，并让它们 attend 到已缓存的前缀 keys”。

### 淘汰与 pinning {#eviction}

GPU 显存是有限的。prefix cache 满了以后必须淘汰块，而淘汰一个热门长前缀会让下一次命中机会变成一次昂贵的完整 prefill。

**LRU（least recently used）** 是常见默认策略。vLLM 维护一个 free-block LRU 队列：引用计数降为 0 的块进入队列尾部，allocator 需要显存时从队列头部取最久未使用的块。仍被活跃请求引用的块不能被淘汰。

**pinning 高频前缀** 是常见生产调优。系统可以按 hit count 找出 top-k system prompt，把对应 block 标记为不可淘汰，避免某个高流量 prompt 短暂空闲后被其他前缀冲掉。

**最短驻留时间** 可以处理一个微妙的 LRU 边界情况：如果一个长 system prompt 占了较大缓存空间，但请求频率不够高，一波不同前缀的请求可能在它被再次访问前把它淘汰。让新计算出的块至少驻留 \\(T\\) 秒，可以给昂贵前缀一次被复用的机会。

## 部分匹配与调度组合 {#partial-matches-and-composition}

### radix tree 查找 {#radix-tree}

SGLang 没有只用扁平哈希表，而是用 **radix tree**，也可以理解为 token 序列上的 trie / prefix tree。树结构天然适合做部分前缀匹配：

```text
Root
├── [sys_block_0, sys_block_1]           <- system prompt blocks
│   ├── [query_A_block]  -> Req A KV
│   ├── [query_B_block]  -> Req B KV
│   └── [turn1_block, turn2_block]       <- multi-turn conversation
│       └── [turn3_block]                -> Req C KV (3 turns)
└── [other_prefix_block]                 -> different system prompt
```

查找新请求时，从 root 开始沿着树匹配 block，直到 token 分叉。已经走过的路径就是**最长已缓存前缀**；这条路径上的 block 都是 cache hit，剩下的后缀需要 prefill。

radix tree 相比扁平哈希表有两个优势：

- **一次遍历**就能找到最长匹配前缀，不需要对每个 block 分别哈希和探测
- **结构化共享**会直接体现在数据结构里，共享前缀就是共享子树

### 与 chunked prefill 的关系 {#interaction}

prefix caching 和 [chunked prefill]({{< relref "chunked-prefill" >}}) 可以干净组合。缓存命中的块会在 chunked schedule 开始前被跳过，所以 chunked prefill 看到的有效 prompt 长度只剩下未命中的后缀：

```text
prompt: [1024 cached tokens][512 uncached tokens]

chunked prefill sees only: 512 tokens
  iter 1: [tokens 0-511 prefill]  (only 1 chunk needed)
  iter 2: [decode]
```

两者同时开启时：

- prefix caching 消除已缓存部分的 prefill
- chunked prefill 把剩余 prefill 和 decode 交错执行
- 结果是更低的 TTFT，同时减少对 TPOT 的干扰

## prefix caching 什么时候有用 {#when-it-helps}

prefix caching 对“长前缀重复出现”的流量非常有效；如果请求之间没有前缀局部性，它就帮不上太多。

**收益高的场景：**

- RAG、agent、代码助手这类大量请求共享长 system prompt 的服务
- 历史上下文随轮次增长的多轮对话
- 使用共享 prompt template 的批量推理

**收益低或没有收益的场景：**

- 每个请求都有独一无二的前缀，例如随机用户文档作为上下文
- 共享前缀短于一个 block，在块粒度下没有值得缓存的内容
- 请求速率很高但分散在许多不同 system prompt 上，导致 cache thrashing
- 瓶颈在 decode 而不是 prefill，因此 prefix caching 无法改善 TPOT

最后一点是最重要的运维边界：prefix caching 只节省 *prefill*。请求进入 decode 后，会逐 token 生成并在每一步读取 KV cache；这部分成本和 prompt KV 是来自缓存还是现场计算没有关系。

## 总结 {#summary}

prefix caching 利用了生产 LLM 流量的一个结构性事实：许多请求共享长前缀。它把这个事实转成系统优化：

- **链式块哈希**用完整前缀状态识别缓存块，而不是只看局部 token id
- **零拷贝共享**通过 paged attention 的引用计数 block table 复用物理 KV block
- **缓存策略**用 LRU、pinning 和最短驻留时间让热门前缀留在显存里
- **radix tree**在需要部分复用时高效找到最长匹配前缀

当共享前缀很长且重复出现时，收益会非常大。以 4096-token system prompt 和 128-token query 为例，大约 97% 的 prefill 投影计算可以被消除，这会直接降低 cache hit 请求的 TTFT。

即使有 paged attention、continuous batching、chunked prefill 和 prefix caching，prefill 与 decode 仍然在争抢同一块 GPU。下一步是把它们彻底分离，这就是 [disaggregated prefill]({{< relref "disaggregated-prefill" >}}) 要解决的问题。
