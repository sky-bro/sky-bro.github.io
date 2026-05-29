
## 显存管理问题 {#memory-management-problem}

### 碎片化问题 {#fragmentation}

上一篇 [KV cache]({{< relref "kv-cache" >}}) 解释了为什么自回归解码可以缓存 key 和 value。KV cache 帮我们避免了大量重复计算，但也引出了一个新的系统问题：**这些不断增长的缓存到底放在哪里？**

最朴素的做法是：为每个请求分配一整块连续 GPU 显存，大小按它可能生成的最大长度来预留。但真实服务里，我们事先不知道输出会有多长，所以只能猜一个上限，为最坏情况留空间。

这会带来两类浪费：

**内部碎片。** 每个请求都按最大输出长度预留缓存，但多数请求会更早结束。假设预留了 2048 个 token 的 KV cache，请求实际在 300 个 token 结束，那么剩下 1748 个 token 槽位在整个请求生命周期里都空着。

**外部碎片。** 请求在不同时间结束，它们释放的连续块会散落在显存池里。之后的新长请求需要一块大的**连续**区域。即使总空闲显存足够，也可能没有单个连续区域能容纳它。

```text
GPU memory (naive allocation):

[Request A: ████████░░░░░░░░░░░░░░░]  ← 1/3 used, 2/3 wasted
[Request B: ████████████░░░░░░░░░░░]  ← 1/2 used, 1/2 wasted
[Request C: ██░░░░░░░░░░░░░░░░░░░░░]  ← early finish, lots of waste
[          free (fragmented)         ]
[Request D: ????  can't fit! total   ]
[           free is enough, but no   ]
[           contiguous block exists  ]
```

{{< figure src="/images/posts/paged-attention/memory-fragmentation.svg" caption="<span class=\"figure-number\">Figure 1: </span>朴素连续分配（左）会因为内部碎片和外部碎片浪费大量显存；paged attention（右）把固定大小 block 分散放入共享池，将显存利用率推到约 96%。" width="100%" >}}

vLLM 论文（Kwon et al., 2023）测到，朴素实现的 GPU 显存利用率只有 **20-38%**。其余显存主要浪费在碎片上。

### 借鉴操作系统分页 {#os-analogy}

操作系统早就解决过类似问题：进程不需要拿到一整块连续物理内存。物理内存被切成固定大小的 **frame**，进程看到的是由固定大小 **page** 组成的虚拟地址空间，再由 **page table** 把虚拟页映射到真实物理 frame。

物理 frame 不需要连续。只要映射表正确，分散的物理页也能组成连续的虚拟地址空间。

Paged attention 把这个思想搬到了 KV cache 管理里：

| OS 概念 | paged attention 中的对应物 |
| --- | --- |
| physical page frame | **KV block**：GPU 显存中的固定大小连续块 |
| virtual page | **logical block**：某个请求 KV 序列中的第 i 段 |
| page table | **block table**：每个请求自己的 logical -> physical 映射 |
| page fault | 向 block manager 申请一个新 block |

于是，我们不再为每个请求分配一块大的连续区域，而是把可用 GPU 显存切成固定大小 KV blocks（常见是每块 16 个 token）。请求需要多少就逐块申请，物理 block 可以来自显存池的任意位置。

## Paged attention 如何表示 KV cache {#data-structures}

### KV block {#the-kv-block}

每个 KV block 存储 `block_size` 个 token 的 key/value，覆盖所有 Transformer 层：

```text
KV Block #7  (block_size = 16):
  layer  0 ->  K[0..15] shape (16, d_head)
               V[0..15] shape (16, d_head)
  layer  1 ->  K[0..15], V[0..15]
  ...
  layer 31 ->  K[0..15], V[0..15]
```

block pool 在服务启动时预分配：

```text
Physical block pool (N blocks total):
[ B0 | B1 | B2 | B3 | B4 | B5 | B6 | B7 | ... | B_N ]
  A     A    A    B    free  C    A   free  ...
```

每个 block 要么空闲，要么属于某个请求；在共享场景下，也可以被多个请求或候选序列引用。

### block table {#the-block-table}

每个序列维护一张 block table，把 logical block index 映射到 physical block number。

```text
Request A (38 tokens generated, block_size = 16):

  logical block 0  ->  physical block  1   (full: 16 tokens)
  logical block 1  ->  physical block  5   (full: 16 tokens)
  logical block 2  ->  physical block  3   (6 tokens, partially filled)
```

要找到 token \\(t\\) 在哪里：`physical_block = block_table[t // block_size]`，block 内 offset 是 `t % block_size`。物理 block 不需要连续；block table 负责把它们拼成逻辑上连续的序列。

{{< figure src="/images/posts/paged-attention/block-table-mapping.svg" caption="<span class=\"figure-number\">Figure 2: </span>logical blocks 0、1、2 通过 block table 映射到显存池中分散的 physical blocks；attention kernel 在运行时根据这张表间接寻址。" width="100%" >}}

### block manager {#the-block-manager}

中心化的 block manager 负责追踪哪些 block 空闲、共享 block 的引用计数，以及分配和释放：

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

## 分页上的注意力与共享 {#attention-and-sharing}

### 对非连续 blocks 做 attention {#paged-attention-kernel}

标准 attention 通常假设 \\(K\\) 和 \\(V\\) 是连续张量。Paged attention 改变了这个假设：KV pairs 分散在多个 blocks 中，attention kernel 需要按 block table 把它们 gather 起来。

计算上，这等价于把 key/value 序列切成多个 blocks，并用 **online softmax** 增量合并结果：

```text
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

这种 online softmax 让每个 block 可以独立加载。CUDA kernel 不需要先把 KV 拼成连续大张量，而是在寄存器中维护 softmax 统计量，逐块累积输出。

数学上，它和对拼接后的完整序列做标准 attention 等价。关键恒等式是：

$$\text{softmax}([a_0, a_1, \ldots]) \cdot [V_0; V_1; \ldots] = \text{OnlineSoftmax}(\{(a_j, V_j)\}_j)$$

其中 `OnlineSoftmax` 表示上面的增量合并过程。

### 并行采样中的 copy-on-write {#cow}

并行采样会从同一个 prompt 生成多个独立候选，例如 beam search 或 \\(n > 1\\) 的采样。没有 paged attention 时，每个候选都需要一份完整 prompt KV cache，显存消耗会随候选数线性放大。

block 级共享让所有候选可以共享 prompt 对应的 physical blocks，并用引用计数记录共享关系：

```text
Parallel sampling (3 candidates from the same prompt):

  logical block 0  ->  physical block 2  (ref_count = 3, shared)
  logical block 1  ->  physical block 6  (ref_count = 3, shared)

  Candidate 1 ->  block 9   (ref_count = 1, exclusive)
  Candidate 2 ->  block 11  (ref_count = 1, exclusive)
  Candidate 3 ->  block 14  (ref_count = 1, exclusive)
```

当某个候选需要写入一个仍然被共享的 block 时，就触发 **copy-on-write**：

1. 分配一个新的空闲 block；
2. 用 `gpu_memcpy` 把共享 block 内容复制到新 block；
3. 原 block 的 `ref_count` 减 1；
4. 更新该序列的 block table，让它指向新 block；
5. 把新 token 写入这个已独占的新 block。

{{< figure src="/images/posts/paged-attention/cow-sharing.svg" caption="<span class=\"figure-number\">Figure 3: </span>三个采样候选以零额外显存共享 prompt blocks（ref_count = 3）；当 candidate 2 需要写入共享 block 时，copy-on-write 会先分配一个新的独占 block。" width="100%" >}}

这和 Linux 在 `fork()` 后处理 copy-on-write 的方式非常接近。

### prefix caching 自然出现 {#prefix-caching}

block 级 KV 管理还有一个很漂亮的性质：跨请求共享变得直接。如果两个请求有相同前缀，例如同一个 system prompt，那么这些前缀 blocks 的内容完全相同，因此可以共享同一批 physical blocks。

实现方式是：对每个 token block 的内容计算 hash，把 hash 作为全局 block cache 的 key。新请求进入时，按 block 遍历 prompt：hash 命中就复用已有 physical block 并增加引用计数；未命中就正常计算 KV，并把新 block 放入 cache。

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

被缓存的 blocks 可以跳过 prefill。attention kernel 读取它们时和读取普通 block 没有区别，只是这部分 KV 不需要重新 forward。对于一个 4096-token system prompt，如果 `block_size = 16`，就有 256 个 blocks 可以在所有共享该 prompt 的请求之间复用。[prefix caching 专文](/posts/prefix-caching/) 会进一步讨论 eviction policy 和 radix tree 如何让它在生产环境中可用。

## 为什么这对服务吞吐有意义 {#serving-impact}

### 显存利用率：30% -> 96% {#memory-utilization}

| 实现方式 | GPU 显存利用率 | 浪费原因 |
| --- | --- | --- |
| 朴素连续分配 | 20-38% | 最大长度预留 + 外部碎片 |
| paged attention | ~96% | 每个请求最多只有最后一个未填满 block 产生内部碎片 |

Paged attention 最坏情况下的内部碎片大约是 \\(\text{block\_size} / 2\\) 个 token（平均最后一个 block 半满）。如果 `block_size = 16`，平均只浪费 8 个 token 的槽位。相比朴素连续分配浪费的几百到几千个 token 槽位，这几乎可以忽略。

这不只是省显存。显存利用率从 30% 提到 96% 意味着同一张 GPU 上可以同时容纳大约 **3 倍** 的请求，直接提升并发和吞吐。

### 与 continuous batching 的关系 {#continuous-batching-connection}

Paged attention 不是孤立存在的。它天然适合 [continuous batching](/posts/continuous-batching/)：每个 decoding iteration 都可以加入新请求，而不是等整个 batch 结束。

continuous batching 会制造一个动态负载：请求不断进入、完成，KV blocks 不断分配和释放。block manager 可以优雅地处理这种变化：释放的 blocks 立即回到池里，新请求直接从池里拿 block，不需要 compaction，也不需要搬移已有请求。block table 的间接寻址吸收了物理位置变化。

当 block pool 完全耗尽时，scheduler 可以**抢占**低优先级请求：要么把它们的 KV blocks swap 到 CPU 内存（代价高，PCIe 带宽大约几十 GB/s），要么直接丢弃 KV cache，稍后重新 prefill（对短 prompt 通常更简单也更快）。

## 总结 {#summary}

Paged attention 的核心洞察可以浓缩成一句话：**KV cache 不需要物理连续，只需要可寻址。**

它从操作系统虚拟内存中借来了几件武器：

- **block table** 吸收物理不连续；
- **online softmax** 让 attention kernel 可以从分散地址 gather blocks；
- **引用计数** 支持并行候选和共享前缀请求之间的 zero-copy sharing；
- **细粒度 block 分配** 消除外部碎片，每个请求只剩最多半个 block 的内部碎片。

结果是：GPU 显存利用率从约 30% 提升到约 96%，同样硬件上可以承载约 3 倍并发请求。这也是 vLLM 以及现代 LLM serving engine 的基础显存管理技术。

当 KV cache 显存问题被控制住后，下一个问题就是：请求不断进入和完成时，如何让 GPU 始终保持忙碌？这就是 [continuous batching](/posts/continuous-batching/) 要解决的问题。
