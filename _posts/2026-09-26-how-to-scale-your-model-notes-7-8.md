---
layout: post
title: "How To Scale Your Model 笔记(7-8)"
date: 2026-09-26
permalink: /posts/2026/09/how-to-scale-your-model-notes-7-8/
description: "从新 Query 与历史 KV 的数据流，推导推理的容量、延迟、吞吐和多卡布局，再连接 prefill、KV 交接与 decode 的服务速率。"
tags:
  - Machine Learning
  - Transformer
  - ML Systems
  - Learning Notes
related_posts: true
math: true
---

自回归推理反复执行同一件事：让已确定的输入经过模型，读取历史信息，再选择下一个 token。输入是否已经确定、历史保存在哪里、一次同时处理多少请求，决定了计算形状、数据流量和用户等待时间。

<style>
.scaling-note {
  border-left: 4px solid var(--global-theme-color, #1b87ad);
  background: rgba(27, 135, 173, 0.07);
  background: color-mix(in srgb, var(--global-theme-color, #1b87ad) 8%, transparent);
  padding: 0.85rem 1.1rem;
  margin: 1.4rem 0;
  border-radius: 0 6px 6px 0;
}
.scaling-note > :last-child { margin-bottom: 0; }
.scaling-note strong { color: var(--global-theme-color, #1b87ad); }
.scaling-nav { font-size: 0.92rem; line-height: 1.9; }
.scaling-nav a { display: inline-block; margin-right: 1rem; }
.scaling-wide { overflow-x: auto; margin: 1rem 0; }
.scaling-wide table { min-width: 650px; }
.scaling-wide th, .scaling-wide td { white-space: nowrap; }
.scaling-figure { margin: 1.5rem 0; }
.scaling-figure .figure-scroll { overflow-x: auto; }
.scaling-figure svg { display: block; width: 100%; min-width: 520px; height: auto; color: var(--global-text-color, #222); }
.scaling-figure text { fill: currentColor; font-family: inherit; font-size: 15px; }
.scaling-figure .scaling-allowed { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.25; }
.scaling-figure .scaling-masked { fill: currentColor; fill-opacity: 0.06; }
.scaling-figure .scaling-cell { stroke: var(--global-theme-color, #1b87ad); stroke-opacity: 0.45; stroke-width: 1; }
.scaling-figure figcaption { font-size: 0.85rem; color: var(--global-text-color-light, #666); margin-top: 0.5rem; }
</style>

<nav class="scaling-nav" aria-label="文章目录">
<a href="#generation">1. 生成过程</a>
<a href="#attention">2. 新 Query 与历史 KV</a>
<a href="#cache">3. 缓存、分页与复用</a>
<a href="#latency">4. 计算、延迟与吞吐</a>
<a href="#layout">5. 多卡布局</a>
<a href="#batch">6. 70B 的 batch 预算</a>
<a href="#serving">7. 服务调度与阶段配比</a>
<a href="#cost">8. 单位资源成本</a>
</nav>

## <span id="generation">1. 从输入问题到输出 token</span>

程序先将角色标记、历史对话和新问题整理为 token 序列。这段输入已经全部确定，可以逐层批量计算：每层产生各位置的 Q、K、V，执行因果 attention 和 MLP，并保存该层的 K/V。最后输入位置的输出经词表投影，给出第一个回答 token 的分数。这个阶段称为 **prefill**。

选出第一个回答 token \\(y_1\\) 后，下一次调用把 \\(y_1\\) 作为输入。它经过全部模型层，读取历史 KV、追加自己的 KV，再预测 \\(y_2\\)。这种逐步推进称为 **decode**：

```text
已知 prompt ── prefill ──→ 选择 y1
历史 KV + y1 ── decode ──→ 选择 y2
历史 KV + y2 ── decode ──→ 选择 y3
                         ……
```

**采样出一个 token，与让这个 token 经过模型，是两个时刻。** 刚选出的 \\(y_1\\) 尚未产生自己的各层 KV；处理它之后，缓存才增加一个位置。

普通 decode 每条请求通常只输入一个新 token，是因为后面的输入还没有确定。矩阵乘法本身完全可以处理多行：已知的 prompt 或后缀可以成块输入，不同请求各自已确定的新 token 也可以组成 batch。用空白占位符代替尚未生成的 token，会改变模型的条件输入。

结束通常由外层程序控制：检测模型选出的结束标记，或触发最大输出长度、停止序列、取消等条件后退出。推理不因输入新问题而更新模型权重。[原书的生成过程](https://jax-ml.github.io/scaling-book/inference/#the-basics-of-transformer-inference)

## <span id="attention">2. 新 Query 只处理新行，K/V 包含全部允许的来源</span>

设某一层已经缓存 \\(S_0\\) 个历史位置，本次输入 \\(m\\) 个新位置，模型宽度为 \\(D\\)，单头宽度为 \\(H\\)。先看一个 Query 头及它对应的 KV 头：

$$
X_{\rm new}:[m,D]
\quad\longrightarrow\quad
Q_{\rm new},K_{\rm new},V_{\rm new}:[m,H].
$$

新 Query 需要查询历史和本块允许的位置，因此沿**位置轴**拼接：

$$
K_{\rm all}=\begin{bmatrix}K_{\rm old}\\K_{\rm new}\end{bmatrix}:[S_0+m,H],
\qquad
V_{\rm all}:[S_0+m,H].
$$

随后：

$$
A=\operatorname{softmax}\!\left(
\frac{Q_{\rm new}K_{\rm all}^{\mathsf T}}{\sqrt H}+\mathcal M
\right):[m,S_0+m],
$$

$$
O_{\rm new}=AV_{\rm all}:[m,H].
$$

\\(\mathcal M\\) 在不可见位置取负无穷，其余位置为零。前 \\(S_0\\) 列是历史，对本块所有 Query 都可见；后 \\(m\\) 列是本块来源，只屏蔽严格上三角，保留对角线和以下位置。

例如缓存 `abc`，本次输入已知后缀 `xefg`。Q 有 4 行，K/V 有 7 行，attention 权重矩阵为 \\([4,7]\\)：

<figure class="scaling-figure">
<div class="figure-scroll">
<svg viewBox="0 0 650 315" role="img" aria-labelledby="mask-title mask-desc">
<title id="mask-title">新 Query 与历史及本块来源的因果 attention</title>
<desc id="mask-desc">四行 Query 分别来自 x、e、f、g，七列来源是历史 a、b、c 和新输入 x、e、f、g。每行都能读取历史，只屏蔽本块中比自己更晚的位置。</desc>
<text x="236" y="25" text-anchor="middle">历史 KV</text>
<text x="432" y="25" text-anchor="middle">本块新增 KV</text>
<text x="30" y="125">新 Query</text>
<text x="180" y="59" text-anchor="middle">a</text>
<text x="236" y="59" text-anchor="middle">b</text>
<text x="292" y="59" text-anchor="middle">c</text>
<text x="348" y="59" text-anchor="middle">x</text>
<text x="404" y="59" text-anchor="middle">e</text>
<text x="460" y="59" text-anchor="middle">f</text>
<text x="516" y="59" text-anchor="middle">g</text>
<text x="127" y="100" text-anchor="middle">x</text>
<rect class="scaling-cell scaling-allowed" x="152" y="75" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="208" y="75" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="264" y="75" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="320" y="75" width="56" height="42"/>
<rect class="scaling-cell scaling-masked" x="376" y="75" width="56" height="42"/>
<text x="404" y="102" text-anchor="middle">×</text>
<rect class="scaling-cell scaling-masked" x="432" y="75" width="56" height="42"/>
<text x="460" y="102" text-anchor="middle">×</text>
<rect class="scaling-cell scaling-masked" x="488" y="75" width="56" height="42"/>
<text x="516" y="102" text-anchor="middle">×</text>
<text x="127" y="142" text-anchor="middle">e</text>
<rect class="scaling-cell scaling-allowed" x="152" y="117" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="208" y="117" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="264" y="117" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="320" y="117" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="376" y="117" width="56" height="42"/>
<rect class="scaling-cell scaling-masked" x="432" y="117" width="56" height="42"/>
<text x="460" y="144" text-anchor="middle">×</text>
<rect class="scaling-cell scaling-masked" x="488" y="117" width="56" height="42"/>
<text x="516" y="144" text-anchor="middle">×</text>
<text x="127" y="184" text-anchor="middle">f</text>
<rect class="scaling-cell scaling-allowed" x="152" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="208" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="264" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="320" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="376" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="432" y="159" width="56" height="42"/>
<rect class="scaling-cell scaling-masked" x="488" y="159" width="56" height="42"/>
<text x="516" y="186" text-anchor="middle">×</text>
<text x="127" y="226" text-anchor="middle">g</text>
<rect class="scaling-cell scaling-allowed" x="152" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="208" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="264" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="320" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="376" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="432" y="201" width="56" height="42"/>
<rect class="scaling-cell scaling-allowed" x="488" y="201" width="56" height="42"/>
<line x1="320" y1="70" x2="320" y2="250" class="scaling-cell" style="stroke-width:2"/>
<text x="348" y="276" text-anchor="middle">4 个新 Query × 7 个来源 → 4 个新输出</text>
<text x="348" y="302" text-anchor="middle">蓝色：可见；×：被因果 mask 屏蔽</text>
</svg>
</div>
<figcaption>历史三列对所有新 Query 可见；右侧四列只屏蔽严格上三角。四行分别读取 4、5、6、7 个位置。</figcaption>
</figure>

这里的矩阵描述逻辑依赖，不要求实现把完整 \\(A\\) 写入 HBM。普通 decode 是 \\(m=1\\) 的特例；无缓存的整段 prefill 是 \\(S_0=0\\) 的特例；命中前缀后的 prefill 与分块 prefill 通常得到矩形 attention。

<div class="scaling-note" markdown="1">
**只拿新 Q 和新 K/V 做 attention，会遗漏历史。** 正确形状是 \\([m,H]\times[H,S\_0+m]\\)，输出仍只有新位置的 \\([m,H]\\)。历史不是重新作为全部输入行跑一遍模型，而是通过每一层的 KV cache 参与读取。
</div>

### 不能把各块独立跑完模型，再拼接缓存

当前层的新 K/V 投影直接读取当前层输入，并不直接用旧 KV 做投影。但该层 attention 读取历史后，输出表征已经包含上下文；下一层的 Q/K/V 又来自这个输出：

```text
本层输入 → 新 Q/K/V → attention 读取历史 → 本层输出
                                             ↓
                                        下一层输入
                                             ↓
                                        下一层 Q/K/V
```

在标准逐 token embedding、归一化和正确位置条件下，首层 K/V 投影可以分别计算；首层 attention 仍必须读取允许的历史。若各块都以空历史独立执行所有层，后续块的中间表征已经不同，最后拼接 KV 不能补回漏掉的跨块计算。

### 容量线性增长，累计读取与计算可以更快增长

单层单 Query 头读取 \\(S\\) 个来源时，\\(QK^{\mathsf T}\\) 和 \\(AV\\) 的主要计算合计约 \\(4SH\\) FLOPs。初始缓存 \\(S_0\\) 个位置，随后实际送入模型 \\(T\\) 个新位置，累计为：

$$
F_{\rm attention}\approx4H\sum_{i=1}^{T}(S_0+i)
=4H\left(TS_0+\frac{T(T+1)}2\right).
$$

固定历史贡献 \\(TS_0\\)，新位置之间的依赖贡献三角项；\\(T\ll S_0\\) 时前者可能主导。与此不同，每次只新增一组固定大小的 K/V，持久缓存随位置数线性增长。上述 \\(T\\) 指实际处理的位置数，不直接等于已返回给用户的 token 数。

## <span id="cache">3. 缓存省掉什么，仍然要读取什么</span>

### KV 容量与 GQA

模型有 \\(L\\) 层、\\(N\_{KV}\\) 个 KV 头，每头宽度 \\(H\\)，每元素 \\(p\_{kv}\\) bytes。每请求每新增位置的逻辑 KV 为：

$$
c_{KV}=2LN_{KV}Hp_{kv}\quad\text{bytes}.
$$

等长 \\(B\\) 请求、各有 \\(S\\) 个缓存位置，无共享和额外复制时，逻辑总量为 \\(BSc\_{KV}\\)；不同长度时将 \\(BS\\) 替换为 \\(\sum_b S_b\\)。物理占用还需计入布局复制、分配预留、量化元数据和碎片。

分组查询注意力 GQA 让多个 Query 头共享一个 KV 头。若每组有 \\(g\\) 个 Query，它们仍各自计算匹配权重与输出。理想情况下，bf16 KV 从 HBM 各读一次，主要计算约为 \\(4gSH\\) FLOPs，KV 读取为 \\(4SH\\) bytes，这部分强度约为 \\(g\\) FLOPs/byte。

固定 \\(g\\) 而历史加倍，计算与 KV 读取同时加倍，强度不变。减少 KV 头数可以减少缓存及 K/V 投影，却不同比减少固定 Query 头数下的匹配计算。GQA 也不是可以直接从现成模型中随意删去 KV 头的运行选项。

### 前缀缓存复用的是已完成的计算

固定模型权重、位置与计算条件，精确相同输入前缀的各层 KV 可以复用。缓存 `abcdef`，新请求为 `abcxefg`，一般只能直接复用 `abc`；后面的 `ef` 虽然字符相同，前文已经改变。真实比较对象是包含角色标记等信息的 token 前缀。

命中前缀缓存省去旧位置的投影、MLP 等重算，但新 Query 仍需读取这段历史。随机采样改变之后选中的 token，不会自动让确定性前向中的同一已知前缀 KV 变成随机量。

### 分页改变物理分配，完整 attention 的依赖仍在

请求最终会生成多长通常事先未知。为每条请求预留最大长度的连续 KV 空间容易浪费；按需扩展连续空间又可能遇到搬迁和碎片。

[PagedAttention](https://arxiv.org/abs/2309.06180)通过块表将逻辑位置映射到按需分配的物理块，允许 KV 不连续保存，也方便公共前缀的物理共享。分页不压缩每个向量，不表示只读取最后一页，也不改变完整 attention 对历史来源的依赖。尾页和管理结构仍有开销。

### 物理存一份，不保证从 HBM 只读一次

取 32 层、8 个 KV 头、\\(H=128\\)、bf16，每位置为 128 KiB。六条请求精确共享 4096 位置前缀，各有 1024 位置私有后缀：

| 项目                 |                                      KV 容量 |
| -------------------- | -------------------------------------------: |
| 公共前缀，只存一份   |                                      512 MiB |
| 每请求私有后缀       |                                      128 MiB |
| 六条请求共享后的总量 |   \\(512+6\times128=1280\\) MiB，即 1.25 GiB |
| 六条请求各存完整历史 | \\(6\times(512+128)=3840\\) MiB，即 3.75 GiB |

如果 decode 内核逐请求扫描历史，且请求间没有片上复用，同一个公共前缀地址仍会被反复从 HBM 读取。本步旧 KV 流量仍可达到 3.75 GiB，尽管物理只存了 1.25 GiB；这里不计本步新增 KV 和其他流量。

要减少这类重复读取，可以先加载一个公共 KV 块，让多个请求的 Query 使用它，再换下一块。每个请求仍保留自己的 attention 累积与归一化状态，不要求整个长前缀一次驻留片上。

<div class="scaling-note" markdown="1">
**分别统计逻辑数据量、物理容量和实际读取量。** 前缀共享能减少重复存储；能否同时减少 HBM 流量，还取决于内核访问顺序和片上复用。容量预留也不能直接当成每轮必读的数据。
</div>

## <span id="latency">4. Prefill、decode 与用户看到的时间</span>

### 已知输入行数决定权重复用

一个有 \\(P_W\\) 个参数的稠密投影，处理 \\(M\\) 行输入，前向主要工作量约为 \\(2MP_W\\) FLOPs。推理没有计算训练所需的 \\(dX\\) 和 \\(dW\\)，不能沿用训练的 \\(6MP_W\\)。把新 token 送回模型是下一次前向，不是反向传播。

Prefill 的许多已知位置可以共用权重 tile：片上加载一块权重后，用于多行输入。各行仍需自己的乘加，并行工作量增加不意味着墙钟时间不变。普通 decode 每请求只有一行，常通过合并不同请求增加权重复用。

长 prefill 的权重矩阵乘法可能计算受限，小 batch decode 则可能读取受限。这个判断要针对算子和具体形状，不能作为整个模型的固定标签。

投影部分常用 \\(2PS\\) 粗估单请求 prefill 工作。按第 8 章算例，\\(P=70\times10^9\\)、\\(S=8192\\)、16 卡、每卡 197 TFLOPs/s、给定有效利用率 40%，约为：

$$
t_{\rm prefill}\approx\frac{2PS}{NC\eta}\approx0.91\ \mathrm{s}.
$$

这未显式计入 attention 的位置配对计算，\\(\eta\\) 也不是硬件常数。因果 prefill 每头允许的位置对为 \\(S(S+1)/2\\)；一条 8192 位置序列，与同时处理两条独立 4096 位置序列，投影工作近似相同，attention 工作不同。FlashAttention 可以避免完整分数方阵驻留 HBM，但不消除需要计算的位置配对。[原书 prefill 估算](https://jax-ml.github.io/scaling-book/applied-inference/#what-about-prefill)

### 单请求间隔与系统吞吐

设每轮理想读取共享权重 \\(W\\) bytes，每请求读取历史 KV \\(K\\) bytes，HBM 带宽为 \\(\beta\\)，暂只考虑读取成本：

$$
t_{\rm step}\approx\frac{W+BK}{\beta},\qquad
R_{\rm token}=\frac{B}{t_{\rm step}}
=\frac{\beta}{W/B+K}.
$$

增大 \\(B\\) 摊薄每个输出 token 的权重流量，却不会自动摊薄每请求自己的 KV。系统吞吐提高，单请求相邻 token 的间隔约为 \\(t\_{\rm step}\\)，也可能变长；不能用 \\(t\_{\rm step}/B\\) 当作单用户等待。这个简化模型的吞吐极限为 \\(\beta/K\\)，实际还受容量、计算和延迟目标限制。

TTFT（Time To First Token）衡量提交请求到首 token 的时间；ITL（Inter-Token Latency）衡量相邻输出 token 的间隔。TTFT 可以包含排队与网络，ITL 也可能因调度暂停而波动。

若 prefill 产生第一个输出，返回 \\(N\_{\rm out}\\) 个 token 后即停，忽略其他开销：

$$
t_{\rm total}\approx t_{\rm prefill}
+\sum_{i=1}^{N_{\rm out}-1}t_{\mathrm{decode},i}.
$$

例如 prefill 为 400 ms、decode 每次 20 ms、返回 101 个 token，总时间为 2.4 s。只将 prefill 减半，首 token 变为 200 ms、总时间 2.2 s；只将 decode 减半，首 token 仍为 400 ms、总时间 1.4 s。优化对象取决于首响应、输出长度和完整完成时间的要求。

### 量化要区分存储格式与计算格式

只把权重从 bf16 压缩为 int8、片上转换后仍用 bf16 运算，会减少权重读取，而不会把主要乘加工作量减半。实际路径需要检查压缩数据是否直接送入计算、是否产生展开后的 HBM 副本，以及缩放、转换和缓冲开销。权重量化也不自动压缩 KV。

在权重读取主导、激活流量较小的投影近似中，令每参数 \\(p_w\\) bytes：

$$
t_W=\frac{Pp_w}{N\beta},\qquad
t_{\rm math}\approx\frac{2PB}{NC},\qquad
B_{\rm crit}\approx\frac{Cp_w}{2\beta}.
$$

仅将 \\(p_w\\) 减半而 \\(C\\) 不变，临界 batch 减半；若适用计算吞吐也翻倍，临界 batch 可以保持不变。能否得到这些收益，还要验证数值质量与内核实现。

## <span id="layout">5. 多卡推理：KV 可能随卡数增加而复制</span>

### 先明确是增加副本，还是扩大同一副本

两张卡各运行一个完整模型，可以并行服务不同请求；两张卡共同做张量并行 TP，则一起推进同一模型副本。固定每副本 batch 为 1，独立副本每步为 \\(t_1\\)，两卡 TP 每步为 \\(t_2\\)：前者总吞吐为 \\(2/t_1\\)，后者为 \\(1/t_2\\)。

\\(t_2<t_1\\) 表示单请求更快，但要在这个比较中提高系统吞吐，需要 \\(t_2<t_1/2\\)。容量或 batch 不同时，必须重新建立比较条件。

### 70B 示例中的完整头布局

以下采用 70B 近似：80 层、64 个 Query 头、8 个 KV 头、\\(H=128\\)。权重与 KV 均按 int8 存储、每元素 1 byte，矩阵计算按 bf16 预算。暂不计量化元数据，每请求每位置 KV 为：

$$
c_{KV}=2\times80\times8\times128=163840\ \mathrm{bytes}=160\ \mathrm{KiB}.
$$

历史为 8192 位置时，每请求逻辑 KV 为 **1.25 GiB，约 1.342 GB**。GB 按 \\(10^9\\) bytes，GiB 按 \\(2^{30}\\) bytes。

若按完整 Query 头切分，并要求各卡本地保存所需的完整 KV 头：

| TP 卡数 | 每卡 Query 头 | 每卡需要的 KV 头 | 全机 KV 副本数 |
| ------: | ------------: | ---------------: | -------------: |
|       8 |             8 |                1 |              1 |
|      16 |             4 |                1 |              2 |

16 卡时，同一组 8 个 Query 分到两卡，各卡仍需相同 KV 头，因此每卡 KV 是逻辑总量的 \\(1/8\\)，不是 \\(1/16\\)。加卡可以减少本地权重，却不一定减少本地 KV。

例如每卡 16 GB，另列 2 GB 工作空间与其他占用，权重近似按 16 卡均分，\\(B=64\\) 时：

$$
M_{\rm chip}\approx4.375+64\times0.16777216+2
=17.112\ \mathrm{GB}.
$$

单卡放不下。只比较全机容量与唯一逻辑数据总量会漏掉复制。权重均分本身也是近似，实际重复保存的 K/V 投影等需依实现另计。

### 减少 KV 复制，需要改变数据放置和交换

可以让每卡保存一部分请求的**全部 KV 头**。做 attention 前，把这些请求的新 Q 从原 TP 布局送到历史所在卡；算完后，再把各 Query 头的输出送回后续投影需要的布局。

```text
按头/特征切分的 Q
      ↓ 跨卡重排
按请求归集 Q，与本地长期驻留的 KV 做 attention
      ↓ 跨卡重排
按头/特征切分的输出，进入后续投影
```

交换的是较小的新 Q 和输出，长历史无需每步迁移。16 卡均衡分配时，每卡保存 \\(B/16\\) 请求的 8 个 KV 头，较“每卡保存所有请求的 1 个 KV 头”减少一半 KV。它没有要求每卡复制完整模型，也不同于增加独立模型副本。[原书 KV 切分讨论](https://jax-ml.github.io/scaling-book/inference/#sharding-the-kv-cache)

### TP 通信不会自动随本地计算一起缩小

对 MLP down 投影，每卡计算：

$$
Z_r:[B,F/N],\quad W_r:[F/N,D],\quad O_r=Z_rW_r:[B,D].
$$

完整输出是 \\(O=\sum_r O_r\\)。若后续要求复制完整输出，需要 AllReduce；若需要分片结果，可以设计 ReduceScatter 等布局。卡数翻倍缩短本地点积长度，但部分和的 \\([B,D]\\) 形状不变。

通信可粗略拆成启动与带宽两项：

$$
t_{\rm coll}\approx\alpha(N)+\kappa(N)\frac{BDp_a}{\beta_{\rm net}}.
$$

\\(p_a\\) 是传输激活的每元素字节数，\\(\alpha\\)、\\(\kappa\\) 和有效带宽依赖算法、拓扑与消息大小。小消息可能由启动延迟主导。当本地计算与权重读取变短，原本被覆盖的通信更容易暴露在关键路径上。

只压缩权重而保持激活格式不变，不会自动缩小这个归约数组。通信占总时间比例上升，也可能只是其他阶段变快；需区分通信本身持续时间、重叠程度和实际阻塞时间。

## <span id="batch">6. 70B 算例：容量允许的 batch，不一定满足延迟</span>

沿用上述 16 卡完整 KV 头布局。参数取[第 8 章的预算值](https://jax-ml.github.io/scaling-book/applied-inference/#visualizing-the-latency-throughput-tradeoff)：每卡 HBM 带宽 \\(\beta=8.2\times10^{11}\\) bytes/s，bf16 计算峰值 \\(C=1.97\times10^{14}\\) FLOPs/s。带宽是算例口径，不是实测吞吐。目标是每次 decode 步进不超过 12 ms。

先采用以下简化：本地权重每步读一次；同卡 Query 对 KV 理想复用；权重矩阵乘法内部计算与读取可重叠；attention 由 KV 读取主导，并与投影阶段串行计时。暂不计 TP 通信、新 KV 写入、其他激活流量和转换开销。

$$
t_{\rm step}\approx
\max\left(\frac{W_{\rm local}}\beta,\frac{2PB}{NC}\right)
+\frac{K_{\rm local}}\beta.
$$

\\(2PB\\) 粗估投影与 MLP，不包含 \\(QK^{\mathsf T}\\) 和 \\(AV\\)；只有 attention 计算能被其读取阶段覆盖，才可用最后一项近似。这个分阶段模型也不能改成全部 FLOPs 与全部 bytes 的一个总 max：总资源下限未必能满足各层的实际依赖。

### 每改变一个变量，都更新其依赖项

在 \\(S=8192\\) 时，每卡权重为 4.375 GB，每请求本地 KV 为 0.16777216 GB：

$$
M_{\rm chip}(B)=4.375+0.16777216B+2\quad\mathrm{GB},
$$

$$
t_{\rm step}(B)\approx
\max(5.3354,\ 0.0444162B)+0.2046002B\quad\mathrm{ms}.
$$

容量最多允许 57 条请求；投影的计算/读取分界约为 \\(B=120.1\\)，所以容量允许范围内仍在权重读取分支。但 \\(B=57\\) 的步时约 17.0 ms，不满足 12 ms。

两项约束共同给出最大整数 **\\(B=32\\)**，步时约 11.883 ms，系统吞吐约 **2693 tokens/s**。把 \\(B\\) 增大时，不能沿用旧 batch 的 KV 读取耗时。

### 历史缩短，可能切换到计算分支

换成历史 \\(S=2048\\) 的工作负载，每请求 KV 缩小到四分之一。如果将 batch 从 32 增到 128，\\(BS\\) 不变，KV 容量和理想读取相同，但新 token 投影工作增加四倍。

此时：

$$
t_{\rm step}(B)\approx
\max(5.3354,\ 0.0444162B)+0.0511500B\quad\mathrm{ms}.
$$

\\(B=128\\) 已进入计算分支，步时约 12.232 ms，超过目标。重新求解得到最大整数 **\\(B=125\\)**，约 11.946 ms、10464 tokens/s；容量允许的上限则是 229。

这里比较的是不同实际历史长度的请求，不能通过直接丢弃同一请求需要的历史来获得等价结果。

<div class="scaling-note" markdown="1">
**先确定工作负载和布局，再写出每一项对 batch、历史长度和卡数的依赖。** \\(BS\\) 相同只保证这个模型中的 KV 项相同，不保证投影 FLOPs、通信或总时间相同。求得候选后，还要检查所用计算/读取分支是否自洽。
</div>

### 加入通信，仍可能降低总成本

回到 \\(S=8192\\)，改用按请求保存 KV 的无复制布局。假定已完成历史放置，16 卡均衡分配，所有层的新 Q/输出重排合计增加 **0.8 ms**，完全不与其他工作重叠；缓冲包含在每卡 2 GB 的其他占用中。该通信值是题设，原先省略的 TP 通信等成本仍未计入。

<div class="scaling-wide" markdown="1">

| Batch |  每卡容量 |      步时 |      系统吞吐 | 满足 12 ms |
| ----: | --------: | --------: | ------------: | ---------- |
|    32 |  9.059 GB |  9.409 ms | 3401 tokens/s | 是         |
|    48 | 10.402 GB | 11.046 ms | 4346 tokens/s | 是         |
|    64 | 11.744 GB | 12.683 ms | 5046 tokens/s | 否         |

</div>

在这三个候选中选择 \\(B=48\\)。每请求本地 KV 读取成本下降，释放出时间余量，允许在相同间隔目标内处理更多请求；新增通信没有抵消这一收益。它是指定布局和成本假设下的比较，不是已验证的软件栈性能或全局最优配置。

以上都只是当前历史长度的快照。生成继续推进，KV 会增长，真实服务还需为长度分布、峰值工作区和尾延迟留出余量。

## <span id="serving">7. 连续批处理与分离服务：统一到 requests/s</span>

### 空出一个位置，不代表新请求已准备好

连续批处理（continuous batching）按生成轮次移除完成请求、接纳就绪请求，不必等待整批同时结束。每一行必须对应正确的请求、KV 和位置。刚到达的请求仍需 prefill，不能因为 decode 空出一行就直接进入。

如果长 prefill 与 decode 共享设备且不能同时执行，立即处理 prefill 会拉长现有用户的 ITL；延后它则增加新用户 TTFT。分块 prefill 在每块保留之前的 KV，块间可安排 decode，或形成混合批，从而缩小调度粒度。但分块可能增加启动和重复读取，小块的矩阵效率也可能下降。[Sarathi-Serve](https://arxiv.org/abs/2403.02310)研究了这类调度。

### 分离设备后，KV 需要交接

Prefill/decode 分离让两组设备分别承担两阶段。两组预先装有模型权重，请求进入 decode 前需要交接各层 KV、位置状态，以及首 token 或相应 logits；布局不同时还要转换。

如果先显示首 token，再花 80 ms 传 KV，接着花 20 ms 完成下一次 decode，那么首到第二个 token 的间隔为 100 ms，后续才可能回到 20 ms。TTFT 小不保证后续输出连续平稳。

### Token 生成率与请求完成率不是同一单位

给定一个独立的稳态例子：每个 prefill 设备组平均 0.90 s 准备一条请求；每个 decode 组维持 48 条活跃请求，平均步时 11 ms，每请求平均需要 **512 次后续 decode 前向**。这里使用平均工作量，且不把前面某个固定历史快照当成完整生成过程。

Prefill 每组准备率为 \\(1/0.90\approx1.111\\) requests/s。Decode 每组每秒完成 \\(48/0.011\\) 次请求 token 步进，每条请求消耗平均 512 次，因此：

$$
R_{\rm decode}=\frac{B}{t_{\rm step}\bar G}
=\frac{48}{0.011\times512}\approx8.523\ \mathrm{requests/s}.
$$

理想供满一个 decode 组需至少 8 个完整 prefill 组。若只有 4 个 prefill 组，其上限为 4.444 requests/s，单独再加 decode 组不会提高这个稳态模型的端到端速率。设备组可以含多张卡，不等于一张芯片。

### 交接网络是第三个阶段

对上述 70B int8 KV，8192 位置 prompt 的交接量为 1.34217728 GB/请求。若设备池之间共享的有效单向带宽为 10 GB/s，无额外副本传输，其服务上限为：

$$
R_{\rm transfer}=\frac{W_{\rm net}}{C_{KV}}
\approx7.451\ \mathrm{requests/s}.
$$

8 个 prefill 组、1 个 decode 组时，三个阶段的能力为：

| 阶段    |             能力 |
| ------- | ---------------: |
| Prefill | 8.889 requests/s |
| KV 交接 | 7.451 requests/s |
| Decode  | 8.523 requests/s |

在请求充足、缓冲与调度理想的条件下，系统上限取三者最小值，瓶颈是交接网络。目标 8 requests/s 至少需要约 **10.737 GB/s** 交接带宽；这是平均流量的必要条件，不保证排队和尾延迟达标。

将一条请求的 KV 均分到目标卡，可以平衡各卡接收量，却不减少跨同一条共享链路的总字节量。不同请求能在三个阶段并行处理，所以稳态吞吐也不能用“各阶段单请求耗时相加后取倒数”估算。

<div class="scaling-note" markdown="1">
**分离服务要在相同的 requests/s 口径下比较 prefill、KV 交接和 decode。** 每步生成多少 token、每秒完成多少请求、单用户等多久，是不同的量。长期工作守恒需要平均生成需求，不能无依据地用输出长度中位数替代。
</div>

## <span id="cost">8. 最少芯片能运行，不等于每 token 成本最低</span>

在无额外复制、权重与 KV 均衡分片的理想布局中，\\(N\\) 卡、每卡容量 \\(H\_{\rm mem}\\)、其他占用 \\(R\\)、模型权重总量 \\(W\\)，可供 KV 的总容量约为：

$$
N(H_{\rm mem}-R)-W.
$$

最小可行配置可能让权重占去大部分容量。扩大同一副本时无需新增一整份模型，额外空间能容纳更多请求，batch 的增幅可能大于卡数增幅。但 KV 复制、计算、通信和延迟目标都会限制收益。

同型号设备每卡每秒费用为 \\(c\\)，满载 decode 的设备成本近似为：

$$
\mathrm{cost/token}=c\frac{Nt_{\rm step}}B,
\qquad
R_{\rm per\ chip}=\frac{B}{Nt_{\rm step}}.
$$

比较配置时，应固定工作负载与延迟要求，再比较系统吞吐 \\(B/t\\)、每芯片吞吐 \\(B/(Nt)\\) 和峰值内存。仅看卡数、单步延迟或满载利用率不足以决定成本。

满载结论也不能直接外推到低到达率：设备持续按原规模付费而实际产出下降，每 token 成本会上升；排队凑 batch 又会增加 TTFT。端到端预算还需加入 prefill、交接网络及实际空闲时间。

本文对应 [How to Scale Your Model 第 7 章](https://jax-ml.github.io/scaling-book/inference/)与[第 8 章](https://jax-ml.github.io/scaling-book/applied-inference/)。基础计算与存储预算见[笔记(1-4)](/posts/2026/09/how-to-scale-your-model-notes-1-4/)，训练并行见[笔记(5-6)](/posts/2026/09/how-to-scale-your-model-notes-5-6/)。数值案例按文中指定假设计算，真实布局、流量、重叠和有效带宽需要通过 profiling 核对。
