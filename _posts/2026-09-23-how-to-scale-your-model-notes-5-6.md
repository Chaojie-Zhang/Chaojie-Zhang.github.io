---
layout: post
title: "How To Scale Your Model 笔记(5-6)"
date: 2026-09-23
permalink: /posts/2026/09/how-to-scale-your-model-notes-5-6/
description: "训练并行的数据分工、通信组与内存生命周期；从 Llama 70B 的参数和 FLOPs 预算推导并行配置的约束。"
tags:
  - Machine Learning
  - Transformer
  - ML Systems
  - Learning Notes
related_posts: true
math: true
---

训练并行需要同时解决三个问题：模型状态和激活如何放进设备内存，各卡如何分担计算，以及计算所需的数据如何到达。增加设备数会减少一部分本地工作，却不一定同比减少通信；分片保存的权重，也不一定在计算时保持分片。

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
.scaling-wide table { min-width: 780px; }
.scaling-wide th, .scaling-wide td { white-space: nowrap; }
.scaling-mesh { margin: 1.5rem 0; }
.scaling-mesh .mesh-scroll { overflow-x: auto; }
.scaling-mesh svg { display: block; width: 100%; min-width: 530px; height: auto; color: var(--global-text-color, #222); }
.scaling-mesh text { fill: currentColor; font-family: inherit; font-size: 15px; }
.scaling-mesh .card { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.10; stroke: var(--global-theme-color, #1b87ad); }
.scaling-mesh .link { stroke: var(--global-theme-color, #1b87ad); stroke-width: 2; }
.scaling-mesh figcaption { font-size: 0.85rem; color: var(--global-text-color-light, #666); margin-top: 0.5rem; }
</style>

<nav class="scaling-nav" aria-label="文章目录">
<a href="#states">1. 训练状态</a>
<a href="#dp-fsdp">2. 数据与状态分片</a>
<a href="#tp-mesh">3. 张量切分与通信组</a>
<a href="#pipeline">4. 微批次与流水线</a>
<a href="#budget">5. 70B 训练预算</a>
<a href="#scaling">6. 并行度与扩展限制</a>
</nav>

## <span id="states">1. 训练需要保存哪些数据</span>

前向计算用输入和权重产生激活；反向计算用这些激活和上游梯度，得到参数梯度及传向前一层的梯度；优化器再根据参数梯度更新权重。它们占据不同的存储，生命周期也不同。

以一种混合精度 Adam 实现为例：

| 对象          | 精度 | 每参数占用 | 用途                           |
| ------------- | ---- | ---------: | ------------------------------ |
| 计算用权重    | bf16 |    2 bytes | 矩阵计算                       |
| 主权重        | FP32 |    4 bytes | 保留较高精度的参数更新         |
| 参数梯度      | FP32 |    4 bytes | 本次更新的梯度，可跨微批次累积 |
| Adam 一阶状态 | FP32 |    4 bytes | 梯度的历史统计                 |
| Adam 二阶状态 | FP32 |    4 bytes | 梯度平方的历史统计             |

这套配置的模型状态预算为 \\(18P\\) bytes，\\(P\\) 是参数个数。它不包含激活或通信工作空间，也不是所有训练实现都使用的常数：主权重是否独立保存、梯度与优化器状态的精度，都可能改变预算。

Q、K、V 等输入相关结果属于激活；\\(dW_Q\\) 等属于参数梯度；Adam 状态则保存跨训练步骤的历史信息。缩小 batch 可以减少激活，但不会减少模型参数对应的状态数组大小。

设备峰值必须按同一时刻的存活对象计算：

$$
M_{\rm peak}=\max_u\left[
M_{\rm state}(u)+M_{\rm activation}(u)+M_{\rm temporary}(u)
\right].
$$

这里的临时空间包括聚合权重、暂存梯度、通信缓冲和算子工作区。已经计入某个数组的存储，不能因为它参与了另一项操作就再计一次。

## <span id="dp-fsdp">2. 数据并行与状态分片：梯度贡献怎样合并</span>

### 数据并行 DP：不同数据，同一组参数

数据并行（Data Parallelism，DP）让各卡保存相同模型，处理不同样本。对线性层：

$$
Y_r=X_rW,\qquad g_r=X_r^{\mathsf T}G_r,
$$

\\(r\\) 表示设备，\\(G_r\\) 是输出的上游梯度。虽然 \\(X_r\\) 只有一部分数据行，\\(g_r\\) 仍与整个 \\(W\\) 同形状：每个参数都可能被本地样本使用。

因此，各端梯度是**同一组参数上的不同数据贡献**，需要逐元素归约。AllReduce（全归约）完成求和并把结果交给组内所有设备；它不是沿矩阵行列拼接。

若各端 loss 都取本地均值，有效 token 数为 \\(n_r\\)，全局平均梯度为：

$$
g=\frac{\sum_r n_rg_r}{\sum_r n_r}.
$$

只有各端有效数量相同，才可直接平均各端梯度。同步后，各端从相同参数与优化器状态执行相同更新，继续保持一致。

固定全局 batch，增加 DP 卡数会减少本地计算和激活。完整模型状态并不缩小，梯度通信对象也不按卡数缩小。例如每端有 2 GB 的完整形状梯度，8 卡 AllReduce 的数组大小仍是 **2 GB**；0.25 GB 可以是环算法中的一个分块，但不是归约结果的完整大小。实际传输量还取决于算法。

本节按完整序列分配数据。若直接把同一序列的不同位置分到不同设备，attention 还需要跨位置的数据交换，不能沿用“前向无需通信”的结论。

### 全分片数据并行 FSDP：分开保存，按需恢复

全分片数据并行（Fully Sharded Data Parallelism，FSDP）进一步把参数、梯度与优化器状态分片保存。在计算后释放完整权重的方案中，一个参数组经历：

1. **AllGather（全收集）权重**：拼齐当前计算所需的权重，各端处理自己的数据。
2. **释放完整权重**：其他模型层继续使用同一批临时空间。
3. **反向前再次聚合**：输入梯度 \\(dX=GW^{\mathsf T}\\) 仍需要权重。
4. **ReduceScatter（归约后分散）梯度**：合并各端数据贡献，各端只保留自己负责参数的全局梯度，再更新对应状态。

是否在前向后释放、提前聚合多少组，会改变通信次数和峰值。不能仅凭“AllReduce 可以分成 ReduceScatter 加 AllGather”，就断言 FSDP 与 DP 的整步通信量总是相同。[PyTorch FSDP 的分片策略](https://docs.pytorch.org/docs/stable/fsdp.html)

FSDP 节省空间的原因是：**所有层都曾完整出现，不等于所有层同时完整驻留。**

设有四个模型层，每层 2 GB 权重，两卡各保存每层的一半。仅计权重，并假设完整缓冲包含本地分片、不重复保存：

| 时刻             |                     每卡占用 |
| ---------------- | ---------------------------: |
| 所有层都保持分片 |          \\(4\times1=4\\) GB |
| 当前一层恢复完整 |        \\(3\times1+2=5\\) GB |
| 同时预取下一层   | \\(2\times1+2\times2=6\\) GB |

这里轮流使用临时空间的是**不同模型层**。预取下一层增加同时存活的数据，换取其通信与当前层计算重叠的机会；是否更快还取决于带宽竞争和依赖关系。

若实现保留本地分片，并另外分配完整聚合缓冲，则要把该缓冲全额计入。例如持久状态 4.5 GB、保存激活 6 GB、其他临时空间 1 GB，再额外分配 0.8 GB 权重缓冲，峰值为 **12.3 GB**。当前组的 0.2 GB 本地分片已在 4.5 GB 内，不能再加；只有它确实与完整缓冲共享存储时，才可少算这 0.2 GB。

<div class="scaling-note" markdown="1">
**FSDP 分片的是状态的保存；当前运算仍可能需要聚合后的权重。** 如果某个矩阵完整权重为 30 GB，而指定执行方式必须先在一张 24 GB 卡上聚合它，仅增加 FSDP 卡数不能解决问题。需要让运算本身也能分片，或改变聚合与执行方式。
</div>

## <span id="tp-mesh">3. 张量并行：同一次运算由多卡共同完成</span>

### 两层 MLP 的切分

张量并行（Tensor Parallelism，TP）让权重在执行矩阵乘法时也保持分片。以普通两层 MLP 为例：

$$
H=\phi(XW_{\rm in}),\qquad Y=HW_{\rm out},
$$

其中 \\(X:[m,D]\\)、\\(W\_{\rm in}:[D,F]\\)、\\(W\_{\rm out}:[F,D]\\)，\\(\phi\\) 为逐元素非线性。两卡沿中间宽度 \\(F\\) 切第一矩阵的列和第二矩阵的对应行：

$$
H_r=\phi(XW_{\mathrm{in},r}):[m,F/2],
\qquad
Z_r=H_rW_{\mathrm{out},r}:[m,D].
$$

\\(H_r\\) 是不同隐藏特征的完整数值，可以直接做非线性。\\(Z_r\\) 则是同一输出位置上的部分和，最终 \\(Y=Z_0+Z_1\\)。若输入在两卡复制，可在末尾 AllReduce，得到复制的完整输出。

若输入边界沿特征维度分片，每端最初只有 \\([m,D/2]\\)，则先 AllGather 输入，再计算，末尾用 ReduceScatter 合并部分和并各留一半输出特征。**ReduceScatter 后的两块结果要按列拼接，不能再次逐元素相加。**

末尾 ReduceScatter 比 AllReduce 少做一个聚合阶段，但入口多了一次 AllGather。相同数组大小、精度和带宽近似下，AG＋RS 与 AR 的通信成本同量级；应比较整个算子块及其边界布局。残差等逐元素操作在布局对齐时可以本地执行，是否再聚合由后续算子决定。

### 同一张卡同时属于两个通信组

将 8 卡组织成 4 个数据处理单元，每个单元用 2 路 TP。横向两卡共同处理一份数据；纵向相同 TP 位置的四卡，组成 FSDP 通信组。

<figure class="scaling-mesh">
<div class="mesh-scroll">
<svg viewBox="0 0 640 350" role="img" aria-labelledby="mesh-title mesh-desc">
<title id="mesh-title">8 卡的张量并行与全分片数据并行分组</title>
<desc id="mesh-desc">每行两张卡组成张量并行组，共同处理一份数据。左列卡0、2、4、6组成一个全分片数据并行组，右列卡1、3、5、7组成另一个组。两列对应不同的计算权重分片。</desc>
<text x="205" y="25" text-anchor="middle">计算左侧权重块</text>
<text x="430" y="25" text-anchor="middle">计算右侧权重块</text>
<line class="link" x1="205" y1="92" x2="205" y2="108" stroke-dasharray="5 5"/>
<line class="link" x1="205" y1="152" x2="205" y2="168" stroke-dasharray="5 5"/>
<line class="link" x1="205" y1="212" x2="205" y2="228" stroke-dasharray="5 5"/>
<line class="link" x1="430" y1="92" x2="430" y2="108" stroke-dasharray="5 5"/>
<line class="link" x1="430" y1="152" x2="430" y2="168" stroke-dasharray="5 5"/>
<line class="link" x1="430" y1="212" x2="430" y2="228" stroke-dasharray="5 5"/>
<text x="20" y="76">数据 A</text><text x="20" y="136">数据 B</text><text x="20" y="196">数据 C</text><text x="20" y="256">数据 D</text>
<line class="link" x1="275" y1="70" x2="360" y2="70"/>
<line class="link" x1="275" y1="130" x2="360" y2="130"/>
<line class="link" x1="275" y1="190" x2="360" y2="190"/>
<line class="link" x1="275" y1="250" x2="360" y2="250"/>
<rect class="card" x="135" y="48" width="140" height="44" rx="5"/><text x="205" y="76" text-anchor="middle">卡 0</text>
<rect class="card" x="360" y="48" width="140" height="44" rx="5"/><text x="430" y="76" text-anchor="middle">卡 1</text>
<rect class="card" x="135" y="108" width="140" height="44" rx="5"/><text x="205" y="136" text-anchor="middle">卡 2</text>
<rect class="card" x="360" y="108" width="140" height="44" rx="5"/><text x="430" y="136" text-anchor="middle">卡 3</text>
<rect class="card" x="135" y="168" width="140" height="44" rx="5"/><text x="205" y="196" text-anchor="middle">卡 4</text>
<rect class="card" x="360" y="168" width="140" height="44" rx="5"/><text x="430" y="196" text-anchor="middle">卡 5</text>
<rect class="card" x="135" y="228" width="140" height="44" rx="5"/><text x="205" y="256" text-anchor="middle">卡 6</text>
<rect class="card" x="360" y="228" width="140" height="44" rx="5"/><text x="430" y="256" text-anchor="middle">卡 7</text>
<text x="520" y="76">TP 组</text><text x="520" y="136">TP 组</text><text x="520" y="196">TP 组</text><text x="520" y="256">TP 组</text>
<text x="205" y="306" text-anchor="middle">FSDP：0、2、4、6</text>
<text x="430" y="306" text-anchor="middle">FSDP：1、3、5、7</text>
<text x="320" y="338" text-anchor="middle">横向合作处理同一份数据；纵向汇总不同数据的贡献</text>
</svg>
</div>
<figcaption>逻辑分组示意，连线不代表实际物理网络拓扑。列标题指各卡计算时需要恢复的权重块；闲置时，该块又沿纵向分片保存。</figcaption>
</figure>

以卡 0 为例：它与卡 1 合作计算数据 A；又与卡 2、4、6 一起保存左侧权重块的分片，并合并数据 A、B、C、D 对这块权重的梯度。卡号只是标签，决定分组的是**相同权重位置与不同数据贡献的对应关系**。

“一个数据处理单元”与“一个 FSDP 通信组”不是同一件事。图中有四个数据处理单元，却只有两个 FSDP 通信组，每组四卡。未使用流水线并行时，这些卡可以参与全部模型层；按层聚合参数并不等于给各层固定分配不同设备。

<div class="scaling-note" markdown="1">
**几种并行方式沿不同维度分工，不能排成单一的粗细层级。** DP 分数据，FSDP 分片保存模型状态，TP 切同一层内部运算，PP 把不同层交给不同阶段，CP 切同一序列的位置。这些分工可以组合。
</div>

## <span id="pipeline">4. 微批次：减少同时存活的数据，也引入调度问题</span>

一个训练 batch 可以拆成多个 microbatch（微批次），分别前向、反向，将参数梯度按有效样本或 token 数正确加权累积，最后统一更新。权重在这些微批次之间保持不变；若每份之后都更新，就改变了训练步骤。

依次完成每份的前向与反向，可以尽早释放其保存激活。但模型状态仍在，不能把训练总内存整体除以微批次数。

### 流水线并行 PP 的时间顺序

流水线并行（Pipeline Parallelism，PP）把不同模型层交给不同 stage（阶段）。前向传边界激活，反向传损失对边界激活的梯度；后者不是参数梯度。多个微批次让不同阶段可以同时工作。

下面只分析前向：两阶段、四个微批次，每阶段处理一份耗时 \\(\tau\\)，忽略通信且负载均衡。

| 时隙   | 1    | 2   | 3   | 4   | 5    |
| ------ | ---- | --- | --- | --- | ---- |
| 阶段 0 | μ1   | μ2  | μ3  | μ4  | 空闲 |
| 阶段 1 | 空闲 | μ1  | μ2  | μ3  | μ4   |

推广到 \\(p\\) 个阶段、\\(K\\) 个微批次，第一份在 \\(p\tau\\) 后完成，剩余 \\(K-1\\) 份每隔 \\(\tau\\) 完成一份：

$$
T_{\rm forward}=(K+p-1)\tau,\qquad
U_{\rm busy}=\frac{K}{K+p-1}.
$$

\\(U\_{\rm busy}\\) 是这条时间轴中的设备忙碌比例，不是 FLOPs 利用率。每个微批次的端到端延迟仍是 \\(p\tau\\)，流水线提高的是连续处理的吞吐。

固定总 batch 时，微批次越多，每份越小，\\(\tau\\) 也随之变化。若每阶段总纯计算为 \\(C\\)，每份还有不可重叠固定开销 \\(\delta\\)，则：

$$
T_{\rm forward}=(K+p-1)\left(\frac{C}{K}+\delta\right).
$$

例如 \\(p=3\\)、\\(C=8\\) ms、\\(\delta=0.5\\) ms，\\(K=4\\) 时为 15 ms，\\(K=16\\) 时反而为 18 ms。气泡减少了，固定开销却增加了；实际还需考虑小矩阵效率。

训练还多一层约束：发送了边界输出，不代表本地反向所需的输入和中间激活已能释放。若先全部前向、再全部反向，每份需保存 \\(a\\) bytes，保存量可能累积为 \\(Ka\\)。固定 batch 下 \\(a\propto1/K\\)，总保存量未必下降。交错前向与反向可以缩短存活时间，但需按具体调度统计峰值，不能把上面的前向公式直接当作完整训练 step 时间。

## <span id="budget">5. 从 70B 模型结构计算训练预算</span>

### 参数与 FLOPs

采用[第 6 章的 Llama 70B 配置](https://jax-ml.github.io/scaling-book/applied-training/#what-does-llama-3-look-like)：80 层，模型宽度 \\(D=8192\\)，gated MLP 宽度 \\(F=28672\\)，64 个 Query 头、8 个 KV 头，每头宽度 \\(h=128\\)，词表 \\(V=128256\\)，输入与输出词表权重不共享。

| 模块                   | 参数数目                |    结果 |
| ---------------------- | ----------------------- | ------: |
| Gated MLP              | \\(3LDF\\)              | 56.371B |
| Q/K/V/O 投影           | \\(2LD(n_q+n\_{kv})h\\) | 12.080B |
| 输入与输出词表         | \\(2DV\\)               |  2.101B |
| 合计，忽略 norm 等小项 |                         | 70.552B |

Gated MLP 有两份 \\(D\to F\\) 投影，分别产生特征和门控，再经 \\(F\to D\\) 投影返回模型宽度，因此是三份 \\(DF\\) 参数。KV 头数减少会缩小 K/V 投影，输出投影仍接收全部 Query 头的结果，不随 KV 头数同比缩小。

对一个稠密线性层 \\(Y=XW\\)，前向 \\(XW\\)、权重梯度 \\(X^{\mathsf T}G\\)、输入梯度 \\(GW^{\mathsf T}\\) 各约需要 \\(2mP_W\\) FLOPs。假设两种梯度均需计算，且不重算激活，则该层每 token 的训练工作量约为 \\(6P_W\\)。

稠密投影主导时，用模型参数量 \\(P\\) 作近似：

$$
F_{\rm step}\approx6PB_{\rm tok},\qquad
F_{\rm total}\approx6PN_{\rm tok}.
$$

\\(B\_{\rm tok}\\) 是每次参数更新包含的 token 数，\\(N\_{\rm tok}\\) 是训练全程 token 数。固定后者，batch 翻倍会让每步工作量翻倍、步数减半；总主要 FLOPs 不变，但矩阵效率、通信频率和优化过程会变化。

\\(6P\\) 不是精确的逐算子预算。输入 embedding 是查表；attention 的位置配对计算不由参数量直接计出；激活重算、逐元素运算和优化器更新也需检查。长上下文尤其不能忽略 attention 成本。

### 有效吞吐与日历时间

把参数量取整为 70B、训练数据量设为 15T token：

$$
F_{\rm total}\approx6\times70\times10^9\times15\times10^{12}
=6.3\times10^{24}\ \mathrm{FLOPs}.
$$

若有效模型吞吐为 1 EFLOPs/s，时间约 72.9 天；若为 4 EFLOPs/s，则约 18.2 天。这里的有效模型吞吐指按同一模型 FLOPs 口径计算的进度除以运行时间，并非设备峰值。

若只有硬件峰值，可以写成：

$$
t_{\rm calendar}\approx
\frac{F_{\rm model}}{N_{\rm dev} f_{\rm peak}\,\mu\,a}.
$$

其中 \\(\mu\\) 为正常训练步骤期间的模型 FLOPs 利用率（MFU），\\(a\\) 为未被 \\(\mu\\) 计入的有效训练时间占比，例如中断和恢复带来的损失。峰值必须与精度、稠密或稀疏模式匹配。

<div class="scaling-note" markdown="1">
**时间估算的分子与分母必须使用同一口径。** 若 MFU 的分子按不含重算的模型 FLOPs 统计，重算耗时已经降低了 MFU，不能再给分子机械加一次重算倍率。若吞吐已按完整日历时间计算，也不能再扣一次故障恢复比例。
</div>

### 状态容量与激活容量

仍用取整的 70B 参数和前述 18 bytes/参数配置，模型状态为 1.26 TB。另设 \\(B\_{\rm tok}=4\times10^6\\)，全部前向后再反向，每层保存两份 bf16 的 \\([B\_{\rm tok},D]\\) 激活检查点，则逻辑保存量为：

$$
M_{\rm saved}=2LB_{\rm tok}D\times2\ \mathrm{bytes}
\approx10.49\ \mathrm{TB}.
$$

两项小计约 11.75 TB，尚未计临时缓冲和并行复制。将 batch 按完整序列分为四份，依次完成每份前向、反向并释放其激活，最后统一更新：模型状态仍为 1.26 TB，保存激活约为 2.62 TB，小计约 **3.88 TB**。这些是逻辑数据量，不能直接除总卡数就宣布每卡峰值可行。

激活 checkpoint 通过少存中间结果、反向时重算来省内存；用于故障恢复的训练 checkpoint 则把权重、优化器等状态写入持久存储。两者解决不同问题。以上 TB、GB 为十进制；后面的 MiB 使用 \\(2^{20}\\) bytes。

## <span id="scaling">6. 并行配置先满足约束，再比较通信</span>

### 序列条数与 token 行数

对于等长序列：

$$
B_{\rm tok}=B_{\rm seq}S.
$$

\\(B\_{\rm seq}\\) 是序列条数，\\(S\\) 是每条序列的 token 数。设有 \\(N_F\\) 个数据处理单元，每个单元的线性层输入先是 \\([B\_{\rm seq}/N\_F,S,D]\\)，合并前两轴后才是 \\([m,D]\\)，其中 \\(m=B\_{\rm tok}/N_F\\)。一条长度 4096 的序列对应 4096 行，不是一行。

下面使用 **8192 卡的假想配置**，不将它等同于原书的完整 TPU pod。无流水线并行、无上下文切分，要求所有数据处理单元在当前微批次都有完整序列可处理。若每个 TP 组有 \\(t\\) 张卡，则：

$$
N_F=\frac{N}{t}\le B_{\rm seq}
\quad\Longrightarrow\quad
t\ge\frac{N}{B_{\rm seq}}.
$$

1024 条序列至少需要 8 路 TP；当前微批次只有 256 条时，至少需要 32 路。更小的 TP 组并非完全不能运行，而是这个完整序列分工无法让所有数据处理单元同时有数据。

上下文并行（Context Parallelism，CP）可以把一条序列切到多个设备，却不能消除 attention 的依赖。例如卡 A 保存位置 1–4，卡 B 保存位置 5–8：因果 attention 中，B 的位置 6 仍需访问 1–6 的 K/V，远端部分必须获得；A 的位置 3 则不需要未来位置。

### 加卡为什么可能不再加速

只看一个 bf16 权重矩阵 \\(W:[D,F]\\)。无 TP、每卡处理 \\(m\\) 个 token 时，前向计算约为 \\(2mDF\\) FLOPs，FSDP 聚合的目标为 \\(2DF\\) bytes。以目标数组大小近似大通信组的传输规模，网络算术强度的数值约为 \\(m\\) FLOPs/byte。

固定全局 token 数继续增加设备，本地 \\(m\\) 下降，完整权重聚合目标却不变。计算缩短到不足以覆盖通信之后，扩展就受到限制。

[原书的 TPU v5p 简化模型](https://jax-ml.github.io/scaling-book/training/#fully-sharded-data-parallelism-fsdp)使用约 459 TFLOPs/s 的单芯片计算峰值；若理想地使用三轴、合计 540 GB/s 的网络带宽，比值为约 **850 FLOPs/byte**。本地只有 512 或 256 token 时，都在该模型的通信侧。这些是书中带宽与重叠假设下的估算，不是任意 collective 的实测保证；若 token 跨完整序列分配，还需另算 CP 的通信。

### FSDP 与 TP 的两项竞争

固定总卡数 \\(N\\)，仅分析一个 up 投影 \\(W:[D,F]\\)。每个数据处理单元有 \\(t\\) 张 TP 卡，共有 \\(N/t\\) 个单元。假设入口激活沿特征维度分片，权重和激活均为 bf16：

| 聚合对象        | 每卡聚合后需要的 shape    | 完整目标大小                |
| --------------- | ------------------------- | --------------------------- |
| FSDP 权重聚合   | \\([D,F/t]\\)             | \\(V_W=2DF/t\\)             |
| TP 输入激活聚合 | \\([B\_{\rm tok}t/N,D]\\) | \\(V_X=2B\_{\rm tok}Dt/N\\) |

提高 TP 并行度，每卡计算的权重块变小；但数据处理单元减少，每个单元分到更多 token，完整输入激活反而变大。这是在两种通信对象之间重新分配成本，模型总参数没有改变。

作为**单投影准备阶段的简化估算**，令两次 AllGather 串行、有效带宽相同，忽略启动延迟及有限组大小的传输系数，则最小化时间等价于最小化：

$$
V(t)=\frac{2DF}{t}+\frac{2B_{\rm tok}Dt}{N}.
$$

暂时允许 \\(t\\) 连续变化：

$$
\frac{dV}{dt}=-\frac{2DF}{t^2}+\frac{2B_{\rm tok}D}{N}=0
\quad\Longrightarrow\quad
t_* = \sqrt{\frac{FN}{B_{\rm tok}}}.
$$

这个式子只是当前两项通信模型的最小点，不是完整训练的最优 TP 公式。两条路径带宽不同、通信可重叠或 TP 度数改变网络路由时，目标函数也要改变。

取 \\(N=8192\\)、\\(D=8192\\)、\\(F=28672\\)、\\(S=4096\\)，精确的 token 数按序列条数乘长度计算：

<div class="scaling-wide" markdown="1">

| 序列数 | TP 路数 | 数据处理单元数 | 每单元 token 行数 | 权重目标 | 激活目标 | 两项目标之和 |
| -----: | ------: | -------------: | ----------------: | -------: | -------: | -----------: |
|   1024 |       8 |           1024 |              4096 |   56 MiB |   64 MiB |      120 MiB |
|   1024 |      32 |            256 |             16384 |   14 MiB |  256 MiB |      270 MiB |
|    256 |      16 |            512 |          序列不足 |        — |        — |       不可行 |
|    256 |      32 |            256 |              4096 |   14 MiB |   64 MiB |       78 MiB |
|    256 |      64 |            128 |              8192 |    7 MiB |  128 MiB |      135 MiB |

</div>

1024 条序列时，连续最小点为 \\(\sqrt{56}\approx7.48\\)，8 路是可检查的候选。256 条序列时，连续最小点升为 \\(\sqrt{224}\approx14.97\\)，但 16 路产生 512 个数据处理单元，序列不够。若候选只有 8、16、32、64，则在上述约束和近似下选择 **32 路**。

表中是完整目标数组大小，不是全网流量，也未统计整步所有 collective。该选择还没有证明整个模型可行：64 个 Query 头可以按每卡两个分到 32 卡，但 8 个 KV 头不能直接切成 32 份完整头，需要 KV 复制或其他布局，并计入相应开销。

<div class="scaling-note" markdown="1">
**连续最小值给出候选位置；数据分配、算子布局和内存容量决定可行范围。** 完整方案还需逐项检查头与矩阵的切分、每卡同时存活的数据，以及所有层前向和反向的通信依赖。最后以相同工作负载验证数值结果、稳定 step 时间和峰值内存，才能判断实际收益。
</div>

本文对应 [How to Scale Your Model 第 5 章：训练并行](https://jax-ml.github.io/scaling-book/training/)与[第 6 章：Llama 训练估算](https://jax-ml.github.io/scaling-book/applied-training/)。矩阵切分、Roofline 和 Transformer 基础预算见[笔记(1-4)](/posts/2026/09/how-to-scale-your-model-notes-1-4/)。数值案例中的缓冲策略、8192 卡布局及单投影优化均按正文明确的假设计算，不代表实际训练配置或测量结果。
