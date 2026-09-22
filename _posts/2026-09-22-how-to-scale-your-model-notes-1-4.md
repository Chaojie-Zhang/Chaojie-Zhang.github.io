---
layout: post
title: "How To Scale Your Model 笔记(1-4)"
date: 2026-09-22
permalink: /posts/2026/09/how-to-scale-your-model-notes-1-4/
description: "从 Roofline、数据复用与流水线，到矩阵切分、通信和 Transformer 预算：沿着数据流建立模型计算与存储的物理图像。"
tags:
  - Machine Learning
  - Transformer
  - ML Systems
  - Learning Notes
related_posts: true
math: true
---

一个模型有多少参数，并不能直接告诉我们它需要多大内存、生成一个 token 要多久，或者增加一倍芯片能快多少。要回答这些问题，还得知道：这次输入触发了哪些计算，数据从哪里来，同一份数据能用几次，以及下一步到底在等什么。

最近读完了 [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) 的前四章：Roofline、TPU、矩阵切分与 Transformer 数学。对有物理背景的我来说，数量级和比例关系比较容易上手；更值得停下来想清楚的，是公式背后的实际过程。比如，为什么流水线总时间还有启动或收尾？矩阵形状已经正确，为什么结果还不是完整的？只有一个新 token，为什么 attention 仍要读几千个位置？

这篇笔记沿着这些问题重新组织学习内容。前两章建立单芯片的计算与搬运图像，第三章把它扩展到多芯片，第四章再把具体模型放进去。文中的数值案例是推导练习，不是硬件实测；一些帮助理解的小例子和解释也并非原文逐句翻译。

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
.scaling-note strong, .scaling-em { color: var(--global-theme-color, #1b87ad); }
.scaling-figure { margin: 1.5rem 0; }
.scaling-figure .scaling-scroll { overflow-x: auto; padding-bottom: 0.3rem; }
.scaling-figure svg { display: block; width: 100%; min-width: 570px; height: auto; color: var(--global-text-color, #222); }
.scaling-figure .accent { fill: var(--global-theme-color, #1b87ad); }
.scaling-figure .load { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.15; stroke: var(--global-theme-color, #1b87ad); }
.scaling-figure .compute { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.36; stroke: var(--global-theme-color, #1b87ad); }
.scaling-figure text { fill: currentColor; font-family: inherit; font-size: 14px; }
.scaling-figure figcaption { color: var(--global-text-color-light, #666); font-size: 0.85rem; margin-top: 0.4rem; }
.scaling-nav { font-size: 0.92rem; line-height: 1.9; }
.scaling-nav a { display: inline-block; margin-right: 1rem; }
</style>

<nav class="scaling-nav" aria-label="文章目录">
<a href="#roofline">1. 计算与搬运</a>
<a href="#pipeline">2. 复用与流水线</a>
<a href="#sharding">3. 切分与通信</a>
<a href="#transformer">4. 模型中的数据流</a>
<a href="#kv">5. 单步生成与 KV</a>
<a href="#memory">6. 内存生命周期</a>
<a href="#budget">7. 一次完整预算</a>
</nav>

<h2 id="roofline">1. 先分清：装得下、搬得动、算得完</h2>

讨论速度前，至少有三种不同的约束：容量用 bytes 衡量，带宽用 bytes/s 衡量，计算吞吐用 FLOPs/s 衡量。参数量只是可学习标量的个数，乘上每个元素的存储字节数，才变成权重容量。

在一个简化的 TPU 图像里，大量数据保存在 HBM，分块搬入较小的片上存储 VMEM，再由矩阵计算单元 MXU 或向量计算单元 VPU 使用。两类计算单元按运算类型分工，不是所有数据都依次经过两者。整个矩阵也不必一次装入 VMEM，关键是当前工作集能否放下，以及后面还会不会重读。

用 \\(\mathcal F\\) 表示一次任务的计算量，\\(C\\) 表示适用于该精度和运算类型的计算峰值；用 \\(Q\\) 表示经过某条接口的累计字节数，\\(\beta\\) 表示该接口的带宽上限。则资源给出的时间下限是：

$$
t_{\rm compute,min}=\frac{\mathcal F}{C},\qquad
t_{\rm transfer,min}=\frac{Q}{\beta},\qquad
t\geq\max\left(\frac{\mathcal F}{C},\frac{Q}{\beta}\right).
$$

这个最大值首先是一个**下限**。即使计算与搬运不能重叠，下限仍然成立；要接近它，则需要合适的数据依赖、调度、利用率和重叠程度。真实运行中还有启动、同步、其他算子等成本。[原书第 1 章](https://jax-ml.github.io/scaling-book/roofline/)

<div class="scaling-note" markdown="1">
**先给字节数指定一条路径。** 模型存了 10 GB，不等于每次计算恰好搬 10 GB。某些数据可能留在片上重复使用，某些又可能从 HBM 读很多次。HBM 流量和芯片间流量也必须分别记账。
</div>

### 算术强度：搬来的数据做了多少工作？

定义算术强度 \\(I=\mathcal F/Q\\)，单位为 FLOPs/byte。计算与搬运时间相等时，\\(I=C/\beta\\)。因此吞吐量的理想上限可以写成：

$$
R\leq\min(C,\beta I).
$$

低强度时，数据供给限制了计算吞吐，上限随 \\(I\\) 增加；强度足够高后，算力形成水平的“屋顶”。比较实测性能时，应该同时看芯片峰值与该任务对应的屋顶。例如芯片峰值为 6 GFLOPs/s，任务的带宽屋顶为 4 GFLOPs/s，实测 3 GFLOPs/s：这既是峰值的 50%，也是任务屋顶的 75%。仅凭前一个百分数，无法断言还有一倍加速空间。

矩阵乘法最能体现复用的作用。设：

$$
X_{m\times k}W_{k\times n}=Y_{m\times n}.
$$

每个输出需要一个长度为 \\(k\\) 的点积，总计算量约为 \\(2mkn\\)。若三个矩阵均以 bf16 存储，每元素 2 bytes；输入各从 HBM 读一次，输出写一次，中间量保留在片上，则：

$$
Q=2(mk+kn+mn),\qquad
I\approx\frac{mkn}{mk+kn+mn}.
$$

当 \\(m\ll k,n\\)，权重读取占主导，得到 \\(I\approx m\\)。它的含义是：同一份权重服务更多输入行，每行分摊的权重搬运量下降。**从一次一行计算变成一次八行计算，权重读取量可以不变，但总流量并没有自动减少，新增输入和输出仍要搬。** 如果与八次各自重新加载权重的一行计算相比，才省掉了七次权重读取。

这个近似不能无限延伸。若 \\(m=k=n=1024\\)，强度约为 \\(1024/3\\)，不是 1024；若固定 \\(k=n=1024\\)，只让 \\(m\\) 增大，则：

$$
I\approx\frac{kn}{k+n+kn/m}\longrightarrow\frac{kn}{k+n}=512.
$$

权重的固定成本可以不断分摊，但每增加一行，仍要读取 \\(k\\) 个输入、写回 \\(n\\) 个输出。这个极限把复用的收益与边界一起显示出来。

## <span id="pipeline">2. 复用与流水线：容量和时间顺序开始重要</span>

上面的“输入各读一次”是一种理想数据流。真实芯片片上空间有限，同一个权重块再次被需要时，可能已经被替换了。[原书第 2 章](https://jax-ml.github.io/scaling-book/tpus/)讨论的存储层次，正是把公式落实到访问顺序的起点。

例如，8 MB 权重分成两个 4 MB 块 \\(W_1,W_2\\)，片上留给权重的空间只够一块。四组输入都要用这两个块：按输入组处理，每组依次装入两个权重块，会读取 32 MB；先让所有输入用完 \\(W_1\\)，再统一换成 \\(W_2\\)，权重读取量只有 8 MB。

后一个方案可能增加输入重读，是否更快要看完整预算。输出也不能只数“写了几次”：若 \\(W=[W\_1\mid W\_2]\\) 是按输出列切分，那么 \\(Y=[XW\_1\mid XW\_2]\\)。左右两部分分两次写回，仍是一份输出，并非写了两份完整矩阵。

### 同一块先搬后算，不同块可以同时进行

搬运与计算如何重叠，最初让我停下来想了很久。单个块存在严格依赖，必须先搬到再计算；但计算第一个块时，可以准备第二个块。双缓冲让两个单元分别使用不同的缓冲区，随后交换角色。

设共有 \\(q\\) 块，每块搬运耗时 \\(a\\)、计算耗时 \\(c\\)。假设两阶段各自串行、彼此可并行，缓冲足够，暂不计输出写回和额外开销。我更容易从“慢的单元持续工作”理解总时间：

$$
t_{\rm pipeline}=
\begin{cases}
qa+c,&a\geq c,\\
a+qc,&c\geq a.
\end{cases}
$$

搬运慢，就连续搬完所有块，末尾补一次计算；计算慢，就先等第一块到达，再连续算完所有块。因此：

$$
\boxed{t_{\rm pipeline}=q\max(a,c)+\min(a,c)
=a+c+(q-1)\max(a,c).}
$$

<figure class="scaling-figure">
<div class="scaling-scroll">
<svg viewBox="0 0 700 295" role="img" aria-labelledby="pipeline-title pipeline-desc">
<title id="pipeline-title">四个数据块的两种流水线</title>
<desc id="pipeline-desc">计算较慢时，每块搬运2微秒、计算3微秒，总耗时14微秒。搬运较慢时，每块搬运4微秒、计算3微秒，总耗时19微秒。每块计算都在自己的数据到达之后开始。</desc>
<text x="12" y="22" font-weight="600">计算慢：搬运 2 μs，计算 3 μs</text>
<text x="12" y="57">搬运</text><text x="12" y="95">计算</text>
<g transform="translate(90,36)">
<rect class="load" x="0" y="0" width="54" height="29" rx="3"/><text x="27" y="20" text-anchor="middle">1</text>
<rect class="load" x="54" y="0" width="54" height="29" rx="3"/><text x="81" y="20" text-anchor="middle">2</text>
<rect class="load" x="135" y="0" width="54" height="29" rx="3"/><text x="162" y="20" text-anchor="middle">3</text>
<rect class="load" x="216" y="0" width="54" height="29" rx="3"/><text x="243" y="20" text-anchor="middle">4</text>
<rect class="compute" x="54" y="38" width="81" height="29" rx="3"/><text x="94" y="58" text-anchor="middle">1</text>
<rect class="compute" x="135" y="38" width="81" height="29" rx="3"/><text x="175" y="58" text-anchor="middle">2</text>
<rect class="compute" x="216" y="38" width="81" height="29" rx="3"/><text x="256" y="58" text-anchor="middle">3</text>
<rect class="compute" x="297" y="38" width="81" height="29" rx="3"/><text x="337" y="58" text-anchor="middle">4</text>
<text x="0" y="88">0</text><text x="135" y="88" text-anchor="middle">5</text><text x="216" y="88" text-anchor="middle">8</text><text x="297" y="88" text-anchor="middle">11</text><text x="378" y="88" text-anchor="middle">14 μs</text>
</g>
<text x="12" y="164" font-weight="600">搬运慢：搬运 4 μs，计算 3 μs</text>
<text x="12" y="199">搬运</text><text x="12" y="237">计算</text>
<g transform="translate(90,178)">
<rect class="load" x="0" y="0" width="108" height="29" rx="3"/><text x="54" y="20" text-anchor="middle">1</text>
<rect class="load" x="108" y="0" width="108" height="29" rx="3"/><text x="162" y="20" text-anchor="middle">2</text>
<rect class="load" x="216" y="0" width="108" height="29" rx="3"/><text x="270" y="20" text-anchor="middle">3</text>
<rect class="load" x="324" y="0" width="108" height="29" rx="3"/><text x="378" y="20" text-anchor="middle">4</text>
<rect class="compute" x="108" y="38" width="81" height="29" rx="3"/><text x="148" y="58" text-anchor="middle">1</text>
<rect class="compute" x="216" y="38" width="81" height="29" rx="3"/><text x="256" y="58" text-anchor="middle">2</text>
<rect class="compute" x="324" y="38" width="81" height="29" rx="3"/><text x="364" y="58" text-anchor="middle">3</text>
<rect class="compute" x="432" y="38" width="81" height="29" rx="3"/><text x="472" y="58" text-anchor="middle">4</text>
<text x="0" y="88">0</text><text x="189" y="88" text-anchor="middle">7</text><text x="297" y="88" text-anchor="middle">11</text><text x="405" y="88" text-anchor="middle">15</text><text x="513" y="88" text-anchor="middle">19 μs</text>
</g>
</svg>
</div>
<figcaption>自绘示意。横轴比例相同，浅色是搬运，深色是计算。上图的搬运空档来自双缓冲复用；下图的计算空档来自等待数据。</figcaption>
</figure>

<div class="scaling-note" markdown="1">
**稳态产出间隔、单块延迟和整个任务时间，是三个不同的量。** 每隔 3 μs 完成一块，不代表一块只经历 3 μs。有限任务还有启动、收尾；整个任务的资源下限也不自动等于有限流水线的精确时间。
</div>

例如全任务纯搬运 0.4 ms、纯计算 0.2 ms，均匀分为 \\(q\\) 块，上述模型给出 \\(t=0.4+0.2/q\\) ms。分两块时是 0.5 ms；忽略启动收尾才取 0.4 ms。实际分块也有调度和效率成本，不能由这个模型推出“切得越细越好”。

## <span id="sharding">3. 多芯片：先问缺什么，再选通信操作</span>

把模型切到多块设备后，增加了一类数据路径。TPU 的 ICI 是芯片间互联，PCIe 连接主机与加速器，DCN 是主机网络；每种路径有自己的带宽和延迟。芯片怎样连接、消息要经过几跳、哪些传输能同时进行，都会改变结果。逻辑 mesh 换一个名字，并不会改变物理链路。

但在估通信时间之前，更基础的问题是：**每个设备现在算出的东西，到底是什么？** [第三章](https://jax-ml.github.io/scaling-book/sharding/)的主线可以从一个输出元素开始：

$$
C_{ik}=\sum_j A_{ij}B_{jk}.
$$

若按输出行或列分工，一个设备可以负责某些输出位置的完整求和。若沿 \\(j\\) 分工，各设备则负责同一输出位置的不同求和项。两者都能得到一个矩阵，但含义完全不同。

以两设备为例，沿归约维度切分：

$$
A=[A_0\mid A_1],\qquad
B=\begin{bmatrix}B_0\\B_1\end{bmatrix},\qquad
C=A_0B_0+A_1B_1.
$$

每端的乘积都已经具有整个 \\(C\\) 的形状，却只包含部分贡献。把它们拼接会错，逐元素求和才对。相反，若沿 \\(B\\) 的输出列切分，\\(AB_0\\) 和 \\(AB_1\\) 是不同输出区域，应按列拼接。

还有一种更隐蔽的错误：本地两个矩阵的内维长度相等，但分别对应全局 \\(j\\) 的不同区间。这样的乘法在 shape 上合法，数学上却配错了元素。检查布局时，需要追踪全局索引，而不只看维度数字。

### 四种 collective 对应四种数据需求

| 操作                 | 起点缺什么                           | 结果是什么                             |
| -------------------- | ------------------------------------ | -------------------------------------- |
| AllGather            | 缺其他设备持有的不同数据块           | 按原索引拼齐，通信组内各端都有一份     |
| AllReduce（sum）     | 同一索引还缺其他设备的贡献           | 逐元素求和，各端都有完整结果           |
| ReduceScatter（sum） | 缺其他贡献，但最终只需保留一部分输出 | 先合并贡献，再让各端保留数值完整的分片 |
| AllToAll             | 数据所有者需要重新分配               | 重排分片，通常不做求和                 |

“完整”也有两层含义：一块数据可以只覆盖部分索引，却已经数值完整；也可以覆盖所有索引，却每个数都只是部分和。这个区别决定了是否能进入后续非线性操作。通常：

$$
\operatorname{ReLU}(P_0+P_1)\ne
\operatorname{ReLU}(P_0)+\operatorname{ReLU}(P_1).
$$

<div class="scaling-note" markdown="1">
**输出区域不完整，与数值贡献不完整，要分开判断。** 数值完整的输出分片，可以直接做逐元素非线性；尚未归约的部分和通常不可以。需要相加时，还要问这些贡献是否已经在本地，不是每次求和都需要网络 AllReduce。
</div>

### 通信公式里的体积，到底是谁的体积？

这里用 \\(V\_{\rm comm}\\) 表示通信公式中的数组体积，避免与 attention 的 Value 矩阵混淆。对于一个有 \\(n\\) 个设备的通信组：

| 操作          | 每端开始时                        | 每端结束时                              | \\(V\_{\rm comm}\\) 的定义   |
| ------------- | --------------------------------- | --------------------------------------- | ---------------------------- |
| AllGather     | 一块，大小 \\(V\_{\rm comm}/n\\)  | 拼齐后的 \\(V\_{\rm comm}\\)            | 组内拼齐的一份数组           |
| AllReduce     | 大小 \\(V\_{\rm comm}\\) 的部分和 | 大小相同的完整结果                      | 每端那个完整形状的部分和数组 |
| ReduceScatter | 大小 \\(V\_{\rm comm}\\) 的部分和 | 大小 \\(V\_{\rm comm}/n\\) 的完整结果块 | 归约前每端的部分和数组       |

在 AllReduce 中，不能因为有四个设备，就把每端 2 MiB 的完整形状部分和改称为 8 MiB 的结果。它们是**同一组索引的四份贡献**。其他 mesh 轴还可能切着数据，因此“通信组内拼齐”也不一定等于整个全局张量。

看单向环更容易推导成本。AllGather 从每端一块开始，经过 \\(n-1\\) 轮，每端每轮发送 \\(V\_{\rm comm}/n\\)。若单方向链路带宽为 \\(b\\)，忽略延迟：

$$
t_{\rm AG}=\frac{n-1}{n}\frac{V_{\rm comm}}{b}.
$$

同样大小数组的环形 ReduceScatter 有相似传输体积；AllReduce 可以分成 ReduceScatter 再 AllGather，因此有两阶段成本。原书在大消息、双向环的近似下，把可用双向带宽记为 \\(W\\)，得到 \\(t\_{\rm AG}\approx V\_{\rm comm}/W\\)、\\(t\_{\rm AR}\approx2V\_{\rm comm}/W\\)。有限设备数、启动延迟、拓扑与带宽定义都需要保留，不能将单向式和双向式混用。

ReduceScatter 的环上过程也不是“每端只管自己最后留下的那块”。每端对各输出块都有贡献；块在环上传递时，与沿途设备的对应贡献相加。最后每端留下一个完整求和块，AllGather 再把这些块拼齐。

### 前后两层一起设计，通信才可能真正减少

设两层为 \\(H=\sigma(AW\_{\rm in})\\)、\\(C=HW\_{\rm out}\\)。将 \\(W\_{\rm in}\\) 按输出列切分，得到数值完整的 \\(H\\) 列块，可直接计算非线性。再把 \\(W\_{\rm out}\\) 按对应输入行切分，各端计算 \\(H_rW\_{{\rm out},r}\\)，最后求和得到 \\(C\\)。

这样，放大的中间表征 \\(H\\) 可以保持切分，不必为了进入下一层先 AllGather。它把“第一层的输出布局”变成了“第二层恰好需要的输入布局”。是否最终 AllReduce，还是只做 ReduceScatter，则继续取决于下一步需要什么输出布局。

## <span id="transformer">4. 回到 Transformer：从功能恢复矩阵</span>

知道矩阵乘法怎样计数之后，Transformer 的困难主要变成了另一件事：这么多矩阵，各自存的是什么、作用于什么？我发现从功能重新画一遍，比背一串符号更可靠。[原书第 4 章](https://jax-ml.github.io/scaling-book/transformers/)提供了完整预算的框架。

以下采用同一套记号：

| 符号                   | 含义                                               |
| ---------------------- | -------------------------------------------------- |
| \\(B\\)                | 独立请求或序列数                                   |
| \\(T\\)                | 每条序列本次处理的新位置数                         |
| \\(S\\)                | attention 可读取的来源位置数，含历史与本次新增位置 |
| \\(D,F,H\\)            | 模型宽度、MLP 中间宽度、每个头的宽度               |
| \\(N_Q,N\_{KV}\\)      | Query 头数、Key/Value 头数                         |
| \\(L,V\_{\rm vocab}\\) | Transformer 层数、词表大小                         |

默认讨论无 bias 的主要矩阵、gated MLP、因果自注意力，且 \\(N_QH=D\\)，各类头宽相同。归一化参数和逐元素开销不混入主要矩阵预算。

从模型入口到出口，主表征的宽度是 \\(D\\)：

```text
token ID → embedding 查表 → [B,T,D]
    → attention 子层：读取允许位置的内容，回到 [B,T,D]
    → MLP 子层：逐位置变换特征，回到 [B,T,D]
    → 重复 L 层 → 最终归一化
    → 词表投影 [D,V_vocab] → logits [B,T,V_vocab]
```

残差连接将每个子层写成 \\(x+f(x)\\)，因此分支输出要与输入同形状。常见 pre-norm 结构先归一化再进入分支。LayerNorm 沿每个 token 的 \\(D\\) 个特征统计，不跨不同 token；若特征维切到多设备，需要合并统计量，若每个 token 的完整特征在本地，则无需为此通信。

**参数是规则，激活是当前输入经过规则后得到的状态。** 训练完成后，\\(W_Q\\) 固定；\\(Q=XW_Q\\) 则随输入变化。Embedding 表也属于权重：输入端查表得到表征，输出端做词表打分。若两端共享同一参数表，只计 \\(DV\_{\rm vocab}\\) 个参数，而且不乘层数。

### Gated MLP：两条支路都读取完整输入

普通 MLP 是 \\(\sigma(XW\_{\rm in})W\_{\rm out}\\)。Gated MLP 多了一条由输入生成的调节支路：

$$
U=XW_u,\qquad G=\sigma(XW_g),\qquad
Z=U\odot G,\qquad Y=ZW_{\rm out}.
$$

\\(W_u,W_g\\) 都是 \\([D,F]\\)，\\(W\_{\rm out}\\) 是 \\([F,D]\\)。两条支路都接收完整的 \\(D\\) 维输入；得到的 \\(U,G\\) 都是 \\([B,T,F]\\)，逐元素相乘后仍是这个形状，而不是拼成两倍宽度。

可以把 \\(U\\) 理解为待使用的特征，\\(G\\) 为当前输入生成的调节量。它不必处于 0 到 1 之间，也不要求每个通道都有明确的人类语义。[GLU 变体论文](https://arxiv.org/abs/2002.05202)

这里有一个值得单独澄清的线性代数问题：**不同通道有不同的固定增益，仍然是线性变换。** 例如 \\(f(x_1,x_2)=(2x_1,3x_2)\\)，满足叠加原理。固定向量 \\(g\\) 的逐元素缩放，也可以写成 \\(U\operatorname{diag}(g)\\)。门控的关键在于 \\(G\\) 随输入变化；训练后权重固定，不代表门控激活固定。

因此，gated MLP 有三份主要权重，共 \\(3DF\\) 个参数；每个 token 依次承担三次投影，主要计算量为 \\(6DF\\)。这两个因子“3”和“6”分别来自矩阵份数与乘加计数，不能与 bf16 的 2 bytes 混在一起。

### Attention：匹配条件、来源描述与被读取的内容

对一个头，先从输入生成 Query、Key、Value，再执行：

$$
R=QK^\mathsf T/\sqrt H,\qquad
A=\operatorname{softmax}_{\rm source}(R+\mathrm{mask}),\qquad
O=AV.
$$

Query 提供当前读取者的匹配条件，Key 提供每个来源用于匹配的描述，Value 提供要传递的内容。\\(W_Q\\) 学习的是**从当前输入生成 Query 的规则**，不是保存一个固定查询；投影到学习方向的图像有助于理解，但这些方向不要求正交，也没有固定频率含义。

对于一个 Query，attention 输出通常是多个来源的加权组合，例如 \\(o=0.1v_1+0.7v_2+0.2v_3\\)。匹配分数为零不等于权重为零，因为 \\(e^0=1\\)；所有允许位置分数相同，会均匀读取。被 mask 的位置则在 softmax 前设为负无穷，使其权重为零。

各头先独立得到 \\(H\\) 维内容，拼接为 \\(N_QH=D\\)，再通过 \\(W_O\\) 组合成 \\(D\\) 维表征。**这里仍是模型内部的表征，并非词表上的概率。** 只有最后经过词表投影，维度才变成 \\(V\_{\rm vocab}\\)。

## <span id="kv">5. 只有一个新 token，仍然可以读取一整段历史</span>

这是把 attention 图像连接到推理预算时最值得展开的一步。单步生成意味着每请求只有一个新 Query 位置，即 \\(T=1\\)；它没有说来源位置也只有一个。

固定权重的标准因果推理中，追加未来位置不会改变旧位置的结果，所以可以在各层保留旧 Key/Value。新 token 生成自己的 Q/K/V，将新 K/V 追加到缓存，再用新 Q 读取整个允许的来源集合。历史的投影与 MLP 不必重跑，但历史 K/V 仍参与新一步 attention。

单请求、单头的 shape 链最直接：

$$
q:[1,H],\quad K:[S,H],\quad V:[S,H],
$$

$$
qK^\mathsf T:[1,S],\qquad AV:[1,H].
$$

来源增加，使求和项变多；它不使输出向量变宽。每个历史位置贡献固定大小的 K/V 行，缓存沿位置轴增长。

### GQA 共享的是 K/V，不是 Query 的读取结果

标准 Multi-Head Attention（MHA）中，每个 Query 头对应自己的 K/V 头。Grouped-Query Attention（GQA）让一组 Query 头共享指定的一组 K/V；Multi-Query Attention（MQA）则让全部 Query 头共享一组 K/V。这是架构层面对表示自由度、投影成本与缓存成本的取舍，并非无条件删掉现有模型的头。[GQA 论文](https://arxiv.org/abs/2305.13245)

以 \\(N_Q=32,N\_{KV}=8,H=128\\) 为例，每四个 Query 头共享一组 K/V。各 Query 可以不同，所以它们对同一 K 的匹配权重和最终输出仍可以不同。Q/K 点积要求的是**每头宽度匹配**，不要求两类头的数量相同。

对应的投影权重也由功能决定：

$$
W_Q:[D,N_QH],\quad W_K,W_V:[D,N_{KV}H],\quad W_O:[N_QH,D].
$$

把多个头合并成一次大矩阵投影，是计算组织方式；进入匹配时，还要恢复每个头的对应关系。完整激活 shape 为：

$$
Q:[B,T,N_Q,H],\qquad K,V:[B,S,N_{KV},H],
$$

$$
A:[B,N_Q,T,S],\qquad O:[B,T,N_Q,H].
$$

<div class="scaling-note" markdown="1">
**每个 Query 头只读分配给它的一个 KV 头，不会再遍历所有 KV 头。** 因此，主要匹配与加权读取计算量为 \\(4BTSN\_QH\\)，这里不再乘 \\(N\_{KV}\\)。减少 KV 头数会减少缓存和 K/V 投影，但不按相同比例减少所有 Query 的匹配与读取 FLOPs。
</div>

也不要丢掉 batch 轴。四条独立请求，各有自己的历史；它们共享模型权重，但不默认共享 K/V 内容。把四条历史无隔离地拼在一起，会允许请求之间相互读取信息，改变模型的语义。

假设已有 8192 个缓存位置，四条请求各处理一个新 token，可以逐步检查：

| 当前状态           | Shape                | 哪个维度在变化                |
| ------------------ | -------------------- | ----------------------------- |
| 新 Q               | \\([4,1,32,128]\\)   | 每请求只有一个新读取者        |
| 新 K/V，各自       | \\([4,1,8,128]\\)    | 每请求各新增一行缓存          |
| 追加后的 K/V，各自 | \\([4,8193,8,128]\\) | 来源位置从 8192 增至 8193     |
| Attention 权重     | \\([4,32,1,8193]\\)  | 每个 Query 对来源位置分配权重 |
| 所有头的输出       | \\([4,1,32,128]\\)   | 每头输出宽度仍为 128          |
| 拼接并经输出投影   | \\([4,1,4096]\\)     | 回到模型宽度，尚未到词表      |

### 缓存容量怎样数

若每个 KV 元素用 \\(p\_{kv}\\) bytes，\\(L\\) 层、\\(B\\) 条等长请求、每条已有 \\(S\\) 个来源位置，则：

$$
\boxed{C_{KV}=2LBSN_{KV}H p_{kv}.}
$$

第一个 2 来自 K 和 V 两份数据；bf16 的 2 bytes 是另一个因子。每条请求各增加一个位置，缓存增量为 \\(2LBN\_{KV}H p\_{kv}\\)。请求长度不同时，用长度之和替代 \\(BS\\)。实际分配还需要考虑预留、padding 和碎片。

标准 MHA 可用 \\(N\_{KV}H=D\\) 化简，GQA 一般不可以。MLP 的中间宽度 \\(F\\) 则根本不决定 KV 行宽——它属于模型中另一条数据路径。

## <span id="memory">6. 内存预算看生命周期，优化看改变了什么</span>

算出每个张量大小以后，还不能把所有出现过的张量相加当作峰值。正确的问题是：**某个时刻，哪些数据必须同时存在？**

$$
C_{\rm peak}=\max_t\sum_{i\text{ 在 }t\text{ 时存活}}\operatorname{bytes}(i).
$$

纯前向串行执行时，许多中间激活用完即可释放或被覆盖；残差输入需保留到加法；KV cache 要跨生成步骤保留。权重和各层 KV 有层数因子，不意味着临时激活峰值也简单乘层数。

例如一个 gated MLP，残差输入占 64 MiB，两条中间支路各 192 MiB。两支路都就绪时占 448 MiB；若门控乘积覆盖其中一条支路，释放另一条后剩 256 MiB。分配 64 MiB 输出后是 320 MiB。仅这几步便能看出，峰值取决于存活集合和原地操作，而不是最后输出多大。

### 训练为什么保存激活，又为什么可以重算？

对于 \\(Y=XW\\)，令上游梯度为 \\(G=\partial\mathcal L/\partial Y\\)。

$$
\frac{\partial\mathcal L}{\partial W}=X^\mathsf TG,\qquad
\frac{\partial\mathcal L}{\partial X}=GW^\mathsf T.
$$

我更容易通过“汇总谁的贡献”恢复它们。每个权重被很多 token 使用，所以 \\(dW\\) 要沿 token 轴求和；单 token 对权重的梯度是输入与输出梯度的外积。每项输入又影响多个输出特征，所以 \\(dX\\) 要沿输出特征求和。结果 shape 必须与求导对象相同；全是方阵时还要看求和指标，不能只靠 shape 猜。

这也解释了为什么反向需要前向输入 \\(X\\)。可以把它一直保存，也可以保存更早的状态，轮到反向时重新计算。按 block checkpoint，长期保留块输入，反向时重建当前块内部激活，便是用额外计算换较少的长期存储。

对主要矩阵乘法，前向和两个梯度各约 \\(2MDF\\) FLOPs。若完整重跑一次前向，前向加反向的预算从三个单位变为四个单位。但实际内存峰值仍需加上当前重算中间量、梯度和工作区；运行时间也不保证正好增加三分之一。[原书反向与 checkpointing](https://jax-ml.github.io/scaling-book/transformers/#gradient-checkpointing)

### FlashAttention：不保存完整方阵，仍然计算位置配对

逻辑上存在 \\(T\times S\\) 个 attention 分数，不代表必须把整张表写进 HBM。对单 Query，输出可写成：

$$
o=\frac{\sum_s e^{r_s}v_s}{\sum_s e^{r_s}}.
$$

在共同指数尺度下，对两个来源块分别累计分子向量 \\(n_1,n_2\\) 和分母标量 \\(z_1,z_2\\)，最后合并为 \\((n_1+n_2)/(z_1+z_2)\\)。不需要存下全部最终概率；但不能把各块分别归一化后的输出直接相加。实际稳定算法还需维护最大分数，并将不同块的累积量重缩放到同一尺度。

[FlashAttention](https://arxiv.org/abs/2205.14135)利用分块与融合减少中间结果的 HBM 读写。标准 dense attention 仍要计算允许的位置配对；无历史且 \\(T=S\\) 时，主要矩阵乘法仍随 \\(T^2\\) 增长。**减少存储与搬运，不等于取消对应的数学运算。**

MoE 则从另一方向改变预算：模型拥有多个专家，但每个 token 只调用其中几个。若有 \\(E\\) 个专家、每个 \\(P_e\\) 参数、每 token 选 \\(k\\) 个，则专家池参数为 \\(EP_e\\)，每 token 使用的是 \\(kP_e\\)。前者决定完整专家池的权重容量，后者影响主要专家 FLOPs；还要另计路由、负载不均与通信。这再次说明，“拥有的数据”“本次使用的数据”和“累计搬运的数据”不能混成一个数。

## <span id="budget">7. 把四章连起来：一次完整的生成预算</span>

先把每层主要账目汇总。以下 K/V 投影只处理本次新增的 \\(T\\) 个位置，历史已经缓存；匹配部分按完整 \\(T\times S\\) 矩形计数。

| 模块           | 参数个数         | 主要前向 FLOPs     |
| -------------- | ---------------- | ------------------ |
| Gated MLP      | \\(3DF\\)        | \\(6BTDF\\)        |
| Q/O 投影       | \\(2DN_QH\\)     | \\(4BTDN_QH\\)     |
| K/V 新位置投影 | \\(2DN\_{KV}H\\) | \\(4BTDN\_{KV}H\\) |
| 匹配与加权读取 | 无新增权重       | \\(4BTSN_QH\\)     |

因此：

$$
P_{\rm layer}=3DF+2DH(N_Q+N_{KV}),
$$

$$
P_{\rm model}\approx LP_{\rm layer}+DV_{\rm vocab}
\quad\text{（共享词表权重）}.
$$

一个容易被“每 token 约两倍参数量 FLOPs”掩盖的部分，是不含新增权重的 \\(QK^\mathsf T\\) 和 \\(AV\\)。没有历史、\\(T=S\\) 时，它们与层内权重矩阵乘法的计算量之比为：

$$
\frac{4BT^2D}{2BT P_{\rm layer}}=\frac{2TD}{P_{\rm layer}}.
$$

Batch 在比值中约掉，但两项绝对计算量都随 batch 增长。令 gated MLP 的 \\(F=8D/3\\)、\\(N\_{KV}=N_Q/4\\)，有 \\(P\_{\rm layer}=10.5D^2\\)，两项在 \\(T=5.25D\\) 附近相等。这个分界依赖架构，也依赖是否按完整方阵计数。因果内核若确实跳过上三角，需改用实际允许的位置对数，不能机械套同一个常数。

### 四条请求，各向前生成一步

采用学习中的假想模型：32 层，\\(D=4096\\)、\\(F=11008\\)、32 个 Query 头、8 个 KV 头、每头宽度 128、词表 32000。Gated MLP，共享输入/输出词表权重，权重和 KV 均为 bf16。忽略 norm、bias 等较小参数及逐元素开销，不做设备切分。

四条独立请求各已有 8192 个缓存位置，本次各处理一个新 token，来源数变为 8193。首先检查容量：

| 项目                             |                     结果 |
| -------------------------------- | -----------------------: |
| 每层主要参数                     |              177,209,344 |
| 全模型主要参数                   | 5,801,771,008，约 5.802B |
| 权重存储                         |               10.807 GiB |
| 当前四条请求的全部 KV            |                    4 GiB |
| 四条请求各增加一个位置的 KV 增量 |                  0.5 MiB |

这里 B 参数用十进制，GiB 用 \\(2^{30}\\) bytes，MiB 用 \\(2^{20}\\) bytes。权重加当前 KV 约 14.807 GiB，还没有包含临时激活、工作区或其他运行开销。

再看计算量，明确统计的是**整个 batch 的一步**：

| 项目                                           | 32 层合计的主要 FLOPs |
| ---------------------------------------------- | --------------------: |
| Gated MLP：\\(6LBD F\\)                        |         34.628 GFLOPs |
| Q/K/V/O 新位置投影：\\(4LBDH(N_Q+N\_{KV})\\)   |         10.737 GFLOPs |
| 历史匹配与读取：\\(4LB\times8193\times N_QH\\) |         17.182 GFLOPs |
| 最后一次词表投影：\\(2BDV\_{\rm vocab}\\)      |          1.049 GFLOPs |
| 合计                                           |     **63.596 GFLOPs** |

这张表也把最容易混淆的几个尺度分开了：MLP 和投影按本次新 token 数计；attention 按 Query 数与来源长度计；KV 容量按请求、层、历史位置和 KV 头计；词表投影只出现在模型出口。

### 预算怎样变成性能问题

到这里仍不能只拿 63.596 GFLOPs 除以一个算力，就声称得到生成延迟。还得回到前三章，选定执行方案：

- 权重留在哪一级存储？同一 batch 内能否复用，下一步是否要重读？
- KV 在各 Query 头之间怎样共享读取？实际 HBM 流量是多少？
- 哪些张量切分，哪条链路传什么，最终输出布局是什么？
- 哪些阶段能重叠，哪些必须等待完整数值结果？

例如，**仅作一阶教学估算**：若这些权重在 HBM 中，每一步各读一次，且理想复用让每份历史 KV 也只读一次，权重与历史 KV 两项约产生 15.899 GB 读流量。假设相应 HBM 带宽为 1 TB/s，这两项给出约 15.9 ms 的搬运下限；若适用计算峰值为 100 TFLOPs/s，主要矩阵计算下限约为 0.636 ms。这个模型提示应优先审视数据供给。

它不是实测结论：片上驻留可以减少这类 HBM 读取，重复加载、临时量和其他算子又可能增加流量，分布式执行还会改变每设备预算与通信。真正的下一步是测量并解释模型与观测之间的差异。

<div class="scaling-note" markdown="1">
**我希望记住的是一套可重建的分析顺序：从算子功能恢复 shape，从 shape 数计算与存储，再沿数据路径统计流量，最后根据依赖关系组合时间。** 每次说“更快”“更省”时，都明确究竟改变了哪一项，以及代价转移到了哪里。
</div>

前四章把这套语言建立起来了。后续讨论训练并行与推理服务时，batch、量化、缓存、切分和调度，都可以放回这张数据流图里分析，而不必把它们记成互相独立的技巧。

本文主要依据 [How to Scale Your Model 第 1 章](https://jax-ml.github.io/scaling-book/roofline/)、[第 2 章](https://jax-ml.github.io/scaling-book/tpus/)、[第 3 章](https://jax-ml.github.io/scaling-book/sharding/)、[第 4 章](https://jax-ml.github.io/scaling-book/transformers/)，以及文中链接的原始论文。更基础的架构介绍见本站[《理解 Transformer（1）》](/posts/2026/09/transformer-experimentalist/)。
