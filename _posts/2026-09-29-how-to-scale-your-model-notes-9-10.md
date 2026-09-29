---
layout: post
title: "How To Scale Your Model 笔记(9-10)"
date: 2026-09-29
permalink: /posts/2026/09/how-to-scale-your-model-notes-9-10/
description: "从执行时间线辨认计算、搬运与等待，用 JAX 表达数组分片和集体通信，再追踪 MoE 的分组、分发与返回合并。"
tags:
  - Machine Learning
  - Transformer
  - ML Systems
  - Learning Notes
related_posts: true
math: true
---

同一个矩阵表达式，可以对应不同的本地矩阵形状、权重读取次数、通信路线和临时缓冲区。性能分析需要把三件事连起来：**数学上必须完成什么，编译器安排了什么，设备实际何时完成。** 第 9 章的 profiling 检查执行过程，第 10 章的 JAX 分片与通信描述这个过程。

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
.scaling-figure { margin: 1.5rem 0; }
.scaling-figure .figure-scroll { overflow-x: auto; }
.scaling-figure svg { display: block; width: 100%; min-width: 660px; height: auto; color: var(--global-text-color, #222); }
.scaling-figure text { fill: currentColor; font-family: inherit; font-size: 16px; }
.scaling-figure .scaling-compute { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.26; }
.scaling-figure .scaling-wait { fill: currentColor; fill-opacity: 0.07; }
.scaling-figure .scaling-transfer { fill: var(--global-theme-color, #1b87ad); fill-opacity: 0.6; }
.scaling-figure .scaling-stroke { stroke: var(--global-theme-color, #1b87ad); stroke-width: 1.5; fill: none; }
.scaling-figure figcaption { font-size: 0.85rem; color: var(--global-text-color-light, #666); margin-top: 0.5rem; }
</style>

<nav class="scaling-nav" aria-label="文章目录">
<a href="#measurement">1. 测量与关键路径</a>
<a href="#program">2. 从表达式到执行图</a>
<a href="#budget">3. 预算与优化收益</a>
<a href="#sharding">4. JAX 的数组视角</a>
<a href="#ring">5. 输入流动、权重驻留</a>
<a href="#moe">6. MoE 的计算组织</a>
<a href="#validation">7. 正确性与性能验证</a>
</nav>

## <span id="measurement">1. 测量的终点，以及真正决定它的依赖</span>

### 函数返回、结果就绪、取回主机是三个事件

JAX 可以异步提交计算。Python 得到数组句柄时，shape 和 dtype 已知，设备上的数值计算却可能尚未完成。`block_until_ready()` 等待结果就绪，本身不要求把结果复制到主机；打印数组或转换为 NumPy 数组则可能引入传输。

测稳态单次调用时，应先完成同配置编译与预热，使输入驻留并就绪，再测调用到输出就绪之间的时间：

```python
import time

# f 已经由 jax.jit 包装；x、w 为设备数组。
jax.block_until_ready((x, w))
jax.block_until_ready(f(x, w))  # 编译、预热，不计入本次计时

t0 = time.perf_counter()
y = f(x, w)
jax.block_until_ready(y)
elapsed = time.perf_counter() - t0
```

这个墙钟时间包含主机提交与同步成本。冷启动延迟、含输入传输的请求延迟、设备内算子时间，测量边界各不相同；应按目标指标选择，而非统一排除所有开销。[JAX 计时说明](https://docs.jax.dev/en/latest/benchmarking.html)

### 长通信条可能是在等另一端

设两张卡参与一次 AllReduce。下面是一个受控示意：卡 0 在 4 ms 准备好，卡 1 在 9 ms 准备好；完整输入都就绪后，通信固定需要 2 ms。图中的“通信调用”包含等待。

<figure class="scaling-figure">
<div class="figure-scroll">
<svg viewBox="0 0 840 260" role="img" aria-labelledby="timeline-title timeline-desc">
<title id="timeline-title">两设备准备时间不同，通信调用包含等待</title>
<desc id="timeline-desc">卡零计算零至四毫秒、等待四至九毫秒、通信九至十一毫秒；卡一计算零至九毫秒、通信九至十一毫秒。最终时间由卡一就绪决定。</desc>
<text x="105" y="30">0</text><text x="337" y="30">4</text><text x="627" y="30">9</text><text x="738" y="30">11 ms</text>
<text x="20" y="88">卡 0</text><text x="20" y="153">卡 1</text>
<rect class="scaling-compute" x="110" y="54" width="232" height="52" rx="3"/>
<rect class="scaling-wait" x="342" y="54" width="290" height="52"/>
<rect class="scaling-transfer" x="632" y="54" width="116" height="52" rx="3"/>
<rect class="scaling-compute" x="110" y="119" width="522" height="52" rx="3"/>
<rect class="scaling-transfer" x="632" y="119" width="116" height="52" rx="3"/>
<text x="186" y="86">本地计算</text><text x="445" y="86">等待卡 1</text><text x="672" y="86">通信</text>
<text x="331" y="151">本地计算</text><text x="672" y="151">通信</text>
<path class="scaling-stroke" d="M342 184 v10 H748 v-10"/>
<text x="405" y="220">卡 0 的通信调用：7 ms</text>
<text x="221" y="250">准备阶段 max(4, 9) + 通信 2 = 总时间 11 ms</text>
</svg>
</div>
<figcaption>示意时间线，非实测 trace。等待是否与分块传输交织，需要更细的记录判断。</figcaption>
</figure>

在这些假设下：

$$
T=\max(t_0,t_1)+2\ \mathrm{ms}.
$$

只把卡 0 的计算从 4 ms 减到 2 ms，最终仍在 11 ms 完成，它的通信调用反而从 7 ms 变成 9 ms。条变长，并不能据此认定网络变慢。应沿依赖找出哪一端、哪一个生产者限制了最终输出；这条决定完成时间的依赖链就是**关键路径**。

确定卡 1 就绪较晚之后，还需要继续区分：上游输入迟到、本地搬运较多、矩阵执行效率较低，还是存在额外重排。相同 shape、dtype 和理论 FLOPs，不能排除这些差异。数据已经在 HBM 中，也不表示它已经进入片上缓冲、可以立即参与全部计算。

<div class="scaling-note" markdown="1">
**优化应缩短目标输出的完成时间。** 一条事件的持续时间、某项资源的忙碌时间，以及它对关键路径增加的时间，是不同的量。慢端加速后，关键路径还可能切换到另一端。
</div>

### 存了多少、搬了多少、搬得多快

HBM（High Bandwidth Memory，高带宽内存）容量统计同时驻留的数据；流量统计指定接口上累计通过的数据；带宽统计传输速率。若该接口的速率为 \\(r(t)\\)，完整执行窗口内的累计字节数是：

$$
D_{\mathrm{HBM}}=\int_{t_0}^{t_1}r(t)\,dt.
$$

同一临时缓冲区反复写回和读入，可能显著增加累计流量，却不增加峰值占用。单次突发大小或峰值带宽相同，也不能说明总流量相同。比较流量时，应固定接口、读写口径和完整测量窗口。

内存峰值则取决于数据生命周期。概念上可写为：

$$
M_{\mathrm{peak}}=\max_t\left[
M_{\mathrm{persistent}}+
\sum_{i\in\mathrm{live}(t)}M_i
\right].
$$

这里要按实际缓冲分配理解，避免把共享存储重复计数。权重能装下，不保证权重、激活、输出和工作区同时存活时仍能装下；不同时存活的缓冲也可能被复用。

## <span id="program">2. 从数学表达式读到编译后的执行图</span>

JAX 提供数组运算及自动微分、编译等变换。`jax.jit` 中的 JIT 是 just-in-time compilation，即即时编译。一次典型的编译执行路径是：

```text
Python 函数
  → tracing：记录 JAX 运算及依赖，形成 jaxpr
  → lowering：转成 StableHLO 等编译器表示
  → compilation：优化并生成设备可执行程序
  → 调用、提交、执行
  → 所需结果就绪
```

**XLA**（Accelerated Linear Algebra）是编译器；**HLO**（High Level Operations）是其高层操作表示；**IR**（Intermediate Representation）是中间表示。StableHLO 用于框架与编译器之间的接口，XLA 内部继续优化 HLO。原书 TPU 路径还介绍了 **LLO**（Low-Level Optimizer）及更接近硬件搬运、调度的低层表示。[XLA 架构](https://openxla.org/xla/architecture)

Tracing 会执行 Python 函数体来记录运算结构，但不等于已经用真实输入完成矩阵乘法。也可以显式拆开阶段：

```python
f = jax.jit(matmul)
lowered = f.lower(x, w)
compiled = lowered.compile()
y = compiled(x, w)
y.block_until_ready()
```

普通 `jit` 调用会缓存适用的编译版本。固定其他条件，动态数组只换数值通常可复用程序；shape 改变通常需要新版本。直接调用上面的 `compiled` 对象时，输入形状不兼容会报错，而不会自动替它编译新版本。**缓存的是程序，不是上一次计算结果。** [JAX 编译阶段](https://docs.jax.dev/en/latest/aot.html)

### 求和索引比权重的书写方向更可靠

`einsum('bf,fd->db', x, w)` 中，字符串给数组轴命名；`f` 在输入出现、在输出消失，因此沿它求和，输出轴顺序为 `d,b`：

$$
z_{d,b}=\sum_f x_{b,f}w_{f,d}.
$$

若 `x.shape=(128,256)`、`w.shape=(256,16)`，输出就是 `[16,128]`。改成 `'bf,fd->bd'` 则得到 `[128,16]`。这是逻辑轴顺序，不足以断定设备必然额外执行一次完整转置拷贝。

另一些权重按 **[输出特征，输入特征]** 存放。此时单向量形式为 \\(h_i=\sum_jW\_{ij}x_j\\)，把多个输入向量按行排列后，就是 \\(H=XW^{\mathsf T}\\)。恢复计算时先找收缩维，不能默认权重第一轴总是输入维。

### 一条操作记录要结合三种视图读取

| 视图                     | 主要回答的问题                                   |
| ------------------------ | ------------------------------------------------ |
| Trace Viewer：时间线     | 操作何时开始、结束，哪些活动重叠，哪里出现等待？ |
| Graph Viewer：依赖图     | 输入来自谁，输出交给谁，一个 fusion 内包含什么？ |
| Memory Profile：内存视图 | 哪些缓冲同时存在，哪个时刻形成峰值？             |

不要把不同层级时间轨道上的条全部相加：同一项工作可能同时出现在源码作用域和设备操作轨道中。还要区分**运行时记录**与**编译器估算**：XProf 的操作详情中，FLOPs 和 bytes accessed 可以来自 XLA 的静态分析，并非硬件计数器测得的累计量。[XProf Trace Viewer](https://openxla.org/xprof/trace_viewer)

例如 `bf16[32,32,4096]` 描述逻辑 dtype 与 shape；布局字段进一步描述轴在内存中的次序和分块。`f32[3,5]{1,0:T(2,2)}` 中，轴 1 是更快变化的轴，按 2×2 tile 存储需要补齐到 4×6 个槽位：15 个有效元素占用 24 个物理槽位。Padding 增加存储，不增加逻辑输入。[XLA shape 与 layout](https://openxla.org/xla/shapes)

跨设备 **sharding** 决定哪张卡持有哪些元素；设备内 **layout/tiling** 决定这些元素怎样存放。二者都可能引入搬运，但要分别追踪。操作名如 `fusion.3` 只是入口，应继续检查输入、输出、生产者及内部子图；名字本身不能确定全部计算或通信语义。

## <span id="budget">3. 用预算定位问题，用时间差判断优化</span>

### 局部 FLOPs 除以局部算力

原书给出一个历史 TPU v2 示例：8 个核心组成 4 路数据并行、2 路模型分片。上投影全局输入为 `[32,1024,8192]`，权重为 `[8192,32768]`；每核心实际做：

$$
[8\times1024,8192]\times[8192,16384].
$$

因此本地 FLOPs 与时间预算为：

$$
F_{\mathrm{local}}=2(8\times1024)8192\times16384,
\qquad
T_{\mathrm{compute}}\approx
\frac{F_{\mathrm{local}}}{23\times10^{12}}
\approx95.6\ \mathrm{ms}.
$$

这里的每核心 23 TFLOP/s 沿用原书近似预算，作者报告该操作约 96 ms。也可以用全局 FLOPs 除以 8 核总算力，但不能用局部 FLOPs 再除以总算力。

尾部两路 ReduceScatter 的每端完整归约输入为 128 MiB。沿用书中一跳、全双工合计 120 GB/s 的理想模型，每端发送半份、单方向带宽 60 GB/s，得到约 1.12 ms，对应作者报告的 1.13 ms。这些是原书环境中的对照，不是新的硬件测量。[第 9 章示例](https://jax-ml.github.io/scaling-book/profiling/#looking-at-a-realish-example-profile)

**局部算子接近预算，只说明这项工作执行得高效。** 若执行方案包含不必要的操作、重复读取或昂贵的布局转换，整个程序仍可能有优化空间。

### 同样的计算量，不同的权重复用

考虑一个两投影算例：

$$
H=XW_1^{\mathsf T},\qquad Y=HW_2^{\mathsf T},
$$

$$
X:[8,8192],\quad W_1:[32768,8192],\quad W_2:[8192,32768].
$$

用 8 个核心，初始各端持有一行输入，最终各端也要得到对应的一行完整输出。权重按各自方案预先驻留，容量足够。中间如有逐元素激活，也可在完整的中间特征上本地执行；以下只计两次投影的主要成本。

<div class="scaling-wide" markdown="1">

|          | A：每端处理自己的一行      | B：权重分片，共同处理八行                        |
| -------- | -------------------------- | ------------------------------------------------ |
| 输入     | 本地 `[1,8192]`            | AllGather 后 `[8,8192]`                          |
| 第一投影 | `[1,8192] × [8192,32768]`  | `[8,8192] × [8192,4096]`                         |
| 第二投影 | `[1,32768] × [32768,8192]` | `[8,4096] × [4096,8192]`                         |
| 输出     | 本地完整的一行             | `[8,8192]` 的部分和，再沿 batch 做 ReduceScatter |

</div>

B 的中间 \\(H_r:[8,4096]\\) 是全部八行对应的一部分**完整特征**；下投影的 \\(P_r:[8,8192]\\) 则在所有输出位置上只积累了部分收缩维的贡献，需要跨端求和。

两方案每核心的主要 FLOPs 相同。若本地权重在一次调用中只从 HBM 读取一次，B 的权重读取量是 A 的 1/8：每端权重更小，却被八行输入复用。代价是新增输入收集与输出归约。

若三个阶段不重叠、其他成本相同：

$$
T_A=T_{\mathrm{local},A},\qquad
T_B=T_{\mathrm{AG}}+T_{\mathrm{local},B}+T_{\mathrm{RS}}.
$$

所以 B 更快的条件是：

$$
\boxed{T_{\mathrm{local},A}-T_{\mathrm{local},B}
>T_{\mathrm{AG}}+T_{\mathrm{RS}}.}
$$

<div class="scaling-note" markdown="1">
**应拿省下的本地时间，与新增通信成本比较。** A 原来的全部读取时间不是收益，B 仍要读取自己的权重分片。读取如果原本被计算遮蔽，减少 bytes 也未必缩短本地阶段；本地矩阵形状变化还会改变算力利用率。存在重叠时，应回到完整关键路径比较。
</div>

## <span id="sharding">4. JAX：先分清全局数组与本地分片</span>

JAX 用 **mesh** 为设备建立命名逻辑坐标，用 **PartitionSpec** 描述数组各维沿哪些设备轴切分，再用 **NamedSharding** 将具体 mesh 与切分规则绑定。Mesh 的逻辑邻居不自动等于物理网络邻居。

必须同时给出 mesh 尺寸和数组形状。例如 mesh 为 `X=4, Y=2`：

| 全局数组       | PartitionSpec | 每设备持有    | 复制关系     |
| -------------- | ------------- | ------------- | ------------ |
| `In[8,2048]`   | `P('X','Y')`  | `[2,1024]`    | 无额外复制轴 |
| `W[2048,8192]` | `P('Y',None)` | `[1024,8192]` | 沿 X 复制    |

`P` 的位置对应**数组维度**，字符串对应 **mesh 轴名**。`None` 表示该数组维不切分；普通分片布局中，没被用于切分的 mesh 轴承载副本。只看 `P('X','Y')`，无法知道 X、Y 各有多少设备。

`P(('X','Y'))` 的含义又不同：把**一个数组维度**联合沿 X、Y 切分，本例分成八份。[JAX 分布式数组](https://docs.jax.dev/en/latest/parallel.html)

### 三种模式改变的是布局与通信由谁决定

| 模式                 | 函数体看到什么 | 如何决定分布式执行                       |
| -------------------- | -------------- | ---------------------------------------- |
| Auto                 | 全局数组       | 编译器在约束下推断中间布局与通信         |
| Explicit             | 全局数组       | JAX 传播分片信息，歧义处要求用户指定布局 |
| Manual / `shard_map` | 本地块         | 用户写本地计算及 collective（集体通信）  |

**指定布局与手写通信是两回事。** 例如上表的 `In @ W`，每端 `[2,1024] × [1024,8192]` 得到 `[2,8192]`，但只完成了收缩维 2048 中的一半。若要求输出 `P('X',None)`，须在固定 X、沿 Y 的组内求和，并让 Y 两端都有完整结果，对应 AllReduce。若输出特征还要沿 Y 分片，则可采用 ReduceScatter。不同 X 负责不同输入行，不能互相相加。[原书 JAX 并行模式](https://jax-ml.github.io/scaling-book/jax-stuff/#how-does-parallelism-work-in-jax)

`with_sharding_constraint` 可以在 JIT 函数内部约束中间数组的布局。它是严格约束，不是“请尽量优化”的提示；满足约束可能需要额外通信，也可能改善上下游衔接。效果必须检查编译后的分片与通信，并测量完整执行。[分片约束接口](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html)

### `shard_map` 的本地视角由包装器建立

下面的等价写法把绑定关系直接写出来；假设 `mesh` 已定义，Y 有四个设备：

```python
def local_work(x):
    return x[:4]

f = jax.shard_map(
    local_work, mesh=mesh,
    in_specs=P('Y'), out_specs=P('Y'),
)
```

调用 `f(global_array)` 时，`in_specs` 决定每个函数实例拿到的局部块。若全局数组有 256 项，各端看到 64 项，`x[:4]` 就取**各自分片的前四项**，最后拼为 16 项。变量叫 `local_x` 还是 `x` 不影响语义。写成 `@jax.shard_map(...)` 装饰器时，应与紧随其后的函数定义一起阅读。

输出规格描述如何把局部返回值解释为全局数组。`out_specs=P()` 声明输出在未使用的设备轴上相同，并不会自动计算平均或求和；需要的数值归约必须由 `pmean`、`psum` 等操作完成。[`shard_map` 的输入与输出语义](https://docs.jax.dev/en/latest/notebooks/shard_map.html)

## <span id="ring">5. Collective matmul：输入流动，权重驻留</span>

改变布局，令输入 \\(A[B,D]\\) 沿 D 切分，权重 \\(W[D,F]\\) 沿 F 切分。设备 \\(j\\) 持有部分输入特征 \\(A_j\\)，以及自己负责的输出列对应的**全部权重行** \\(W\_{:,j}\\)。它最终要算：

$$
C_j=\sum_{r=0}^{N-1}A_rW_{r,j}.
$$

这里 \\(r\\) 是输入特征块编号，\\(j\\) 是输出列块编号。可以先 AllGather 完整 A，再计算本地输出列；也可以先用已有输入块计算，然后逐块传入其他输入，每收到一块就匹配相应权重行并累加。

全部 r 覆盖之后，本地 \\(C_j\\) 已经是完整的输出列分片，末尾无需再做输出 AllReduce。求和所需的信息，已经通过**输入交换**到达这张设备。

### 设备编号固定，当前输入的来源在变

用四设备、\\(A:[2,8]\\)、\\(W:[8,8]\\) 举例。每端 `a.shape=(2,2)`、`w.shape=(8,2)`。交换路线是：

```python
perm = [(0, 3), (1, 0), (2, 1), (3, 2)]
```

每一对是 `(发送者, 接收者)`，整张表描述**一次交换**，不是按列表顺序连续转发四次。设备 `idx` 从 `idx+1` 接收，所以第 `step` 轮当前输入块的原始来源为：

```python
source = (idx + step) % 4
```

设备 2 始终负责全局输出列 `4:6`，但依次处理：

| 轮次 | 当前输入   | 本地权重取哪些行 | 对输出的贡献         |
| ---- | ---------- | ---------------- | -------------------- |
| 0    | `A[:,4:6]` | `w[4:6,:]`       | 输入特征 4、5 的贡献 |
| 1    | `A[:,6:8]` | `w[6:8,:]`       | 输入特征 6、7 的贡献 |
| 2    | `A[:,0:2]` | `w[0:2,:]`       | 输入特征 0、1 的贡献 |
| 3    | `A[:,2:4]` | `w[2:4,:]`       | 输入特征 2、3 的贡献 |

Python 切片右端不包含在内；`A[:,4:6]` 保留全部两行和两列，shape 是 `[2,2]`。`w` 已经只含设备 2 负责的两列，因此循环只切它的行。

下面是完整的小规模语义验证例子。它用 CPU 模拟四个设备，不用于衡量真实互联性能；应在新 Python 进程中运行，使设备数量设置先于后端初始化。

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import numpy as np
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 新版 JAX 提供 jax.shard_map；兼容旧版接口。
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

mesh = Mesh(np.array(jax.devices("cpu")[:4]), ("Y",))

@jax.jit
@partial(shard_map, mesh=mesh,
         in_specs=(P(None, "Y"), P(None, "Y")),
         out_specs=P(None, "Y"))
def ring_matmul(a, w):
    idx = jax.lax.axis_index("Y")
    width = a.shape[1]
    current = a
    perm = [(0, 3), (1, 0), (2, 1), (3, 2)]
    for step in range(4):
        source = (idx + step) % 4
        rows = jax.lax.dynamic_slice_in_dim(
            w, source * width, width, axis=0
        )
        update = current @ rows
        result = update if step == 0 else result + update
        if step < 3:
            current = jax.lax.ppermute(current, "Y", perm)
    return result

a_host = np.arange(16, dtype=np.float32).reshape(2, 8)
w_host = np.arange(64, dtype=np.float32).reshape(8, 8)
placement = NamedSharding(mesh, P(None, "Y"))
a = jax.device_put(a_host, placement)
w = jax.device_put(w_host, placement)
actual = np.asarray(ring_matmul(a, w))
expected = a_host @ w_host
assert actual.shape == expected.shape
np.testing.assert_array_equal(actual, expected)
```

该例已在 JAX 0.4.38 的四个 CPU 逻辑设备上通过数值对照；这里只验证索引、通信语义与输出，不代表 TPU/GPU 的性能结果。

`ppermute` 只按路线传值，不求和；累加发生在 `result + update`。最后一块算完就结束，不需要再传一轮。两设备例子中的 `1-idx` 只表示互换，不能直接用于四设备。[`ppermute` 接口](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html)

### 通信与计算可以重叠，仍需检查实际调度

传递当前输入块与用它做本地乘法，在依赖上可以并行；下一块乘法则必须等相应输入到达。这给流水重叠提供了机会，但源码排列本身不是硬件实际重叠的证据。分块还可能让矩阵乘法变小、降低利用率，并增加启动与调度成本。

<div class="scaling-note" markdown="1">
**AllGather 大条消失，不表示通信消失。** 分块传递仍在搬运输入。需要观察的是通信有多少延长了关键路径，即 exposed communication，以及完整矩阵乘法是否更快。
</div>

原书报告默认方案 311 μs、collective matmul 244 μs，另一个计算参照为 224 μs。这是特定环境的结果；`244−224` 不能直接当作精确通信耗时，因为分块与布局也会改变计算效率。[原书 collective matmul](https://jax-ml.github.io/scaling-book/jax-stuff/#manual-sharding-mode-via-shard_map)

## <span id="moe">6. MoE：先按专家分组，最后按原 token 合并</span>

### 路由的是当前层的隐藏向量

常见 Transformer MoE（Mixture of Experts，混合专家）用多个专家替换部分或全部 FFN/MLP 子层，Attention 仍然保留。一个专家通常是完整 MLP；原书练习用单矩阵专家简化计算，重点研究路由和通信。

“按 token 路由”是指按**当前层每个位置的隐藏向量**选择专家。Attention 之后仍是 \\(H:[S,D]\\) 的位置表示，不是已经生成的 token ID。只有经过全部模型层、词表投影和选择，才得到下一个输出 token。

```text
各位置的隐藏向量
  → router 选择专家
  → 按目的设备和专家分组、分发
  → 专家处理各行
  → 输出返回，恢复原 token 对应关系
  → 进入后续子层
```

### Router 与专家怎样共同学习

补充典型学习式路由：对一行隐藏向量 h，router 用参数 \\(R:[D,E]\\) 算专家分数，再选择 top-k：

$$
z=hR,\qquad p=\operatorname{softmax}(z),\qquad
y=\sum_{e\in\operatorname{TopK}(p)}g_e\operatorname{Expert}_e(h).
$$

\\(g_e\\) 的归一化规则由模型决定。Switch 的 top-1 形式保留所选专家的门控权重 \\(p_e\\)，使预测损失可经门控值训练 router；离散 `argmax` 本身不是普通可微函数。上游表示、路由器和专家共同调整，辅助负载均衡目标抑制过度拥挤。[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)

隐藏向量在**表示空间**中，router 和专家的权重在参数空间中。路由学习的是选择哪些处理函数有利于任务，不要求形成数学、物理、生物等人类学科分类。Mixtral 的特定分析中，路由与语法、token 形式的关联比主题分工更明显；这不是所有 MoE 的必然规律。[Mixtral 路由分析](https://arxiv.org/html/2401.04088v1#S5)

### 总参数与每个 token 的激活计算

用忽略偏置的两投影 MLP 说明比例。模型维 D、中间宽度 H 时，参数约 \\(2DH\\)，每 token 主要 FLOPs 约 \\(4DH\\)。若保持专家总参数相同，分成 E 个宽度 H/E 的专家，每行只执行其中 k 个，则专家部分的计算比例约为：

$$
\frac{F_{\mathrm{MoE,experts}}}{F_{\mathrm{dense,MLP}}}\approx\frac{k}{E}.
$$

这个比较忽略 router、通信、激活函数及其他模型子层。它表达的是**减少激活计算的机会**，不保证保留某个固定百分比的模型能力。反过来，也可以固定每行激活的专家大小和数量，增加总专家数，以更多参数容量换取任务表现；效果要通过训练评估。

稀疏执行在 prefill 和 decode 都适用。一批输入合起来仍可能访问所有专家，模型存储、HBM 流量和全模型延迟不会必然按 k/E 缩小。

### 不规则路由如何变成矩阵乘法

设四行输入按顺序为 \\(a_0,a_1,a_2,a_3\\)，专家编号为 `[1,0,1,1]`。先分组：

```text
原顺序：     [a0, a1, a2, a3]
按专家排列： [a1 | a0, a2, a3]
组长度：     [1, 3]
```

专家 0 计算 `[1,D] × [D,F]`，专家 1 计算 `[3,D] × [D,F]`。每行保持独立，同一组只是共享权重。结果 `[o1,o0,o2,o3]` 根据保存的原位置恢复为 `[o0,o1,o2,o3]`。

若每专家固定预留 C 行，输入形状变成 `[E,C,D]`。本例取 C=3，总共六个槽位、四行有效输入，普通稠密计算还会处理填充行。平均负载 S/E 不保证容纳最忙专家；要对任意 top-1 路由都不溢出，最坏 C=S。容量较小的实现必须另行规定溢出处理。

另一种方式把有效行紧凑存放为 `[S,D]`，用 `group_sizes` 记录各组边界。这是 **ragged（不等长）** 表示。`jax.lax.ragged_dot` 接收分组后的输入、专家权重 `[E,D,F]` 和组长度，完成相应矩阵乘法。它避免应用层给每专家补满固定容量，但不保证消除内核对齐开销，也不自动解决设备负载不均。[`ragged_dot` 文档](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html)

如果先让所有专家处理全部输入，再 mask 掉未选结果，被丢弃的输出仍然可能已经花掉了计算时间。**稀疏选择要落实到执行的数据组织中。**

### 专家并行改变数据的目的地

专家分散在不同设备上时，dispatch 把激活送到选中专家所在设备。AllGather 让各端拿到全部输入；AllToAll 式分发则让不同分组到达各自目的地。后者仍需要打包、组长度、位置映射和输出返回，不能把一个通信名称当成完整路由实现。

例如设备 0 初始有 \\(h_0,h_1\\)，驻留专家 0；设备 1 有 \\(h_2,h_3\\)，驻留专家 1。若路由为 `[1,0,0,1]`，只需将 \\(h_0\\) 发往设备 1、\\(h_2\\) 发往设备 0，其余留在本地。专家计算后，对应输出再返回原位置。各组大小不同时，通信接口还需通过填充、分块或变长机制承载这种不规则性。

### Top-k 的合并索引是 token，不是专家

Top-1 可以恢复排列；top-k 中，每个 `(token, expert)` 是一条调用记录，共有 Sk 条。每条记录需要保留原 token 编号、专家编号、门控权重及返回对应关系。Sk 条记录不等于 Sk 次独立跨设备发送：本地专家不需跨设备，同设备多个专家也可能共享一次输入传输。

设 \\(h_0\\) 对专家 0、1 的权重为 0.7、0.3，\\(h_1\\) 为 0.2、0.8。输出以任意顺序返回：

| 返回值 | 对应计算                            | 应放回哪里       |
| ------ | ----------------------------------- | ---------------- |
| u      | \\(\operatorname{Expert}\_1(h_1)\\) | `out[1]`，乘 0.8 |
| v      | \\(\operatorname{Expert}\_0(h_0)\\) | `out[0]`，乘 0.7 |
| w      | \\(\operatorname{Expert}\_1(h_0)\\) | `out[0]`，乘 0.3 |
| z      | \\(\operatorname{Expert}\_0(h_1)\\) | `out[1]`，乘 0.2 |

所以：

$$
\mathrm{out}[0]=0.7v+0.3w,\qquad
\mathrm{out}[1]=0.2z+0.8u.
$$

<div class="scaling-note" markdown="1">
**计算前按专家分组，计算后按原 token 加权归并。** `v+z` 虽然来自同一专家，却属于两个不同 token，不能合成一行。原书的多专家结果平均，是等权重的特例。
</div>

有了对齐的记录，可以用 `out.at[token_ids].add(weights[:, None] * expert_outputs)` 表达返回合并。若专家内部还使用张量并行，它自己的部分和归约是另一层操作，不能与跨专家、按 token 的加权组合混为一谈。[原书 MoE 练习](https://jax-ml.github.io/scaling-book/jax-stuff/#worked-problems)

## <span id="validation">7. 把优化假设变成可检验的对照</span>

先用小规模 reference 检查布局与数值，再评价性能。输入应有区分度：全零、全一或过强的对称性容易掩盖错配。环形矩阵乘法要覆盖每个输入块与对应权重行；MoE 要检查打乱返回顺序后，同一 token 的所有贡献仍被正确加权合并。

上面的整数值 float32 算例中，乘积和中间累加结果可精确表示，所以可以严格比较。一般低精度计算中，不同求和顺序会产生舍入差异，应使用适合精度、规模和用途的容差，例如：

$$
\lvert y-y_{\mathrm{ref}}\rvert
\leq\mathrm{atol}+\mathrm{rtol}\lvert y_{\mathrm{ref}}\rvert.
$$

然后固定硬件、工作负载、精度和计时边界，分别编译预热并重复测量。对照至少需要解释三件事：

| 要验证的判断       | 对应证据                                         |
| ------------------ | ------------------------------------------------ |
| 数学任务没有改变   | 输出 shape、索引对应、数值误差                   |
| 执行方案按预期改变 | 编译后的本地 shape、布局、collective 与依赖      |
| 目标指标改善       | 完整耗时、关键路径和内存峰值；必要时检查实际流量 |

例如“改变分片能复用权重”应进一步落到本地读取是否减少、执行时间是否缩短；“分块通信能隐藏等待”应落到时间线是否重叠、暴露通信是否减少；“MoE 减少激活计算”应同时检查分组效率、最忙专家和分发返回成本。代码表达的并行意图、编译器实现和实测结果，需要逐一对应。

原文：[第 9 章 Profiling](https://jax-ml.github.io/scaling-book/profiling/)、[第 10 章 JAX](https://jax-ml.github.io/scaling-book/jax-stuff/)。系列前文：[笔记(1-4)](/posts/2026/09/how-to-scale-your-model-notes-1-4/)、[笔记(5-6)](/posts/2026/09/how-to-scale-your-model-notes-5-6/)、[笔记(7-8)](/posts/2026/09/how-to-scale-your-model-notes-7-8/)。
