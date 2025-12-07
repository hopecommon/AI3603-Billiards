# **基于工程ROI最大化的随机环境八球智能体架构深度研究报告**

## **1\. 执行摘要与项目背景分析**

### **1.1 项目愿景与约束条件的深度解构**

在当前计算物理与人工智能交叉领域的研究中，针对复杂动力学系统的控制问题始终是一个核心议题。本次任务要求在 pooltool 这一高保真物理仿真环境中，设计并实现一个能够在地质级（Consumer-grade）硬件（RTX 4060）上高效运行，并能显著击败基准对手 BasicAgent 的八球（8-Ball）台球智能体。作为一名兼具实战经验的 AI 系统架构师与计算物理专家，我将从工程投入产出比（Return on Investment, ROI）的视角出发，摒弃学术界对“完美解”的盲目追求，转而构建一个在概率意义上鲁棒、在工程实现上高效的实战型系统。  
任务的核心难点不在于完美的几何计算，而在于对“不确定性”的控制。环境强制开启的高斯噪声（角度 $\\sigma=0.1^\\circ$，位置 $\\sigma=0.003$）彻底改变了问题的数学性质。这不再是一个确定性的几何寻优问题，而是一个**随机微分方程（SDE）约束下的概率最大化问题**。任何试图在毫秒级精度上进行确定性规划的算法，都会在现实的噪声面前崩溃。此外，对手 BasicAgent 虽然使用了贝叶斯优化（Bayesian Optimization），但其策略本质是单步贪婪（Greedy）且缺乏防守意识的。这种非对称的博弈环境为我们提供了巨大的“套利空间”——我们不需要击败完美的上帝，只需要在对手犯错时惩罚它，并迫使它进入其算法盲区。  
本报告将详细阐述一套基于**分层进化策略（Hierarchical Evolution Strategy）的技术方案。该方案拒绝了昂贵且难以调试的端到端深度强化学习（End-to-End Deep RL），选择了基于物理采样的规划算法（Sample-based Planning）**。我们将证明，在已知物理模型的前提下，利用 CMA-ES（协方差矩阵自适应进化策略）进行局部搜索，结合启发式几何规划，是当前算力约束下实现 70%\~80% 胜率的最优解。

### **1.2 物理引擎特性与计算资源的匹配策略**

pooltool 是专为科学工程设计的台球模拟器，其核心优势在于速度与灵活性 1。与基于像素的视觉输入或离散时间步长的物理引擎不同，pooltool 采用\*\*基于事件（Event-based）\*\*的演化算法 2。这意味着系统状态的更新不是通过固定的 $\\Delta t$ 积分，而是通过解析计算下一次碰撞发生的精确时间 $t\_{collision}$ 来推进。  
这种物理特性对我们的算法选型产生了深远影响：

1. **稀疏计算优势**：当台面球体较少或相互分离时，模拟速度极快。这允许我们在局势简单时进行大规模的蒙特卡洛采样（Monte Carlo Sampling）。  
2. **密集计算瓶颈**：当开球或球堆聚集时，碰撞事件呈指数级增长，模拟变慢。因此，我们的算法必须具备\*\*动态计算预算（Dynamic Compute Budget）\*\*的能力，即在复杂局面下减少搜索深度，而在关键球（如黑八）处理上投入更多算力。  
3. **算力分配**：虽然我们拥有 A800 服务器，但考虑到实时对抗的延迟要求，主力推理由 RTX 4060 承担。RTX 4060 的 CUDA 核心擅长并行处理大规模的矩阵运算，这使得并行化模拟（Batch Simulation）成为可能。A800 则主要用于离线参数微调（如调整评价函数的权重），而非实时决策。

## ---

**2\. 核心算法选型 (The ROI Choice)**

在连续动作空间 $(V\_0, \\phi, a, b)$ 中寻找最优击球策略，本质上是一个高维、非凸、含噪声的优化问题。我们需要在几秒钟内给出鲁棒的指令。本节将详细对比三种主流的基于采样的规划算法，并论证为何 CMA-ES 是本场景下的 ROI 之王。

### **2.1 算法候选池的深度技术剖析**

#### **2.1.1 蒙特卡洛树搜索 (MCTS) 及其在连续空间的局限性**

蒙特卡洛树搜索（MCTS）通过构建搜索树来平衡探索（Exploration）与利用（Exploitation），在围棋（AlphaGo）等离散动作空间游戏中取得了巨大成功 4。其核心流程包括选择（Selection）、扩展（Expansion）、模拟（Simulation）和回溯（Backpropagation）。  
然而，在台球这种连续动作空间中，MCTS 面临着严重的**维数灾难**和**离散化困境**：

1. **分支因子爆炸**：台球的击球角度 $\\phi$ 是连续变量。如果我们将其离散化，例如每 $0.1^\\circ$ 一个分支，那么仅第一层就有 3600 个节点。为了覆盖力度 $V\_0$ 和加塞参数，分支因子将达到数万。这导致搜索树极浅，几乎无法进行有效的深层规划，算法退化为随机搜索 5。  
2. **平滑性丢失**：物理世界具有局部平滑性——击球角度微调 $0.01^\\circ$，结果通常也是连续变化的（除非碰到袋角）。MCTS 的离散化切断了这种梯度信息，无法利用“差一点就进了”的反馈来指导搜索。  
3. **改进版的代价**：虽然存在 VG-UCT（Value-Gradient UCT）或 Progressive Widening 等针对连续空间的改进算法 4，但它们通常需要估计梯度或引入复杂的超参数。在工程实现上，这些算法极其复杂，调试难度大，且计算开销（ROI）并不划算。

#### **2.1.2 模型预测路径积分 (MPPI) 的适用性分析**

MPPI (Model Predictive Path Integral) 是一种基于信息论的随机控制算法，广泛应用于机器人轨迹规划和自动驾驶 7。它通过在动作序列上添加噪声，利用重要性采样（Importance Sampling）来加权合成最优控制序列。  
MPPI 的核心优势在于处理**长时间视界的序列控制**（如驾驶汽车连续转弯）。然而，台球击球是一个\*\*单脉冲控制（Single Impulse Control）\*\*问题。一旦球杆击出，球的轨迹就完全由物理定律决定，无法中途修正。在这种场景下，MPPI 的“时域滚动优化”优势无法体现，其数学形式退化为一种加权的随机采样。虽然 MPPI 对噪声鲁棒，但由于缺乏针对单步优化的进化机制，其收敛效率不如专门的进化策略 7。

#### **2.1.3 协方差矩阵自适应进化策略 (CMA-ES)**

CMA-ES 是一种基于种群的黑盒优化算法，专为非线性、非凸、连续优化问题设计 9。它维护一个多维正态分布 $\\mathcal{N}(m, C)$，其中 $m$ 是均值（当前最佳策略），$C$ 是协方差矩阵（搜索范围和方向）。  
在台球 AI 中，CMA-ES 展现出了压倒性的优势：

1. **天然适应连续空间**：不需要离散化，直接在浮点数空间进行操作。  
2. **噪声鲁棒性**：CMA-ES 优化的不是单个点，而是一个**分布**。在 $\\sigma=0.1^\\circ$ 的噪声环境下，CMA-ES 会自动收缩或扩张协方差矩阵，寻找那些“宽容度高”的解（即稍微打偏一点也能进的球），这正是我们对抗环境噪声所需的特性 10。  
3. **利用相关性**：通过更新协方差矩阵，CMA-ES 能够学习参数之间的相关性。例如，在大力击球时，切球角度产生的偏移（Cut-induced Throw）会发生变化。CMA-ES 能自动捕捉这种 $V\_0$ 和 $\\phi$ 之间的耦合关系，调整搜索方向 9。  
4. **极高的工程 ROI**：核心逻辑仅需数十行 Python 代码（或调用 cma 库），且天然支持并行化（每个个体的评估是独立的）。在 RTX 4060 上，我们可以并行评估整个种群，极大地提高了每秒的搜索效率。

### **2.2 算法决策矩阵与最终选型**

为了直观展示各算法在本任务中的适应性，我们构建如下对比矩阵：

| 评估维度 | CMA-ES (进化策略) | MCTS (蒙特卡洛树搜索) | MPPI (模型预测路径积分) |
| :---- | :---- | :---- | :---- |
| **动作空间适应性** | **极高** (原生支持连续参数优化) | **低** (需离散化，导致树浅) | **高** (原生支持连续控制) |
| **噪声鲁棒性** | **高** (基于种群统计，自动平滑噪声) 9 | **中** (需大量采样才能收敛) | **高** (基于随机扰动) |
| **计算效率 (ROI)** | **极高** (收敛快，并行度完美适配 GPU) | **低** (树结构维护开销大，深搜昂贵) | **中** (单步决策优势不明显) |
| **实现与调试难度** | **低** (代码量小，只有几个超参数) | **高** (需处理剪枝、回溯、离散化) | **中** (需调优温度参数 $\\lambda$) |
| **物理梯度利用** | **隐式利用** (通过协方差矩阵捕捉局部地形) | **无** (离散化切断了梯度信息) | **隐式利用** (通过轨迹加权) |

最终选型结论：  
为了实现“高胜率、高 ROI”，我们坚决抛弃纯 MCTS，确立以 分层启发式搜索 \+ 局部 CMA-ES (Hierarchical Heuristic Search \+ Local CMA-ES) 为核心的架构。  
**架构逻辑**：

1. **L1 全局粗筛 (Global Heuristics)**：利用几何规则（Ghost Ball, Bank Systems）快速生成 $N$ 个高潜力的候选动作“种子”（Seeds）。这一步利用先验知识，避免了在全空间盲目搜索。  
2. **L2 局部精调 (Local Optimization)**：针对每一个高分种子，启动一个微型的 CMA-ES 优化器。在局部范围内（如角度 $\\pm 2^\\circ$）进行高斯采样和迭代，利用 pooltool 的物理反馈，在噪声干扰下寻找成功率最高且母球落点最佳的参数组合。

## ---

**3\. 针对性策略设计 (Exploit the Opponent)**

仅仅能把球打进是不够的。要达到 80% 的胜率，我们需要在策略层面碾压 BasicAgent。对手的弱点是我们的最大杠杆：它贪婪（只看进球率）、短视（不看下一杆）、鲁莽（无防守）。我们的策略设计将围绕**风险管理**和**局面控制**展开。

### **3.1 密集奖励函数的设计 (Dense Reward Engineering)**

在强化学习和规划中，稀疏奖励（只在赢球时给分）会导致搜索效率极低 11。为了引导 CMA-ES 快速收敛，我们需要设计一个包含物理洞察的密集奖励函数（Dense Reward Function）13。  
我们将单次击球动作 $a$ 的价值函数 $J(a)$ 定义为：

$$J(a) \= P\_{pot}(a) \\cdot \+ (1 \- P\_{pot}(a)) \\cdot \- C\_{foul}$$  
其中各项的物理与战术含义如下：

* $P\_{pot}(a)$ (进球概率)：这是核心过滤指标。在噪声 $\\sigma$ 下，我们对动作 $a$ 进行 $K=20$ 次并行模拟。

  $$P\_{pot}(a) \= \\frac{1}{K} \\sum\_{k=1}^{K} \\mathbb{I}(\\text{Ball Pocketed in Simulation } k)$$

  工程洞察：如果 $P\_{pot} \< 0.6$，意味着该球在统计上是不稳定的。对于 BasicAgent 这种对手，强行进攻低概率球等于送出球权。  
* **$R\_{pot}$ (基础进球奖励)**：固定的大分值（如 100 分），确保算法优先考虑进球。  
* **$V\_{next}(S')$ (白球控制价值)**：这是击败贪婪对手的关键。BasicAgent 往往打进一球后白球贴库或被挡，导致连杆中断。我们将引入“有效区域”（Effective Zone）模型。  
  * **计算逻辑**：假设当前球进袋，计算下一颗最容易进攻的目标球 $T\_{next}$ 的最佳击球位（通常在 Ghost Ball 连线的反向延伸线上，距离约 30-50cm）。  
  * **评分公式**：$V\_{next} \\propto e^{-\\frac{d^2}{2\\sigma\_{pos}^2}}$，其中 $d$ 是模拟后的白球位置与理想位置的欧氏距离。  
* **$R\_{safe}(S')$ (防守价值)**：当无法进球时，我们需要最大化防守价值。  
  * **距离压制**：$R\_{safe} \\propto \\text{Distance}(\\text{CueBall}, \\text{OpponentBalls})$。BasicAgent 的准确率随距离衰减极快。  
  * **掩护（Snooker）**：检测白球与对手目标球之间是否存在障碍球。每形成一条被阻挡的视线（Line of Sight），奖励大幅增加。  
* **$C\_{foul}$ (犯规惩罚)**：巨大的负分（如 \-500）。任何可能导致白球洗袋或未触碰目标球的动作都必须被严厉禁止。

### **3.2 针对 BasicAgent 的防守反击策略 (Safety Play)**

BasicAgent 的致命弱点是“被迫进攻”。当面对一个进球率极低（如 5%）的局面时，它依然会尝试大力进攻，结果往往是把球堆炸散，给对手留下绝佳机会。  
我们的**主动防守逻辑**如下：

1. **阈值触发**：当所有候选进攻动作的最高 $P\_{pot} \< 30\\%$ 时，强制进入防守模式。  
2. **安全球生成 (Lag Shot)**：  
   * **目标**：将白球轻轻推到台面的死角（如底库库边）或球堆的背面。  
   * **实现**：搜索 $V\_0 \\in \[0.5, 1.5\] m/s$ 的低速动作，优化目标是 $Maximize(\\text{Difficulty for Opponent})$。  
3. **斯诺克陷阱**：利用 BasicAgent 解球能力弱（只会在全空间随机搜索）的特点 1，我们刻意制造斯诺克。一旦对手解球失败（犯规），我们将获得**自由球（Ball-in-Hand）**。这是胜率提升的阶跃点。拿到自由球后，我们的胜率将从 50% 飙升至 90% 以上。

### **3.3 白球控制的数学建模 (Cue Ball Control)**

为了实现细腻的白球控制，我们必须利用 pooltool 的摩擦模型。白球在碰撞后的分离角（90度规则 vs 30度规则）取决于它是滑动（Sliding）还是滚动（Rolling）状态 15。

* **滑动状态**：分离角接近 90度。  
* **滚动状态**：分离角接近 30度（自然跑位）。  
* **控制变量**：击球点的高低（$b$ 参数）。打高杆（Top spin）让球更快进入滚动，分离角变小（前冲）；打低杆（Back spin）保持滑动甚至倒旋，分离角变大（拉杆）。

我们的 CMA-ES 优化向量不仅包含 $V\_0, \\phi$，还必须包含自旋参数 $a, b$。

$$\\text{Optimize } \\vec{x} \= \[V\_0, \\phi, a, b\]$$

目标是让白球停在 $V\_{next}$ 定义的高分区域。通过这种方式，我们不仅是在打球，而是在布局。

## ---

**4\. 鲁棒性工程 (Noise Handling)**

在 $\\sigma=0.1^\\circ$ 的噪声环境下，单次模拟的结果是不可信的。为了保证工程上的鲁棒性，我们需要建立一套**统计护盾**。

### **4.1 蒙特卡洛积分评估 (Monte Carlo Integration)**

我们不能依赖确定性的物理计算。我们需要计算的是期望效用（Expected Utility）。  
对于任意一个候选动作 $\\mathbf{u}$，其真实价值 $J(\\mathbf{u})$ 是在噪声分布上的积分：

$$J(\\mathbf{u}) \= \\int\_{\\Omega} R(\\text{Simulate}(\\mathbf{u} \+ \\xi)) \\cdot p(\\xi) \\, d\\xi$$

其中 $\\xi$ 是环境噪声向量。  
工程实现伪代码：  
我们将利用 multiprocessing 模块进行并行评估。

Python

import numpy as np  
import pooltool as pt  
from multiprocessing import Pool

def evaluate\_action\_robustness(system\_state, action, n\_samples=20):  
    """  
    对单一动作进行鲁棒性评估  
    Input:  
        system\_state: 当前台球系统快照 (深拷贝)  
        action: {V0, phi, a, b}  
        n\_samples: 采样次数 (ROI平衡点: 20次)  
    Output:  
        score: 鲁棒性评分 (Lower Confidence Bound)  
    """  
    results \=  
      
    \# 构建 batch 任务  
    tasks \=  
    for \_ in range(n\_samples):  
        \# 1\. 注入环境噪声  
        \# 注意：噪声是在执行层面发生的，我们需要模拟这种微扰  
        noisy\_phi \= action.phi \+ np.random.normal(0, 0.1 \* np.pi / 180\) \# 0.1度标准差  
        noisy\_pos \= action.cue\_pos \+ np.random.normal(0, 0.003, 2\) \# 位置噪声  
          
        \# 2\. 构建模拟任务  
        task \= (system\_state, noisy\_phi, action.V0, action.a, action.b, noisy\_pos)  
        tasks.append(task)  
          
    \# 3\. 并行执行模拟 (利用 pooltool 的轻量级特性)  
    \# 假设有一个 worker 函数 run\_simulation 处理物理演化  
    outcomes \= parallel\_executor.map(run\_simulation, tasks)  
      
    \# 4\. 统计分析  
    success\_count \= sum(\[1 for r in outcomes if r\['pocketed'\]\])  
    p\_success \= success\_count / n\_samples  
      
    avg\_next\_pos\_score \= np.mean(\[r\['pos\_score'\] for r in outcomes if r\['pocketed'\]\])  
    risk\_factor \= np.std(\[r\['total\_reward'\] for r in outcomes\])  
      
    \# 5\. 评分策略：LCB (Lower Confidence Bound)  
    \# 我们不仅看平均分，还看风险。风险越大，扣分越多。  
    final\_score \= (p\_success \* 100\) \+ avg\_next\_pos\_score \- (1.5 \* risk\_factor)  
      
    return final\_score

### **4.2 动态采样率与算力管理**

为了最大化 ROI，我们不能对所有动作一视同仁。

* **第一阶段（粗筛）**：生成 20-30 个几何候选动作。每个动作仅采样 $N=2$ 次。快速剔除那些“完全打不进”或“必然洗袋”的动作。  
* **第二阶段（精选）**：对保留的 Top-5 动作，启动 CMA-ES。CMA-ES 内部每代种群大小为 8，迭代 5-10 次。这意味着对每个候选动作投入约 40-80 次模拟。  
* **第三阶段（决胜）**：如果是黑八关键球，强制增加采样至 $N=100$，确保万无一失。

这种**金字塔式的算力分配**确保了我们在毫秒级时间内处理简单决策，而在关键决策上展现出超越人类的精确度。

## ---

**5\. 实施路线图 (Roadmap)**

本项目的开发周期紧张，我们将采用**敏捷迭代**的方式，确保在 Day 2 就能产出可用的、胜率提升的版本，并在 Day 7 达到最终目标。

### **Day 1-2: 快速原型与止损 (The Defender)**

目标：不实现复杂的 CMA-ES，仅通过规则修补和蒙特卡洛过滤提升 10-20% 胜率。  
任务清单：

1. **环境搭建与剖析**：  
   * 配置 RTX 4060 开发环境，安装 pooltool 并跑通 Hello World 16。  
   * 阅读 BasicAgent 源码，找到其决策函数 pick\_shot。  
2. **实现“统计过滤器”**：  
   * 在 BasicAgent 选出最佳动作后，不要立即执行。  
   * 将该动作放入 evaluate\_action\_robustness 函数（见 4.1 节），在噪声下模拟 10 次。  
   * **止损逻辑**：如果 10 次模拟中进球次数 \< 5 次（胜率 \< 50%），**强制否决**该动作。  
3. **实现“随机推杆防守”**：  
   * 当进攻动作被否决后，执行一个预设的防守策略：随机选择一个远离所有球的方向，以 $V\_0 \= 1.0 m/s$ 轻推。  
   * **原理**：与其冒着 60% 的风险送球权，不如交出一杆无风险的防守，让对手去处理烂摊子。

**预期成果**：Agent 不再因为贪婪而送出简单球权，胜率显著提升，尤其是在长台进攻的稳定性上。

### **Day 3-5: 核心攻击引擎实现 (The Sniper)**

目标：实现 L1 启发式生成器 \+ L2 CMA-ES 优化器，接管进攻决策。  
任务清单：

1. **几何生成器开发**：  
   * 编写 get\_ghost\_ball\_params(target\_ball, pocket) 函数。利用向量几何计算无摩擦环境下的理想撞击点 19。  
   * 实现 get\_bank\_shot\_candidates，利用镜像法计算简单的库边反弹路线。  
2. **CMA-ES 集成**：  
   * 引入 cma Python 库。  
   * 对接 pooltool 的 simulate 接口作为 Fitness Function。  
   * 定义优化变量范围：$V\_0 \\in \[0.5, 4.0\], \\phi \\in \[\\phi\_{aim}-2^\\circ, \\phi\_{aim}+2^\\circ\]$。  
3. **并行化加速**：  
   * 利用 multiprocessing 模块，将 CMA-ES 的种群评估并行分发到 CPU 核心上（pooltool 的模拟主要是 CPU 密集的，除非使用了 GPU 加速版）。

**预期成果**：Agent 具备了“校准”能力。对于稍微有偏差的几何计算，CMA-ES 能自动修正角度及力度，找到进球的“甜点”。

### **Day 6-7: 走位与决胜版本 (The Strategist)**

目标：加入白球控制和斯诺克解球，达到 80% 胜率。  
任务清单：

1. **走位模块**：  
   * 实现 3.3 节的白球控制模型。在 CMA-ES 的目标函数中加入 $V\_{next}$ 项。  
   * 调试 $R\_{pot}$ 与 $V\_{next}$ 的权重比，通常建议 $R\_{pot} : V\_{next} \\approx 7 : 3$，确保进球第一。  
2. **解斯诺克模块 (Kick Shot)**：  
   * 当无法直接看到目标球时，启动“随机反射采样”。  
   * 向库边随机发射 50 条射线，筛选出能反弹碰到目标球的路径。  
   * 选择力度最小的一条（避免白球乱跑摔袋）。  
3. **A800 参数微调**：  
   * 利用 A800 服务器，让最终版 Agent 与 Day 1 版本或 BasicAgent 进行 1000 局快速对战。  
   * 使用贝叶斯优化（Bayesian Optimization）调整高层超参数（如防守阈值、采样次数、评价权重），榨干最后 5% 的胜率。

## ---

**6\. 技术细节与数据结构**

为了确保方案的可落地性，以下提供核心数据结构的设计。

### **6.1 状态与动作空间定义**

Python

class ShotParameters:  
    def \_\_init\_\_(self):  
        self.V0 \= 0.0      \# 击球速度 (m/s)  
        self.phi \= 0.0     \# 击球方位角 (radians)  
        self.theta \= 0.0   \# 击球仰角 (通常为0)  
        self.a \= 0.0       \# 水平击球点偏移 (spin)  
        self.b \= 0.0       \# 垂直击球点偏移 (spin)  
        self.cue\_id \= 'cue'

class EvaluationResult:  
    def \_\_init\_\_(self):  
        self.success\_rate \= 0.0  
        self.expected\_reward \= 0.0  
        self.safety\_score \= 0.0  
        self.position\_score \= 0.0  
        self.is\_foul \= False

### **6.2 流程图描述 (Flowchart Logic)**

1. **Start**: 接收当前 System State。  
2. **Check Phase**:  
   * 是否被斯诺克？ \-\> Yes: Goto **Kick Shot Generator**.  
   * 是否自由球？ \-\> Yes: Goto **Ball-in-Hand Optimizer** (全台扫描最佳球).  
   * No: Goto **Standard Planner**.  
3. **Standard Planner**:  
   * **Generate**: 遍历所有合法目标球，生成几何种子动作 $S \= \\{s\_1, s\_2,..., s\_n\\}$。  
   * **Filter**: 快速模拟（N=1），剔除无效动作。  
   * **Optimize**: 对 Top-k 种子，运行 **Local CMA-ES**。  
     * Loop (Generations):  
       * Sample population based on $\\mathcal{N}(m, C)$.  
       * Evaluate (Parallel Batch Simulation with Noise).  
       * Update $m, C$.  
   * **Select**: 选择 Score 最高的动作 $s^\*$。  
4. **Safety Check**:  
   * 如果 Score($s^\*$) \< Threshold (0.6):  
     * **Generate Safety**: 生成防守动作（轻推、藏球）。  
     * Return Safety Action.  
   * Else:  
     * Return Attack Action $s^\*$.  
5. **Execute**: 输出指令给底层控制器。

## ---

**7\. 结论**

本报告提出的 **“概率规划+进化策略”** 架构，是针对当前算力约束（RTX 4060）和环境约束（高斯噪声）的最优工程解。它避免了 MCTS 在连续空间的低效和 Deep RL 的训练黑盒问题，利用 CMA-ES 的统计特性将噪声转化为优化的动力。  
通过“防守反击”的博弈策略，我们最大化了对手 BasicAgent 的弱点利用率。这不仅是一个台球 AI，更是一个展示如何在不确定性环境下进行**高 ROI 决策**的工程范例。按照本路线图执行，我们有充分的信心在 7 天内构建出一个胜率稳定在 70%-80% 的高水平智能体。

#### **引用的著作**

1. ekiefl/pooltool: A sandbox billiards game that emphasizes realistic physics \- GitHub, 访问时间为 十二月 6, 2025， [https://github.com/ekiefl/pooltool](https://github.com/ekiefl/pooltool)  
2. Realistic pool simulator \- Evan Kiefl, 访问时间为 十二月 6, 2025， [https://ekiefl.github.io/projects/pooltool/](https://ekiefl.github.io/projects/pooltool/)  
3. The physics of pool/billiards \- Evan Kiefl, 访问时间为 十二月 6, 2025， [https://ekiefl.github.io/2020/04/24/pooltool-theory/](https://ekiefl.github.io/2020/04/24/pooltool-theory/)  
4. Monte-Carlo Tree Search in Continuous Action Spaces with Value Gradients, 访问时间为 十二月 6, 2025， [https://ojs.aaai.org/index.php/AAAI/article/view/5885/5741](https://ojs.aaai.org/index.php/AAAI/article/view/5885/5741)  
5. Extensions of Monte Carlo Tree Search into Continuous Action Spaces \- Aaltodoc, 访问时间为 十二月 6, 2025， [https://aaltodoc.aalto.fi/bitstreams/93d6bda1-8997-4700-8360-7c8ad079277a/download](https://aaltodoc.aalto.fi/bitstreams/93d6bda1-8997-4700-8360-7c8ad079277a/download)  
6. Monte Carlo Tree Search in Continuous Action Spaces with Execution Uncertainty \- IJCAI, 访问时间为 十二月 6, 2025， [https://www.ijcai.org/Proceedings/16/Papers/104.pdf](https://www.ijcai.org/Proceedings/16/Papers/104.pdf)  
7. Sample-efficient Cross-Entropy Method for Real-time Planning \- arXiv, 访问时间为 十二月 6, 2025， [https://arxiv.org/pdf/2008.06389](https://arxiv.org/pdf/2008.06389)  
8. Adaptive Model Predictive Control by Learning Classifiers, 访问时间为 十二月 6, 2025， [https://proceedings.mlr.press/v168/guzman22a/guzman22a.pdf](https://proceedings.mlr.press/v168/guzman22a/guzman22a.pdf)  
9. The CMA Evolution Strategy: A Comparing Review \- ResearchGate, 访问时间为 十二月 6, 2025， [https://www.researchgate.net/publication/227050324\_The\_CMA\_Evolution\_Strategy\_A\_Comparing\_Review](https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review)  
10. Real-Time Policy Optimization for UAV Swarms Based on Evolution Strategies \- MDPI, 访问时间为 十二月 6, 2025， [https://www.mdpi.com/2504-446X/8/11/619](https://www.mdpi.com/2504-446X/8/11/619)  
11. Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL, 访问时间为 十二月 6, 2025， [https://proceedings.neurips.cc/paper/2021/file/1454ca2270599546dfcd2a3700e4d2f1-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/1454ca2270599546dfcd2a3700e4d2f1-Paper.pdf)  
12. Enhancing Online Reinforcement Learning with Meta-Learned Objective from Offline Data, 访问时间为 十二月 6, 2025， [https://ojs.aaai.org/index.php/AAAI/article/view/33784/35939](https://ojs.aaai.org/index.php/AAAI/article/view/33784/35939)  
13. Augmented Memory: Sample-Efficient Generative Molecular Design with Reinforcement Learning | JACS Au \- ACS Publications, 访问时间为 十二月 6, 2025， [https://pubs.acs.org/doi/10.1021/jacsau.4c00066](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)  
14. Beyond Imitation: Recovering Dense Rewards from Demonstrations \- arXiv, 访问时间为 十二月 6, 2025， [https://arxiv.org/html/2510.02493v1](https://arxiv.org/html/2510.02493v1)  
15. The 30 Degree Rule \- pooltool documentation, 访问时间为 十二月 6, 2025， [https://pooltool.readthedocs.io/en/latest/examples/30\_degree\_rule.html](https://pooltool.readthedocs.io/en/latest/examples/30_degree_rule.html)  
16. Installation \- pooltool documentation, 访问时间为 十二月 6, 2025， [https://pooltool.readthedocs.io/en/latest/getting\_started/install.html](https://pooltool.readthedocs.io/en/latest/getting_started/install.html)  
17. Creating a billiards simulator: the transition from equations to code \- Evan Kiefl, 访问时间为 十二月 6, 2025， [https://ekiefl.github.io/2021/03/25/pooltool-start/](https://ekiefl.github.io/2021/03/25/pooltool-start/)  
18. Hello World \- pooltool documentation, 访问时间为 十二月 6, 2025， [https://pooltool.readthedocs.io/en/v0.4.0/getting\_started/script.html](https://pooltool.readthedocs.io/en/v0.4.0/getting_started/script.html)  
19. Ghost-Ball Aiming \- Dr. Dave Pool Info, 访问时间为 十二月 6, 2025， [https://drdavepoolinfo.com/faq/aiming/ghost-ball/](https://drdavepoolinfo.com/faq/aiming/ghost-ball/)  
20. What formula could I apply to a virtual billiards game for calculating the angle of incidence on a kick shot? : r/AskPhysics \- Reddit, 访问时间为 十二月 6, 2025， [https://www.reddit.com/r/AskPhysics/comments/16qhoa7/what\_formula\_could\_i\_apply\_to\_a\_virtual\_billiards/](https://www.reddit.com/r/AskPhysics/comments/16qhoa7/what_formula_could_i_apply_to_a_virtual_billiards/)