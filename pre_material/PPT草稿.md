# PPT 设计草稿：Time-Constrained 8-Ball Pool AI

**总页数：** 12页
**风格建议：** 简洁、科技感（深蓝色或极简白底）、强调图表可视化

---

## Slide 1: 标题页
*   **[排版]：** 居中对齐，背景可使用一张虚化的台球物理仿真图。
*   **[主标题]：** Geometric-Guided Evolution Strategy for Time-Constrained 8-Ball Pool AI
*   **[副标题]：** 基于几何引导与进化策略的时间受限8球台球智能体
*   **[汇报人]：** 林纪帆 (523030910057) / 杨逸凡 (523030910054)
*   **[机构]：** Shanghai Jiao Tong University / AI3603 Course Project

---

## Slide 2: 背景与挑战 (Introduction & Challenges)
*   **[排版]：** 左文右图。
*   **[左侧文字 - 核心矛盾]：**
    *   **高维连续动作空间 (High-Dimensional Continuous Space)**:
        *   $A = (V_0, \phi, \theta, a, b) \in \mathbb{R}^5$
        *   无法使用离散搜索 (No exhaustive search)。
    *   **严格的时间限制 (Strict Time Budget)**:
        *   ~180s / Game (Average)。
        *   物理仿真昂贵 (Expensive Physics Rollouts)。
    *   **环境噪声与稀疏奖励**:
        *   高斯噪声干扰 ($\sigma_{\phi}, \sigma_{spin}$)。
        *   Pot-or-Fail (进球或失败) 的二元结果。
*   **[右侧图片]：** 
    *   展示台球桌的复杂局势图，标注出连续动作空间的5个自由度示意图。

---

## Slide 3: 问题建模 (Problem Formulation)
*   **[排版]：** 上下结构。
*   **[上方 - 目标函数]：**
    *   **Objective:** Maximize Win Rate under Time Constraints.
    *   **Loss Condition:** Catastrophic Fouls (e.g., Illegal 8-ball pot).
*   **[下方 - 奖励函数]：**
    *   $R = R_{pot} + R_{foul} + R_{strategic}$
    *   **Pot:** 进球得分 (+50/-20)。
    *   **Foul:** 犯规惩罚 (-100/-150)。
    *   **Strategy:** $f_{pos}(s')$ (走位) + $f_{def}(s')$ (防守)。

---

## Slide 4: 系统总览 (System Pipeline)
*   **[排版]：** 全屏流程图（核心页）。
*   **[内容]：** 绘制横向或漏斗状流程图：
    1.  **Stage 1:** Ghost Ball Candidate Generation (450 -> 8 candidates).
    2.  **Stage 2:** Fast Filtering (1 rollout).
    3.  **Stage 3:** Refined Evaluation (Catastrophic Filter).
    4.  **Stage 4:** **Selective CMA-ES** (Trigger if score < 60).
    5.  **Stage 5:** Strategic/Safety Planning.
    6.  **Stage 6:** Final Verification.
*   **[视觉强调]：** 用高亮颜色标出 "Geometric Pruning" 和 "CMA-ES" 模块。

---

## Slide 5: 核心方法 I - 几何引导剪枝 (Geometric Pruning)
*   **[排版]：** 左图右文。
*   **[左侧图片]：** 幽灵球 (Ghost Ball) 原理示意图。
    *   显示目标球(T)、袋口(P)、幽灵点(G)与母球(Cue)的连线几何关系。
*   **[右侧文字]：**
    *   **Ghost Ball Heuristic:** 利用几何关系计算理想击球点。
    *   **Pruning Strategy (剪枝策略):**
        *   只考虑Top-3近的目标球。
        *   每个球只考虑Top-2近的袋口。
        *   路径可行性检测 (Path Feasibility Check)。
    *   **Result:** 候选数量从 ~450 降至 ~8，计算量降低 50x。

---

## Slide 6: 核心方法 II - 进化策略微调 (CMA-ES Refinement)
*   **[排版]：** 左右对比图。
*   **[文字内容]：**
    *   **Why CMA-ES?** 几何方法无法处理旋转(Spin)和物理非线性。
    *   **Time-Aware Adaptation:**
        *   仅针对Top-1候选动作。
        *   仅在当前得分不佳时触发 (Selective Trigger)。
        *   小种群 ($\lambda=4$)，少迭代 ($g=2$)。
*   **[图片示意]：** 
    *   展示一个使用Ghost Ball会打偏（受噪声影响），但经过CMA-ES微调参数后成功进球并走位的轨迹对比。

---

## Slide 7: 战略规划与Lambda Ladder (Strategic Planning)
*   **[排版]：** 列表与公式结合。
*   **[内容]：**
    *   **Positional Bonus (走位):** 奖励母球停留在下一目标球的可进攻区域。
    *   **Defensive Logic (防守):** 
        *   当进攻期望低时，惩罚母球留给对手的机会。
        *   $R_{def} \propto - \max(\text{Opponent Pot Probability})$.
    *   **Lambda Ladder (自适应权重):**
        *   根据剩余球数调整激进程度。
        *   剩余球少 -> 权重高 (更加谨慎)。

---

## Slide 8: 风险控制与致命犯规 (Risk Control)
*   **[排版]：** 醒目的警告图标风格。
*   **[核心机制]：**
    *   **Catastrophic Fouls (致命犯规):**
        *   8-Ball in wrong pocket (误进黑8) = **Instant Loss**.
        *   Cue ball scratch with 8-ball = **Instant Loss**.
    *   **Hard Filtering:**
        *   只要在采样中出现一次致命犯规，该动作得分直接置为 $-\infty$。
    *   **Adaptive Verification:**
        *   当黑8在袋口附近时，增加采样次数 (4 -> 6 samples) 以降低方差。

---

## Slide 9: 实验设置 (Experimental Setup)
*   **[排版]：** 两列布局。
*   **[环境]：**
    *   **Simulator:** Pooltool (High-fidelity physics).
    *   **Hardware:** Apple Mac mini (M4).
*   **[对手 (Baselines)]：**
    *   **BasicAgent:** 基础贝叶斯优化 Agent。
    *   **BasicAgentPro:** 带蒙特卡洛树搜索 (MCTS) 的增强版 Agent。
*   **[赛制]：**
    *   120场锦标赛 (Rotation Protocol)。
    *   记录胜率 (Win Rate) 与 决策时间 (Wall-clock Time)。

---

## Slide 10: 实验结果 (Main Results)
*   **[排版]：** 大字号数据强调 + 柱状图。
*   **[关键数据]：**
    *   **Win Rate vs BasicAgent:** **> 90%** (碾压优势).
    *   **Win Rate vs BasicAgentPro:** **显著优势** (具体参考 Win Rate Table，如 75% or 93%).
*   **[时间性能]：**
    *   Avg. Time per Game: **~120-130s** (远低于180s预算).
    *   Over-budget Rate: < 5%.
*   **[结论]：** 在保证速度的前提下，显著提升了竞技水平。

---

## Slide 11: 消融实验 (Ablation Study)
*   **[排版]：** 表格形式展示。
*   **[表格内容]：**
    | Variant | Win Rate | Avg Time (s) | Impact |
    | :--- | :---: | :---: | :--- |
    | **Full System** | **High** | **19.4** | Best Trade-off |
    | w/o Geometric Pruning | High | **52.9** | **Too Slow (Timeout)** |
    | w/o CMA-ES | Lower | 24.6 | 精度下降 |
    | w/o Strategy | Lowest | 29.4 | 易输掉残局 |
*   **[结论]：** 几何剪枝是速度的保障，CMA-ES与策略模块是胜率的保障。

---

## Slide 12: 总结 (Conclusion)
*   **[排版]：** 简洁的总结点。
*   **[Takeaway Messages]：**
    1.  **Geometry as Prior:** 几何知识能极大压缩搜索空间 (450 -> 8)。
    2.  **Selective Refinement:** 好钢用在刀刃上 (CMA-ES on demand)。
    3.  **Risk Awareness:** 在随机环境中，"不输" (防致命犯规) 比 "赢" 更重要。
*   **[Future Work]：** 引入深度强化学习 (Deep RL) 替代手工奖励函数。
*   **[底部]：** Thank You! Q&A.

---