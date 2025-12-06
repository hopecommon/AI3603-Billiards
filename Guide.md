## 1. 基于实际代码的任务与环境理解

### 1.1 评估流程 & 胜率

从 `evaluate.py` 和 `GAME_RULES.md` 可以确定：

* 环境：`env = PoolEnv()`
* 智能体：

  * `agent_a = BasicAgent()`
  * `agent_b = NewAgent()`（你要实现）
* 对战：

  * `n_games = 40`（助教评分也是用 40 局）
  * 每 4 局一循环：

    * 先后手（Player A/B）轮换
    * 球型（solid/stripe）轮换，保证四种组合公平
* 每局完成后，`PoolEnv.get_done()` 返回：

  * `done: bool`
  * `info = {'winner': 'A'/'B'/'SAME', 'hit_count': ...}`
* 统计：

  * `results['AGENT_A_WIN'] / ['AGENT_B_WIN'] / ['SAME']`
  * 分数：赢 1 分，平各 0.5 分
  * 胜率 = 分数 / 40（和你之前说的一致）

### 1.2 环境 API & 观测/动作结构

`poolenv.py` 给出的核心接口（且在最终评测中不可改动）：

* `env.reset(state=None, target_ball=None)`

  * `target_ball` 控制 Player A 的球型（`'solid'` 或 `'stripe'`），从而决定 A/B 的目标球集合。
  * 初始化桌子、球、玩家目标球、击球计数等。
* `env.get_observation(player=None)`

  * 若不传 `player`，返回当前出杆方的观测。
  * 返回三元组 `(balls, my_targets, table)`：

    * `balls: dict[str, Ball]`

      * key: `'cue'`, `'1'`…`'7'`, `'8'`, `'9'`…`'15'`
      * `ball.state.rvw[0]`：位置 (x,y,z)
      * `ball.state.rvw[1]`：速度
      * `ball.state.s`：状态码（0 静止、4 已进袋等）
    * `my_targets: list[str]`

      * 正常：7 个目标球 id
      * 自己球打光后：`['8']`（表明应该打黑 8）
    * `table: Table`

      * 尺寸 `table.w`, `table.l`
      * 袋口 `table.pockets`（`'lb','lc','lt','rb','rc','rt'` 等）
* `env.take_shot(action: dict)`

  * `action = {'V0', 'phi', 'theta', 'a', 'b'}`：

    * `V0 ∈ [0.5, 8.0]` 初速度
    * `phi ∈ [0, 360]` 水平角度（度）
    * `theta ∈ [0, 90]` 垂直角度（度）
    * `a, b ∈ [-0.5, 0.5]` 母球表面的击点偏移（球半径比例）
  * 若 `self.enable_noise=True`，环境会对 action 每个维度加高斯噪声（标准差见下）；最终动作用于真实物理仿真。
  * 返回一个 dict：

    * 必有字段：

      * `ME_INTO_POCKET: list[str]`
      * `ENEMY_INTO_POCKET: list[str]`
      * `WHITE_BALL_INTO_POCKET: bool`
      * `BLACK_BALL_INTO_POCKET: bool`
      * `BALLS: dict`（击球后的所有球状态）
    * 条件字段：

      * `FOUL_FIRST_HIT`：首球犯规
      * `NO_POCKET_NO_RAIL`：没进球且没碰库
      * `NO_HIT`：白球未接触任何球
* `env.get_done()`

  * 若游戏结束：`(True, {'winner': 'A'/'B'/'SAME', 'hit_count': int})`
  * 否则 `(False, {})`

> 注意：环境内部维护了 `self.balls`, `self.table`, `self.cue`，以及一个 `shot_record` 用于渲染回放；还提供了 `save_balls_state / restore_balls_state` 来深拷贝/恢复 balls。

### 1.3 噪声与犯规 / 胜负判定

`GAME_RULES.md` 详细给出了规则和噪声系数：

* 环境噪声（不可改）：

  * `V0`: σ = 0.1
  * `phi`: σ = 0.1°
  * `theta`: σ = 0.1°
  * `a,b`: σ = 0.003
* `PoolEnv` 内部：

  * 60 杆上限：`hit_count >= MAX_HIT_COUNT` 时：

    * 比较 A/B 剩余目标球数（不含 8 号球）
    * 少的一方获胜；相同则 `winner='SAME'`。
* 犯规：

  * 直接判负：

    * 黑 8 + 白球同时进袋
    * 己方目标未清空就打进黑 8
  * 交换球权（但继续本局，状态保持或恢复）：

    * 白球进袋（恢复上一杆状态）
    * 未击中任何球
    * 首球击中对方球或黑 8（且不合法）
    * 无进球且没碰库

### 1.4 BasicAgent 的真实样子（很重要）

`agent.py` 中，BasicAgent 已经是一个**“物理仿真 + 贝叶斯优化 + 手写 reward”** 的强基线：

* 结构：

  * `class BasicAgent(Agent)`，接口：

    ```python
    def decision(self, balls=None, my_targets=None, table=None) -> dict
    ```
  * `analyze_shot_for_reward(shot, last_state, player_targets)`：单杆结果评分函数。
* 搜索：

  * 搜索空间 `pbounds`：

    * `V0 ∈ [0.5,8.0]`, `phi ∈ [0,360]`, `theta ∈ [0,90]`, `a,b ∈ [-0.5,0.5]`
  * 使用 `bayes_opt.BayesianOptimization` + 自己构造的高斯过程（`GaussianProcessRegressor(Matern)`）：

    * 初始随机搜索 `self.INITIAL_SEARCH = 20`
    * 后续 `self.OPT_SEARCH = 10` 次 BO 迭代
    * `alpha = 1e-2`
  * reward 函数中：

    * 用当前观测 `balls, table` 深拷贝出一个 “沙盒世界”（独立于 `PoolEnv`）：

      ```python
      sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
      sim_table = copy.deepcopy(table)
      cue = pt.Cue(cue_ball_id="cue")
      shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
      pt.simulate(shot)
      ```
    * 用 `analyze_shot_for_reward(shot, last_state_snapshot, my_targets)` 计算 reward。
* `analyze_shot_for_reward` 的核心逻辑（单杆）：

  * 强烈鼓励：

    * 每进一个己方球 `+50`
    * 合法打进黑 8 `+100`
    * “什么都没发生但没犯规” 给 `+10`（保守安全）
  * 惩罚：

    * 白球+黑 8 同时进袋：`-150`
    * 单独白球进袋：`-100`
    * 非法打黑 8：`-150`
    * 首球犯规：`-30`
    * 无进球且没碰库：`-30`
    * 打进对手球：每颗 `-20`
* BasicAgent 自己还有一套**独立噪声**（`self.noise_std`）可以在 reward evaluation 里打开，但在 `__init__` 中 `self.enable_noise = False`，即默认**不考虑噪声的鲁棒性**。

**结论：** 你面对的是一个：

> “对当前一杆，做 30 次 BO 采样搜索的单步规划器，reward 只看这一杆是否进球/犯规，几乎不看长远局面，也不考虑环境噪声。”

这非常符合我们之前猜的“物理仿真 + 启发式搜索”，只不过实现得更高级（BayesOpt + GP）。

这也给我们重要启发：

> **要超过它，就要在“鲁棒性 + 长远局面 + 搜索结构”这三点上做文章。**

---

## 2. 在真实代码基础上的候选方法族分析

我重新整理 3 条你真正值得投入时间的路线（都可以混合），并明确它们相对 BasicAgent 的差异。

### 2.1 路线 1：BasicAgent++ —— 改 reward + 加噪声 + 更聪明搜索

**核心思想：**
在 `NewAgent` 里基本照抄 BasicAgent 的结构（使用同样的 pooltool 仿真），但：

1. **更好的单杆 reward**（“看未来”的启发式）：

   * 在 `analyze_shot_for_reward` 的基础上，加入：

     * 母球落点是否：

       * 靠近自己下一颗潜在可进球（距离越短评分越高）。
       * “藏”在一堆球后面，让 BasicAgent 难以进攻（可以粗略估计对手可直线进攻球数）。
     * 打开/解开自己球 cluster 的奖励，避免自己球堆死。
     * 避免把对手球 cluster “解开”的惩罚（如果本杆进球不多但大幅减少对手球 obstruction）。
2. **鲁棒性：在 reward 评估时加“环境级噪声”**

   * BasicAgent 的 reward_eval 是**确定性的**模拟，不含 env 噪声。
   * 我们可以在 `NewAgent` 的 reward 函数中：

     * 对 `(V0,phi,theta,a,b)` 做 K 次重复模拟：

       * 每次在这些参数上加高斯噪声，标准差与环境一致（或略大）：

         * `σ_V0=0.1, σ_phi=0.1°, σ_theta=0.1°, σ_a=σ_b=0.003`。
       * 对每次 shot 调用 `analyze_shot_for_reward`，然后取平均。
     * 这样 reward 近似的是“考虑噪声后的期望收益”，天然偏好**宽容度高的线路**。
3. **搜索策略微调**

   * BasicAgent 用的是 BayesOpt：

     * 优点：样本效率高。
     * 缺点：对于强噪声+非平滑 reward，表现可能不稳定。
   * 你有两个选择：

     * A. 继续用 BayesOpt，但：

       * 降低 `INITIAL_SEARCH` 和 `OPT_SEARCH`，同时每点做多次 MC（trade-off）。
       * 或调 kernel/α 等，让 GP 更平滑。
     * B. 换成更简单的**分层随机搜索（CEM / 多起点随机+局部细化）**：

       * 先粗随机采样 N0 个动作（几何上过滤掉明显不合理的）。
       * 选出 top-k 分数最高的，围绕它们局部采样（高斯微调）。
       * 总仿真次数控制在 <= BasicAgent 的数量级（比如 60–100 次）。

**优点：**

* 实现上可以**重用大量 BasicAgent 的代码**（尤其是 pooltool 沙盒和 reward 分析），工程负担小。
* 和 BasicAgent 比：

  * 更注重**局面安全/控制** → 专门针对它“只看一杆”的弱点。
  * 更注重**噪声鲁棒性** → 针对它“不考虑 env 噪声”的弱点。
* 容易做消融：

  * 去掉噪声平均 / 去掉位置项 / 去掉对手机会项，比较胜率差异。

**风险：**

* 若 reward 设计不好，可能“过度谨慎”导致进攻欲望太弱，从而在 60 杆上限下，剩球比对手多 → 输。
* MC + 搜索次数不好控制的话，运行时间可能偏长，需要你手动平衡。

**预期胜率：**

* 对 BasicAgent 有明显针对后，**70% 以上是现实目标**，通过良好调参和简单局面分析，很有机会冲 75–80%。

---

### 2.2 路线 2：几何候选 + 轻量搜索（结构化动作空间）

**核心思想：**
BasicAgent 搜索的是“裸的 5 维连续空间”，没有利用“台球几何”的结构。我们可以：

1. 先用几何规则构建**有限的候选动作池**（多是“合理线路”）；
2. 再用简单搜索（带噪声的 MC + reward）在这些候选里挑最优。

**候选动作生成（高层动作）示例：**

* 对每个己方目标球 `t` 和每个 pocket `p`：

  * 判断是否存在“母球 → 目标球 → 袋口”的直线线路且未被其他球阻挡：

    * 通过向量点积/叉积判断共线和遮挡。
  * 若可行，则计算 ideal 击球参数：

    * `phi*`: 母球到目标球方向。
    * `V0*`: 距离+经验公式（距离越远力度越大）。
    * `theta*`: 固定小角度或 0。
    * `a*,b*`: 默认小旋转，如少许下塞。
* 安全球候选：

  * 例如“把母球打到某个库边 + 贴近一颗己方/对方球”，让 BasicAgent 之后很难进攻（可粗略优化母球最终位置）。

然后：

* 每个高层动作周围做 3–5 个局部扰动（高斯 noise），形成 20–40 个候选（比 5 维全局搜索更结构化）。
* 对每个候选用前面说的“带噪声 MC + reward”评估。

**相对 BasicAgent 的优势：**

* 搜索空间是“有意义的动作”，而不是随便往任何方向打。
* 对“几何明显好打/安全的线路”有先验意识，样本效率更高。
* 便于你加入很多 domain prior：

  * 优先清理中袋/角袋更好打的球。
  * 避免打散对手的球堆。

**缺点/风险：**

* 实现几何判断比较费时间和脑细胞（但不难，只是细节多）。
* 一开始候选池可能不够丰富，导致错失一些高级击球（如 bank shot/组合球）。

**预期胜率：**

* 只做直线球 + 基本安全球 + MC 搜索：

  * 若设计得不错，有望接近或者超过 BasicAgent；
  * 再加“考虑噪声和位置”的 reward，有希望冲上 **70%**。
* 往上冲 80%，需要你进一步扩充候选类型（反弹球、连杆设计等）。

---

### 2.3 路线 3：启发式 + 轻量值函数 / RL 微调（混合）

**核心思想：**

* 低层动作仍用 路线 1/2 的启发式 + 搜索 决定；
* 再训练一个**值函数或者局部策略网络**，辅助选择和微调。

具体可以有两种形态：

1. **值函数辅助选择**：

   * 输入：局面特征（比如 16 球位置+标志）；
   * 输出：当前局面对“我方最终胜率”的估计 (V(s) \in [-1,1])。
   * 训练数据：

     * 让 BasicAgent vs NewAgent / BasicAgent vs BasicAgent 自博弈；
     * 每一步记录 `(s_t, 最终结果 z ∈ {1,0,-1})`；
     * 用 MSE 拟合 `V(s_t) ≈ z`。
   * 使用：

     * 对每个候选动作 `a_k`，我们模拟出结果局面 `s_k'`；
     * 总评分 `Q_k = immediate_score_k + λ V(s_k')`。
2. **局部微调网络**：

   * 给定启发式的 ideal 参数 ((V0^*, φ^*, θ^*, a^*, b^*))，网络输出小偏移量 (\Delta a)：

     * 来自一个小网络 `Δ = f(s, ideal_params)`；
     * 保证 (|Δ|) 在合理范围内（例如映射到 [-0.2,0.2] 之类）。

**优点：**

* 不依赖 RL 直接从零学台球，训练难度大幅下降；
* 启发式本身就能打，学习模块哪怕不完美，也只是“锦上添花”；
* 非常适合写报告里的 ablation（有/无值函数，有/无微调）。

**缺点：**

* 需要专门写 `train/` 下的训练脚本采集数据，跑若干局自博弈；
* 一定程度的调参成本（网络结构、学习率等）。

**预期胜率：**

* 启发式（路线 1/2）已经做到 65–75% 后，加入值函数很可能再稳定提升一点，对 70–80% 打辅助。

---

## 3. 最推荐的主方案 & 兜底方案（结合真实代码）

### 3.1 主方案：**“BasicAgent++ + 噪声鲁棒 + 简单安全策略”**

> 在 NewAgent 中重用 BasicAgent 的物理仿真框架（`pt.System` + `analyze_shot_for_reward`），
> 加一个“更看重噪声鲁棒和长远局势”的 reward + 搜索策略调整。

**核心设计：**

1. **NewAgent 继承 BasicAgent（建议）**

   在 `agent.py` 布局里，把：

   ```python
   class NewAgent(Agent):
       ...
   ```

   改成：

   ```python
   class NewAgent(BasicAgent):
       def __init__(self):
           super().__init__()
           # 调自己的参数，比如搜索轮数、是否启用reward噪声等
   ```

   这样可以直接复用 `_create_optimizer`、`pbounds` 等内部工具，只在 `decision` 里覆盖/扩展逻辑。

2. **状态特征（用于改进 reward）**

   不需要额外传入，只用 BasicAgent decision 里已有的：

   * `balls`：能判断：

     * 每个球是否 pocketed (`state.s == 4`)；
     * cue 与每颗 target ball 的距离；
     * 球之间的遮挡情况（用几何判断）。
   * `my_targets`：知道自己还剩哪些球，以及是否只剩黑 8。
   * `table`：知道桌面尺寸 & 袋口位置。

3. **动作空间 & 搜索**

   * 仍然在 5D 空间上搜索（借用 BayesOpt 或简单多阶段随机搜索）。
   * 噪声鲁棒策略：

     * 在 reward 函数中，对于每一组 `(V0,φ,θ,a,b)`，做 K 次独立仿真：

       * 各次在参数上加高斯噪声，与 env 的噪声 std 一致或略大；
       * 对每次 `shot` 用增强版 `analyze_shot_for_reward_robust` 评分；
       * 取平均作为该参数点的 target。
   * 若觉得 BayesOpt 对多次 MC 不够友好，可考虑换成：

     * 先随机采样 N0 个点评估；
     * 选出 top-k 的中心，再在其附近微调采样 N1 个。

4. **增强版 reward：`analyze_shot_for_reward_robust`**

   在现有 reward 的基础上，加：

   * **母球位置好坏**：

     * 估计下一杆自己最容易打的目标球 `t_next`（距离 cue 最近、无遮挡）；
     * 加项：`-α * distance(cue, t_next)`（距离越近越好）。
   * **对手机会压制**：

     * 模拟这一杆结束后，假设对手是 BasicAgent：

       * 不必真的跑 BasicAgent，一开始可以用简化指标：

         * 统计对手目标球中“直接可见且无遮挡”的球数 `n_shots_enemy`；
       * 加惩罚：`-β * n_shots_enemy`。
   * **避免帮对手解球**：

     * 如果这杆明显减少了“对手球 cluster 稠密程度”（比如两颗对手球距离明显增大），可加一点 penalty。
   * **避免高风险 8 球**：

     * 若自己目标未清空，就尽量避免把黑 8 打到袋口区域附近（可用黑 8 与最近袋口的距离度量）。

5. **针对 BasicAgent 的策略倾向**

   根据 BasicAgent 的结构，我们可以合理假设它：

   * 专注当前一杆 reward；
   * 对“我方下一杆局面 / 对手机会”认识不足。

   所以 NewAgent 的风格可以：

   * 在自己没有稳高成功率进球时，**优先选择安全球**，让局面变得“高难度进攻 + 带风险”；
   * 通过适度“拖延 + 控制局势”，让 BasicAgent 在 60 杆限制下剩余球更多，从而判负。

   这在比赛逻辑中是合法且非常有效的 exploit：
   你不是要成为完美球手，而是要成为**虐 BasicAgent 的球手**。

---

### 3.2 兜底方案 1：**“复制 BasicAgent + 轻微改 reward”**

如果时间紧、几何/噪声/安全球还没来得及完全实现，可以做一个更轻量的备胎：

* 完全复用 BasicAgent 的 BayesOpt + 模拟框架；
* 只对 `analyze_shot_for_reward` 做几处小改（在 NewAgent 自己版本里）：

  * 略提高“合法无进球”的奖励（鼓励一些安全球行为）；
  * 稍微增加“对手球入袋”的惩罚；
  * 加一项：母球距离桌边的罚/奖，避免白球落到桌中央让对手容易清台。
* 不做多次 MC，只单次评估。

预期：

* 在 BasicAgent 已经不错的基础上，这种改动有很大概率带来**小幅稳定提升**（比如 55–65% → 60–70%），
* 至少不会比 BasicAgent 弱，是很好的“保险线”。

### 3.3 兜底方案 2：几何直线球策略（不使用 BayesOpt）

* 若你发现 BayesOpt 库在机器上不好装 / 不稳定，或者不想在 NewAgent 里依赖它：

  * 只用**几何+随机搜索**：

    * 构造直线可打的目标球+袋组合；
    * 在附近做 10–20 个随机扰动；
    * 按 reward（即便是 BasicAgent 的原版 reward）评分，选最优。
* 这相比 BasicAgent 的优势在于：搜索空间是“有意义的直线球”，可能在较小仿真数下也能打得不错。

---

## 4. 分阶段项目计划（结合真实接口的具体任务）

### 阶段 0：环境熟悉 & baseline 验证

**任务：**

* 跑 `python evaluate.py`：

  * 记录 BasicAgent vs NewAgent(当前随机) 的结果；
  * 通过修改 `agent_b` 为 `BasicAgent()` 验证 BasicAgent vs BasicAgent 约在 50%。
* 用一个极简 NewAgent：

  * 比如 `_random_action`；
  * 或者“简单直线最近球”版本；
  * 查看其与 BasicAgent 的对战结果，作为下界。

**风险 &对策：**

* PoolEnv 输出很多 debug 打印可能比较吵：

  * 可以考虑在调试时临时减小打印（但注意最终评测不要改官方文件）。

---

### 阶段 1：实现主方案的 MVP（NewAgent ≈ BasicAgent++ 的雏形）

**文件组织：**

* `agent.py`：

  * 修改 `NewAgent` 继承 `BasicAgent`，覆盖 `decision`。
* `train/`：

  * 暂时只放一些你自己的测试脚本（不一定立刻做训练），比如：

    * `play_vs_basic.py`：多个版本的 NewAgent 与 BasicAgent 对打。
* `eval/`：

  * `eval_vs_basic.py`：封装“循环调用 evaluate.py，统计平均胜率”。

**MVP 目标：**

* 先做一个**“增强 reward 但不加多次 MC”的 NewAgent**：

  1. 在 NewAgent.decision 内部拷贝 BasicAgent 的逻辑：

     * 但使用你自定义的 `analyze_shot_for_reward_plus`（多考虑一点母球位置/对手机会）。
  2. 仍然使用 BayesOpt（`_create_optimizer`）。
  3. 初步调节 reward 权重，让行为看上去**不明显变蠢**（可以打印一些关键局面观感）。

* 对比：

  * BasicAgent vs BasicAgent
  * BasicAgent vs NewAgent（MVP）

* **MVP 成功标准**：

  * 在几十局测试里，NewAgent 胜率能略高于 BasicAgent（比如 55–60%）。

---

### 阶段 2：加入噪声鲁棒 + 安全球策略（性能提升阶段）

**任务：**

1. **在 reward 中加入参数噪声 MC**

   * 在 NewAgent.reward 函数（或 `decision` 内 wrapper）里：

     * 对给定 `(V0,φ,θ,a,b)` repeat K 次（例如 K=3～5）：

       * 用高斯噪声扰动参数；
       * 模拟 + 评分；
     * 取平均作为 BayesOpt/搜索的 target。
   * 实验：

     * 比较 K=1 vs 3 vs 5 对胜率的影响和运行时间。

2. **加入简易安全球启发式**

   * 简单版本（不需要特别复杂几何）：

     * 当 reward Fn 在所有附近候选上都比较低（认为“没有好进球”）时：

       * 构造几种“安全球模板”，如：

         * 把母球打向最近的库边；
         * 把母球打向一个 cluster，力道小，使其停在球堆后方；
       * 对这些安全球模板也做 MC 评估，加入候选。
   * reward 里强调：

     * 不犯规；
     * 不送对手球进袋；
     * 让对手“可直线击打的球数”尽可能少。

3. **调参与 ablation**

   * 系统性实验：

     * `无 MC` vs `MC(K=3)` vs `MC(K=5)`
     * `无安全球` vs `有安全球`
     * `原 reward` vs `增强 reward`
   * 记录每种设置下，对 BasicAgent 的胜率。

**风险 &对策：**

* MC 太耗时：

  * 你可以在搜索中对候选进行分组：

    * 先只用 K=1 粗评估所有候选，选 top M；
    * 对这 M 个再做 K>1 精评估。
* reward 调太复杂导致“策略风格怪异”：

  * 用可视化 / log 的方式查看典型局面下 agent 的决策，手动微调权重。

---

### 阶段 3：如果有余力——值函数/局部 RL 微调 + 报告实验完善

**任务：**

1. **采集对局数据**

   * 在 `train/` 写脚本：

     * NewAgent（当前版本） vs BasicAgent 连续打若干局；
     * 记录每次击球的 `(state features, immediate reward, final result)`。
   * 状态特征编码：

     * 16 个球的位置 (x,y) + alive flag + type one-hot；
     * 当前方/对手剩余球数；
     * 是否只剩黑 8 等。

2. **训练简单值网络 `V(s)`**

   * 模型：

     * MLP，2–3 层，Hidden=128–256，ReLU。
   * 训练：

     * 标签 z ∈ {1,0,-1}（胜/平/负）；
     * 损失 MSE。
   * 用于：

     * 在搜索评估中加入一项 λ·V(s')。

3. **报告实验**

   * 计划好的图表：

     * NewAgent 不同版本（MVP → MC → MC+安全球 → MC+安全球+V）的胜率曲线。
     * rollouts 数 vs 胜率 & 决策时间的 trade-off。
     * 噪声强度变化下（如果在 train 里可调）的鲁棒性表现。

---

## 5. 关键技术细节与实现建议（更具体一点）

### 5.1 如果用 BayesOpt（沿用 BasicAgent）

* 你可以直接复用 `_create_optimizer`：

  * Matern 核 + `alpha=1e-2` + `SequentialDomainReductionTransformer` 已写好。
* 如果你要在 reward 中用 MC 评估：

  * 注意不要在 reward_eval 内部开太大循环，否则一次 `optimizer.maximize` 会很慢。
  * 实际可以：

    * 保持 `INITIAL_SEARCH=10~15`, `OPT_SEARCH=5~10`；
    * 每个 f(x) 做 K=3 的 MC 即可。

### 5.2 如果改用简单随机搜索 / CEM

* 伪流程（单杆）：

  ```text
  1. 生成 N0 个随机动作 samples, 在 pbounds 内均匀采样。
  2. 对每个样本做 K 次噪声仿真，平均 reward。
  3. 选出 top-k 样本 S_top。
  4. 在 S_top 附近高斯采样 N1 个新样本（方差可减小）。
  5. 再次评估，取全局最优动作。
  ```

* N0, N1, K 需要你现场调优（例如 30, 20, 3）。

### 5.3 观测特征构造示例

即便你暂时不训练网络，也可以用这些特征来写启发式：

* 对己方目标球：

  * 计算每颗球到近几个袋口的几何可打性（遮挡 + 角度）。
* 对对手目标球：

  * 统计“当前对手可直线进攻球数”。
* 对母球：

  * 距离最近己方目标球的距离；
  * 距离桌边、袋口的距离。

这些都只需要你在 `decision(balls, my_targets, table)` 中做一些 geometry 计算即可。
