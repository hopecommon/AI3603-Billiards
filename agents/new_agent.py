"""
agent_optimized.py - 速度优化版 NewAgent

核心优化原理：
1. 智能候选剪枝：只生成高质量候选（90个→15个）
2. 分层采样策略：快速筛选(2次) → 精细评估(6次)
3. 早停机制：高置信度立即返回
4. 移除贝叶斯回退：CMA-ES 足够强大
5. 自适应计算预算：根据局面复杂度动态调整

预期性能：
- 速度提升：3分钟/杆 → 15秒/杆 (12x)
- 胜率保持：70-80% (通过智能剪枝保留高质量候选)
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
import json
from datetime import datetime
import random
import signal
from pathlib import Path

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .agent import Agent

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("[WARNING] CMA-ES 不可用，将使用简化优化")

def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class OptimizedNewAgent(Agent):
    """速度优化版 NewAgent - 保持70%+胜率的前提下提速12倍"""
    
    SOLID_IDS = ['1', '2', '3', '4', '5', '6', '7']
    STRIPE_IDS = ['9', '10', '11', '12', '13', '14', '15']
    BALL_RADIUS = 0.028575
    
    def __init__(self, config: dict | str | Path | None = None):
        super().__init__()

        # Use a dedicated RNG for internal simulations/candidate sampling, so we do not
        # consume the global NumPy RNG state used by the environment noise.
        self.rng_seed = None
        self._rng = np.random.default_rng()
        
        # ============ 速度优化参数 ============
        
        # 1. 分层采样策略（最大瓶颈优化）
        self.samples_quick_filter = 1      # 快速筛选：1次采样（尽量减少慢模拟次数）
        self.samples_normal_eval = 2       # 正常评估：2次采样
        self.samples_final_verify = 4      # 最终验证：4次采样（配合硬过滤与自适应验证）
        self.samples_critical = 6          # 关键球（黑8）：6次采样（从8→6）
        
        # 2. Ghost Ball 智能剪枝
        self.max_target_balls = 3          # 只对最近的若干目标球生成候选
        self.max_ghost_candidates = 8      # 再减少候选，降低慢模拟概率
        self.max_pockets_per_ball = 2      # 每个球只尝试最近2个袋口
        self.max_speed_variants = 2        # 速度变体：2个
        self.max_spin_variants = 3         # 旋转变体：3个
        self.include_contact_fallback_candidates = True
        
        # 3. CMA-ES 优化
        # Default behavior should match `experiments/configs/final.json` unless explicitly overridden.
        self.use_cma_es = True
        self.cma_population_size = 4       
        self.cma_generations = 2           
        self.cma_sigma = 0.5
        self.cma_only_for_top_k = 1
        
        # 4. 移除贝叶斯回退（ROI太低）
        self.use_bayesian_fallback = False
        
        # 5. 早停阈值（更激进）
        self.early_stop_score = 80.0       # 得分>80立即返回（从85→80）
        self.good_enough_score = 60.0      # 得分>60跳过CMA-ES（从70→60）
        
        # 6. 安全球简化
        self.safety_max_candidates = 5
        self.safety_samples = 1
        
        # 7. 动态超时（根据剩余球数）
        self.timeout_base = 10             # 基础10秒
        self.timeout_per_ball = 1.0        # 每球+1秒
        self.timeout_critical = 20         # 黑8给20秒
        
        # ============ 策略参数（保持不变以维持胜率）============
        self.enable_strategic_bonus = True
        self.enable_positional_bonus = True
        self.enable_defense_threat = True
        self.enable_safety_shot = True

        self.cue_next_ball_radius = 1.2
        self.cue_next_ball_weight = 30.0
        self.enemy_threat_radius = 0.8
        self.enemy_distance_weight = 15.0
        self.eight_guard_radius = 0.15
        self.eight_guard_weight = 50.0
        self.safety_trigger_score = 25.0
        self.safety_prefer_margin = 10.0
        self.safe_speed_bounds = (1.8, 2.8)
        self.no_rail_penalty = 100.0
        self.white_scratch_penalty = 250.0
        self.illegal_black_penalty = 350.0
        self.cue_edge_margin = 0.15
        self.cue_edge_weight = 15.0
        self.min_pocket_probability = 0.4

        # ============ 风险控制（针对“即时判负”）============
        # PoolEnv 中以下情况会直接判负：白球+黑8同杆进袋、清台前黑8进袋
        # 我们主要依赖“致命犯规硬过滤”(stats["catastrophic"]>0 直接剔除) + 多阶段验证来规避。
        # 软惩罚项在部分对手上可能过于保守（消融实验显示禁用可提升胜率），因此默认关闭。
        self.catastrophic_foul_penalty = 0.0      # 非法黑8 / 白球+黑8（可在配置中开启）
        self.scratch_extra_penalty = 600.0        # 白球进袋（非即时判负，但会回滚+交换）
        self.first_hit_extra_penalty = 120.0      # 首球犯规（回滚+交换）
        self.no_rail_extra_penalty = 80.0         # 无碰库犯规（回滚+交换）
        self.no_hit_extra_penalty = 200.0         # 未击中任何球（回滚+交换）
        self.max_fallback_verifies = 3            # 最终动作若风险过高，最多额外验证/替换次数
        self.black_pocket_danger_dist = 0.14      # 黑8离袋口过近时，提升验证/更偏向安全（米）
        self.black_risk_extra_verifies = 2        # 黑8高风险局面额外验证次数（降低随机漏检）
        # 不好进攻时的“走位杆”参数（替代随机安全球）
        self.positional_trigger_score = 35.0      # 进攻分低于此时考虑走位杆
        self.positional_prefer_margin = 6.0       # 走位杆超过进攻多少才选
        self.opponent_pot_threat_weight = 35.0    # 走位杆核心：降低对手下一杆轻松进球概率（轻量近似）

        # 走位杆候选规模（控制耗时；走位杆只需要“够用”，不需要大搜索）
        self.positional_max_targets = 1
        self.positional_angle_offsets = (-6.0, -3.0, 0.0, 3.0, 6.0)
        self.positional_speeds = (1.8, 2.4)
        self.positional_spin_variants = ((0.0, 0.0), (0.0, 0.10))
        self.positional_random_angles = 3
        self.positional_early_stop_score = 32.0
        
        self.my_target_type = None

        self._apply_config(config)
        if self.rng_seed is not None:
            self._rng = np.random.default_rng(int(self.rng_seed))
        
        print("OptimizedNewAgent 已初始化 (速度优化版，保持70%+胜率)")

    def _apply_config(self, config: dict | str | Path | None) -> None:
        if config is None:
            return
        if isinstance(config, (str, Path)):
            path = Path(config)
            with path.open("r", encoding="utf-8") as f:
                config_dict = json.load(f)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config must be dict or a JSON path")

        for key, value in config_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown OptimizedNewAgent config key: {key}")
            setattr(self, key, value)
    
    def decision(self, balls=None, my_targets=None, table=None):
        """优化后的决策流程（带性能分析）"""
        import time
        
        if balls is None or table is None:
            return self._random_action()
        
        decision_start = time.time()
        
        # 动态计算超时
        prepared_targets = self._prepare_targets(balls, my_targets)
        is_critical = ('8' in prepared_targets) or (len(prepared_targets) == 1)
        
        if is_critical:
            timeout = self.timeout_critical
        else:
            n_remaining = len(prepared_targets)
            timeout = int(self.timeout_base + n_remaining * self.timeout_per_ball)
        
        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._shot_alarm_handler)
        signal.alarm(timeout)
        
        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            print(f"[OptimizedAgent] 目标={prepared_targets}, 超时={timeout}s, 关键球={is_critical}")
            dangerous_black = self._black8_is_dangerous(balls, table, prepared_targets)
            if dangerous_black and not is_critical:
                print("[OptimizedAgent] 检测到黑8高风险位置：提高验证强度并更保守选择")
            
            # ========== 优化后的决策流程 ==========
            
            # Step 1: 智能生成 Ghost Ball 候选（≈450→8个）
            t1 = time.time()
            ghost_candidates = self._generate_smart_ghost_candidates(
                balls, table, prepared_targets
            )
            t1_elapsed = time.time() - t1
            print(f"[Step 1] 生成 {len(ghost_candidates)} 个候选 - 耗时: {t1_elapsed:.2f}s")
            
            if not ghost_candidates:
                print("[OptimizedAgent] 无候选，尝试生成“保证碰球”的fallback候选")
                fallback_candidates = self._generate_contact_fallback_candidates(balls, table, prepared_targets)
                if not fallback_candidates:
                    print("[OptimizedAgent] fallback 也失败，使用保守动作")
                    return self._conservative_action(balls, table)
                best_action = None
                best_score = -1e9
                for cand in fallback_candidates:
                    score = self._evaluate_action_fast(
                        cand, balls, table, last_state_snapshot, prepared_targets,
                        samples=1, return_stats=False
                    )
                    if score > best_score:
                        best_score = score
                        best_action = cand
                return best_action if best_action is not None else self._conservative_action(balls, table)
            
            # Step 2: 快速筛选（2次采样）
            t2 = time.time()
            quick_scores = []
            for candidate in ghost_candidates:
                if self._is_degenerate_params(candidate):
                    quick_scores.append((-500, candidate))
                    continue
                
                score = self._evaluate_action_fast(
                    candidate, balls, table, last_state_snapshot,
                    prepared_targets, self.samples_quick_filter
                )
                quick_scores.append((score, candidate))
            
            # 按得分排序，取前K个
            quick_scores.sort(key=lambda x: x[0], reverse=True)
            top_k = min(3, len(quick_scores))
            top_candidates = [item[1] for item in quick_scores[:top_k]]
            
            t2_elapsed = time.time() - t2
            total_sims_step2 = len(ghost_candidates) * self.samples_quick_filter
            print(f"[Step 2] 快速筛选 {len(ghost_candidates)}个候选×{self.samples_quick_filter}次采样={total_sims_step2}次模拟 - 耗时: {t2_elapsed:.2f}s")
            print(f"         保留前{len(top_candidates)}个，最高分: {quick_scores[0][0]:.1f}")
            
            # 早停检查：必须先做“致命犯规”验证，否则极易选到非法黑8/白球+黑8导致直接判负
            if quick_scores[0][0] > self.good_enough_score:
                best_candidate = quick_scores[0][1]
                # 自适应验证：一般局面少采样，高风险再加采样
                base_verify = 3 if (not is_critical) else self.samples_critical
                samples_verify = base_verify if is_critical else base_verify + (self.black_risk_extra_verifies if dangerous_black else 0)
                verify_score, verify_stats = self._evaluate_action_fast(
                    best_candidate, balls, table, last_state_snapshot,
                    prepared_targets, samples_verify, return_stats=True
                )
                print(f"[OptimizedAgent] 早停候选验证: score={verify_score:.1f}, catastrophic={verify_stats['catastrophic']}/{verify_stats['samples']}")
                if verify_stats["catastrophic"] == 0 and verify_score > self.good_enough_score:
                    print(f"[OptimizedAgent] 早停确认：验证通过，跳过后续优化")
                    return best_candidate
                print(f"[OptimizedAgent] 早停取消：验证未通过，继续优化")
            
            # Step 3: 精细评估（3次采样）
            t3 = time.time()
            samples = self.samples_critical if is_critical else self.samples_normal_eval
            
            refined_scores = []
            for candidate in top_candidates:
                score, stats = self._evaluate_action_fast(
                    candidate, balls, table, last_state_snapshot,
                    prepared_targets, samples, return_stats=True
                )
                # 精评阶段：一旦出现“即时判负”样本，直接剔除该候选，降低漏检概率
                if stats["catastrophic"] > 0:
                    refined_scores.append((-1e9, candidate))
                else:
                    refined_scores.append((score, candidate))
            
            refined_scores.sort(key=lambda x: x[0], reverse=True)
            best_score, best_action = refined_scores[0]
            
            t3_elapsed = time.time() - t3
            total_sims_step3 = len(top_candidates) * samples
            print(f"[Step 3] 精细评估 {len(top_candidates)}个候选×{samples}次采样={total_sims_step3}次模拟 - 耗时: {t3_elapsed:.2f}s")
            print(f"         最高分: {best_score:.1f}")
            
            # 早停检查3：精评得分高时可跳过 CMA，但必须先做最终验证（否则容易漏掉“误打黑8/白+8”）
            if best_score > self.good_enough_score:
                base_verify = 3 if (not is_critical) else self.samples_critical
                samples_verify = base_verify if is_critical else base_verify + (self.black_risk_extra_verifies if dangerous_black else 0)
                verify_score, verify_stats = self._evaluate_action_fast(
                    best_action, balls, table, last_state_snapshot,
                    prepared_targets, samples_verify, return_stats=True
                )
                print(f"[OptimizedAgent] 早停3验证: score={verify_score:.1f}, catastrophic={verify_stats['catastrophic']}/{verify_stats['samples']}")
                if verify_stats["catastrophic"] == 0 and verify_score > self.good_enough_score * 0.9:
                    print(f"[OptimizedAgent] 早停3确认：验证通过，跳过后续优化")
                    return best_action
                print(f"[OptimizedAgent] 早停3取消：验证未通过，继续搜索/优化")
            
            # Step 4: CMA-ES 优化（仅对前1个候选）
            t4 = time.time()
            cma_improved = False
            # CMA-ES 很耗时，仅在“当前最优还不够好”时尝试
            if self.use_cma_es and CMA_AVAILABLE and (not is_critical) and (best_score < self.good_enough_score):
                cma_candidates = refined_scores[:self.cma_only_for_top_k]
                
                for i, (score, candidate) in enumerate(cma_candidates):
                    try:
                        optimized, opt_score = self._cma_es_optimize_fast(
                            candidate, balls, table, last_state_snapshot,
                            prepared_targets, samples
                        )
                        
                        if opt_score > best_score:
                            improvement = opt_score - best_score
                            best_score = opt_score
                            best_action = optimized
                            cma_improved = True
                            print(f"         CMA-ES候选{i}: {opt_score:.1f} (提升+{improvement:.1f})")
                    
                    except Exception as e:
                        print(f"         CMA-ES失败: {e}")
                        continue
            
            t4_elapsed = time.time() - t4
            if self.use_cma_es and CMA_AVAILABLE and not is_critical:
                total_sims_step4 = self.cma_only_for_top_k * self.cma_population_size * self.cma_generations * samples
                print(f"[Step 4] CMA-ES优化 {self.cma_only_for_top_k}个候选×{self.cma_population_size}种群×{self.cma_generations}代×{samples}采样≈{total_sims_step4}次模拟 - 耗时: {t4_elapsed:.2f}s")
                print(f"         改进: {'是' if cma_improved else '否'}")
            else:
                print(f"[Step 4] 跳过CMA-ES (关键球或未启用)")
            
            # Step 5: 走位杆检查（进攻得分低/黑8高风险时）
            t5 = time.time()
            t5_elapsed = 0.0
            safety_chosen = False
            positional_trigger = None

            if self.enable_safety_shot:
                safety_trigger = self.safety_trigger_score
                if dangerous_black and not is_critical:
                    # 黑8离袋口近时，宁可打防守避免“误打黑8/白+8”的高方差
                    safety_trigger = max(safety_trigger, 45.0)
                positional_trigger = max(safety_trigger, self.positional_trigger_score)
                if best_score < positional_trigger:
                    safe_action, safe_score = self._plan_positional_shot_fast(
                        balls, table, last_state_snapshot, prepared_targets
                    )
                    if safe_action is not None and safe_score > best_score + self.positional_prefer_margin:
                        print(f"[Step 5] 走位杆检查 - 选择走位杆: {safe_score:.1f} (vs 进攻 {best_score:.1f})")
                        safety_chosen = True
                        best_action = safe_action
                        best_score = safe_score

                t5_elapsed = time.time() - t5
                if best_score < positional_trigger:
                    print(f"[Step 5] 走位杆检查 - 耗时: {t5_elapsed:.2f}s, 选择: {'走位杆' if safety_chosen else '进攻'}")
                else:
                    print(f"[Step 5] 跳过安全球检查 (进攻得分{best_score:.1f}已足够)")
            else:
                t5_elapsed = time.time() - t5
                print("[Step 5] 走位/安全策略已禁用")
            
            # Step 6: 最终验证（4次采样）
            t6 = time.time()
            samples_final = self.samples_critical if is_critical else (
                self.samples_final_verify + (self.black_risk_extra_verifies if dangerous_black else 0)
            )
            final_score, final_stats = self._evaluate_action_fast(
                best_action, balls, table, last_state_snapshot,
                prepared_targets, samples_final, return_stats=True
            )
            t6_elapsed = time.time() - t6
            
            print(f"[Step 6] 最终验证 1个动作×{samples_final}次采样 - 耗时: {t6_elapsed:.2f}s, 得分: {final_score:.1f}, catastrophic={final_stats['catastrophic']}/{final_stats['samples']}")

            # 若最终验证仍出现“即时判负”样本，尝试从备选中挑一个更稳的
            if final_stats["catastrophic"] > 0:
                fallback_action = self._pick_safer_action(
                    refined_scores=refined_scores,
                    balls=balls,
                    table=table,
                    last_state_snapshot=last_state_snapshot,
                    prepared_targets=prepared_targets,
                    is_critical=is_critical,
                )
                if fallback_action is not None:
                    print("[OptimizedAgent] 最终验证发现致命风险，已切换到更安全的备选动作")
                    best_action = fallback_action
                else:
                    # 强硬兜底：绝不带“致命风险”出杆。尝试安全球，再尝试保证碰球的兜底候选。
                    print("[OptimizedAgent] 最终验证发现致命风险且无可用备选：强制切换到安全/兜底动作")

                    if self.enable_safety_shot:
                        safety_action, _ = self._plan_positional_shot_fast(
                            balls, table, last_state_snapshot, prepared_targets
                        )
                        if safety_action is not None:
                            verify_score, verify_stats = self._evaluate_action_fast(
                                safety_action, balls, table, last_state_snapshot,
                                prepared_targets, samples_final, return_stats=True
                            )
                            print(f"[OptimizedAgent] safety 再验证: score={verify_score:.1f}, catastrophic={verify_stats['catastrophic']}/{verify_stats['samples']}")
                            if verify_stats["catastrophic"] == 0:
                                best_action = safety_action
                                return best_action

                    contact_candidates = self._generate_contact_fallback_candidates(
                        balls, table, prepared_targets, max_targets=2
                    )
                    best_contact = None
                    best_contact_score = -1e18
                    for cand in contact_candidates:
                        sc, st = self._evaluate_action_fast(
                            cand, balls, table, last_state_snapshot,
                            prepared_targets, samples=min(3, samples_final), return_stats=True
                        )
                        if st["catastrophic"] > 0:
                            continue
                        if sc > best_contact_score:
                            best_contact_score = sc
                            best_contact = cand
                    if best_contact is not None:
                        print(f"[OptimizedAgent] 兜底碰球动作启用: score={best_contact_score:.1f}")
                        return best_contact

                    print("[OptimizedAgent] 所有兜底均失败，使用保守动作")
                    return self._conservative_action(balls, table)
            
            total_elapsed = time.time() - decision_start
            total_sims = total_sims_step2 + total_sims_step3
            if self.use_cma_es and CMA_AVAILABLE and not is_critical:
                total_sims += total_sims_step4
            total_sims += samples_final
            
            print(f"\n[总结] 总耗时: {total_elapsed:.2f}s, 总模拟次数: ~{total_sims}, 最终得分: {best_score:.1f}")
            print(f"       时间分布: 生成{t1_elapsed:.1f}s + 筛选{t2_elapsed:.1f}s + 精评{t3_elapsed:.1f}s + CMA{t4_elapsed:.1f}s + 安全{t5_elapsed:.1f}s + 验证{t6_elapsed:.1f}s")
            
            final_action = best_action if best_action else self._conservative_action(balls, table)
            final_action = self._sanitize_action(final_action)
            return final_action if final_action is not None else self._conservative_action(balls, table)
        
        except TimeoutError:
            print(f"[OptimizedAgent] 超时 ({timeout}s)，使用保守动作")
            return self._conservative_action(balls, table)
        except Exception as e:
            # 兜底：评测时绝不能让 agent 崩溃导致整局中断
            print(f"[OptimizedAgent] decision 异常，使用保守动作: {e}")
            return self._conservative_action(balls, table)
        
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
    
    # ========== 智能候选生成（核心优化）==========
    
    def _generate_smart_ghost_candidates(self, balls, table, player_targets):
        """智能生成高质量 Ghost Ball 候选（≈450→8个）
        
        优化策略：
        1. 只对前3个目标球生成候选
        2. 每个球只尝试最近2个袋口
        3. 速度变体：2个（基础速度 × [0.9, 1.1]）
        4. 旋转变体：3个（无旋转、低杆、高杆）
        5. 预筛选：移除明显不可行的候选
        
        复杂度：3球 × 2袋口 × 2速度 × 3旋转 = 36个候选（再 cap 到 max_ghost_candidates）
        """
        candidates = []
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return candidates
        
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        pockets = list(table.pockets.values())
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in pockets]
        
        # 只处理最近的若干目标球（按距离排序）
        valid_targets = []
        for bid in player_targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            dist = np.linalg.norm(cue_pos - ball_pos)
            valid_targets.append((dist, bid, ball_pos))
        
        valid_targets.sort(key=lambda x: x[0])
        valid_targets = valid_targets[: max(1, int(self.max_target_balls))]
        
        for _, bid, ball_pos in valid_targets:
            # 找到最近的2个袋口
            pocket_dists = [(i, np.linalg.norm(ball_pos - p)) 
                           for i, p in enumerate(pocket_positions)]
            pocket_dists.sort(key=lambda x: x[1])
            nearest_pockets = pocket_dists[:self.max_pockets_per_ball]
            
            for pocket_idx, _ in nearest_pockets:
                pocket_pos = pocket_positions[pocket_idx]
                
                # 计算 Ghost Ball
                ball_to_pocket = pocket_pos - ball_pos
                dist_to_pocket = np.linalg.norm(ball_to_pocket)
                if dist_to_pocket < 1e-3:
                    continue
                
                ball_to_pocket_unit = ball_to_pocket / dist_to_pocket
                ghost_pos = ball_pos - ball_to_pocket_unit * (2 * self.BALL_RADIUS)
                
                # 快速路径检查（简化版）
                if self._is_path_blocked_fast(cue_pos, ghost_pos, balls, bid):
                    continue
                
                # 计算角度
                cue_to_ghost = ghost_pos - cue_pos
                dist_to_ghost = np.linalg.norm(cue_to_ghost)
                if dist_to_ghost < 1e-3:
                    continue
                
                phi = math.degrees(math.atan2(cue_to_ghost[1], cue_to_ghost[0])) % 360
                
                # 计算基础速度
                base_speed = self._calculate_optimal_speed(dist_to_ghost, dist_to_pocket)
                
                # Speed variants around the geometric estimate.
                if int(self.max_speed_variants) <= 1:
                    speed_variants = [base_speed]
                else:
                    multipliers = np.linspace(0.9, 1.1, int(self.max_speed_variants))
                    speed_variants = [base_speed * float(m) for m in multipliers]

                # Spin variants (trimmed pool for ablations).
                spin_pool = [
                    (0.0, 0.0),      # no spin
                    (0.0, -0.15),    # draw
                    (0.0, 0.15),     # follow
                    (0.12, 0.0),     # left
                    (-0.12, 0.0),    # right
                ]
                spin_variants = spin_pool[: max(1, int(self.max_spin_variants))]
                
                for speed in speed_variants:
                    for spin_a, spin_b in spin_variants:
                        candidates.append({
                            'V0': float(np.clip(speed, 0.5, 8.0)),
                            'phi': float(phi),
                            'theta': 2.0,
                            'a': float(spin_a),
                            'b': float(spin_b)
                        })

        # 兜底：加入“直接撞击最近目标球”的候选，避免路径检查过严导致无解
        if self.include_contact_fallback_candidates:
            direct_hits = self._generate_contact_fallback_candidates(balls, table, player_targets, max_targets=2)
            candidates.extend(direct_hits)
        
        # 限制总候选数
        if len(candidates) > self.max_ghost_candidates:
            # 随机采样保持多样性
            indices = self._rng.choice(len(candidates), self.max_ghost_candidates, replace=False)
            candidates = [candidates[i] for i in indices]
        
        return candidates

    def _generate_contact_fallback_candidates(self, balls, table, player_targets, max_targets=1):
        """生成低成本兜底候选：保证至少能碰到合法目标球，避免 NO_HIT/首球犯规。

        设计目标：
        - 不追求进球，只要不犯规并尽量别白球落袋
        - 候选数量小，便于快速评估
        """
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)

        # 选最近的若干目标球
        targets = []
        for bid in player_targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            targets.append((float(np.linalg.norm(ball_pos - cue_pos)), bid, ball_pos))
        if not targets:
            return []
        targets.sort(key=lambda x: x[0])
        targets = targets[:max_targets]

        candidates = []
        angle_offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
        for dist, _, ball_pos in targets:
            direction = ball_pos - cue_pos
            if np.linalg.norm(direction) < 1e-6:
                continue
            base_phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            # 力度：足够推进目标球碰库，避免太大导致白球乱飞/落袋
            base_v0 = float(np.clip(1.6 + dist * 1.8, 1.8, 4.2))
            for off in angle_offsets:
                candidates.append({
                    'V0': base_v0,
                    'phi': float((base_phi + off) % 360),
                    'theta': 2.0,
                    'a': 0.0,
                    'b': 0.0
                })
        return candidates
    
    def _is_path_blocked_fast(self, start, end, balls, ignore_id):
        """快速路径阻挡检查（简化版）"""
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist < 1e-3:
            return False
        
        direction = direction / dist
        
        # 只检查距离路径很近的球
        for bid, ball in balls.items():
            if bid == 'cue' or bid == ignore_id or ball.state.s == 4:
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            to_ball = ball_pos - start
            proj = np.dot(to_ball, direction)
            
            if proj < 0 or proj > dist:
                continue
            
            perp_dist = np.linalg.norm(to_ball - proj * direction)
            if perp_dist < 2.5 * self.BALL_RADIUS:  # 放宽阈值加速
                return True
        
        return False
    
    def _calculate_optimal_speed(self, dist_to_ghost, dist_to_pocket):
        """计算最优速度"""
        total_dist = dist_to_ghost + dist_to_pocket
        
        if total_dist < 0.5:
            return 1.5
        elif total_dist < 1.0:
            return 2.5
        elif total_dist < 2.0:
            return 3.5
        else:
            return min(4.5 + (total_dist - 2.0) * 0.5, 7.0)
    
    # ========== 快速评估函数 ==========
    
    def _evaluate_action_fast(self, action, balls, table, last_state, targets, samples, return_stats=False):
        """快速评估（使用分层采样）

        关键：对 PoolEnv 的“即时判负”规则做硬风控（非法黑8、白球+黑8）。
        """
        action = self._sanitize_action(action)
        if action is None:
            empty = {
                "samples": 0,
                "catastrophic": 0,
                "scratch": 0,
                "foul_first_hit": 0,
                "foul_no_rail": 0,
                "no_hit": 0,
                "illegal_eight": 0,
                "white_and_eight": 0,
            }
            return (-500.0, empty) if return_stats else -500.0
        return self._evaluate_action_fast_impl(
            action, balls, table, last_state, targets, samples, return_stats=return_stats
        )

    def _sanitize_action(self, action):
        """确保动作字典完整且数值在合法范围内，避免评测过程因异常动作崩溃。"""
        if action is None or not isinstance(action, dict):
            return None
        required = ("V0", "phi", "theta", "a", "b")
        if any(k not in action for k in required):
            return None
        try:
            V0 = float(action["V0"])
            phi = float(action["phi"])
            theta = float(action["theta"])
            a = float(action["a"])
            b = float(action["b"])
        except Exception:
            return None

        # clip 到环境允许范围
        V0 = float(np.clip(V0, 0.5, 8.0))
        phi = float(phi % 360.0)
        theta = float(np.clip(theta, 0.0, 90.0))
        a = float(np.clip(a, -0.5, 0.5))
        b = float(np.clip(b, -0.5, 0.5))
        return {"V0": V0, "phi": phi, "theta": theta, "a": a, "b": b}

    def _evaluate_action_fast_impl(self, action, balls, table, last_state, targets, samples, return_stats):
        scores = []
        catastrophic = 0
        scratch = 0
        foul_first_hit = 0
        foul_no_rail = 0
        no_hit = 0
        illegal_eight = 0
        white_and_eight = 0

        def _empty_stats():
            return {
                "samples": 0,
                "catastrophic": 0,
                "scratch": 0,
                "foul_first_hit": 0,
                "foul_no_rail": 0,
                "no_hit": 0,
                "illegal_eight": 0,
                "white_and_eight": 0,
            }
        
        for _ in range(samples):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            # Table 在物理模拟中是只读结构，避免 deepcopy 可以显著省时
            sim_table = table
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            noisy = self._sample_noisy_params(
                action['V0'], action['phi'], action['theta'],
                action['a'], action['b']
            )
            
            if self._is_degenerate_params(noisy):
                return (-500.0, _empty_stats()) if return_stats else -500.0
            
            shot.cue.set_state(**noisy)
            
            try:
                pt.simulate(shot, inplace=True)
            except Exception:
                return (-500.0, _empty_stats()) if return_stats else -500.0
            
            # 使用原有的奖励函数
            base_score = analyze_shot_for_reward(shot, last_state, targets)
            
            # 添加走位和防守奖励
            bonus = self._calculate_strategic_bonus(shot, table, targets, last_state)
            foul_info = self._analyze_fouls(shot, last_state, targets)
            sample_score = base_score + bonus

            # 即时判负：必须极重惩罚（否则会被 +50/球 等奖励掩盖）
            if foul_info["WHITE_AND_EIGHT"] or foul_info["ILLEGAL_EIGHT"]:
                catastrophic += 1
                if foul_info["WHITE_AND_EIGHT"]:
                    white_and_eight += 1
                if foul_info["ILLEGAL_EIGHT"]:
                    illegal_eight += 1
                sample_score -= self.catastrophic_foul_penalty
            elif foul_info["CUE_POCKETED"]:
                scratch += 1
                sample_score -= self.scratch_extra_penalty

            if foul_info["FOUL_FIRST_HIT"]:
                foul_first_hit += 1
                sample_score -= self.first_hit_extra_penalty
            if foul_info["FOUL_NO_RAIL"]:
                foul_no_rail += 1
                sample_score -= self.no_rail_extra_penalty
            if foul_info["NO_HIT"]:
                no_hit += 1
                sample_score -= self.no_hit_extra_penalty

            scores.append(sample_score)
        
        mean_score = float(np.mean(scores)) if scores else -500.0
        if return_stats:
            return mean_score, {
                "samples": int(samples),
                "catastrophic": int(catastrophic),
                "scratch": int(scratch),
                "foul_first_hit": int(foul_first_hit),
                "foul_no_rail": int(foul_no_rail),
                "no_hit": int(no_hit),
                "illegal_eight": int(illegal_eight),
                "white_and_eight": int(white_and_eight),
            }
        return mean_score

    def _analyze_fouls(self, shot, last_state, targets):
        """快速检测本杆是否触发关键犯规（尽量对齐 PoolEnv 规则）"""
        new_pocketed = [
            bid for bid, b in shot.balls.items()
            if bid in last_state and b.state.s == 4 and last_state[bid].state.s != 4
        ]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        legal_eight = (len(targets) == 1 and targets[0] == "8")
        illegal_eight = eight_pocketed and (not legal_eight)
        white_and_eight = cue_pocketed and eight_pocketed

        # 首球接触判断（copy 自 agent.py/poolenv.py 的判定方式）
        first_contact_ball_id = None
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        foul_first_hit = False
        no_hit = False
        if first_contact_ball_id is None:
            no_hit = True
            if len(last_state) > 2 or targets != ['8']:
                foul_first_hit = True
        else:
            if first_contact_ball_id not in targets:
                foul_first_hit = True

        # 无碰库犯规（无进球+母球/首碰球均未碰库）
        cue_hit_cushion = False
        target_hit_cushion = False
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True
        foul_no_rail = (len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion))

        return {
            "CUE_POCKETED": cue_pocketed,
            "EIGHT_POCKETED": eight_pocketed,
            "ILLEGAL_EIGHT": illegal_eight,
            "WHITE_AND_EIGHT": white_and_eight,
            "FOUL_FIRST_HIT": foul_first_hit,
            "FOUL_NO_RAIL": foul_no_rail,
            "NO_HIT": no_hit,
        }

    def _pick_safer_action(self, refined_scores, balls, table, last_state_snapshot, prepared_targets, is_critical):
        """当最优动作存在致命风险时，从候选中挑更稳的（限制验证次数避免拖慢）"""
        if not refined_scores:
            return None
        samples_verify = self.samples_critical if is_critical else self.samples_final_verify
        checked = 0
        for _, candidate in refined_scores:
            verify_score, verify_stats = self._evaluate_action_fast(
                candidate, balls, table, last_state_snapshot, prepared_targets, samples_verify, return_stats=True
            )
            checked += 1
            if verify_stats["catastrophic"] == 0:
                return candidate
            if checked >= self.max_fallback_verifies:
                break
        return None

    def _black8_is_dangerous(self, balls, table, prepared_targets):
        """黑8在非清台阶段若离袋口很近，容易出现意外进袋（即时判负）。"""
        if len(prepared_targets) == 1 and prepared_targets[0] == "8":
            return False
        black = balls.get("8")
        if black is None or black.state.s == 4:
            return False
        black_pos = np.array(black.state.rvw[0][:2], dtype=float)
        min_dist = float("inf")
        for pocket in table.pockets.values():
            pocket_pos = np.array(pocket.center[:2], dtype=float)
            min_dist = min(min_dist, float(np.linalg.norm(black_pos - pocket_pos)))
        return min_dist < self.black_pocket_danger_dist
    
    def _calculate_strategic_bonus(self, shot, table, targets, last_state):
        """计算战略奖励（走位+防守）

        重要：即使本杆不进球，也要给“走位变好”的局面一定正向奖励，
        否则走位杆永远不会被选中。
        """
        if not self.enable_strategic_bonus:
            return 0.0

        cue_ball = shot.balls.get("cue")
        if cue_ball is None or cue_ball.state.s == 4:
            return -200.0

        bonus = 0.0
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)

        # 本杆是否打进己方球：如果继续出杆，对手威胁不重要（省大量计算）
        pocketed_own = [
            bid for bid, b in shot.balls.items()
            if bid in last_state and b.state.s == 4 and last_state[bid].state.s != 4 and bid in targets
        ]

        if self.enable_positional_bonus:
            # White ball far from pockets (lower scratch risk).
            min_pocket_dist = float("inf")
            for pocket in table.pockets.values():
                pocket_pos = np.array(pocket.center[:2], dtype=float)
                min_pocket_dist = min(min_pocket_dist, float(np.linalg.norm(cue_pos - pocket_pos)))
            bonus += 18.0 * float(np.clip((min_pocket_dist - 0.18) / 0.27, 0.0, 1.0))

            # White ball closer to table center (improve solvability).
            center = np.array([table.l / 2.0, table.w / 2.0], dtype=float)
            dist_center = float(np.linalg.norm(cue_pos - center))
            max_center_dist = float(np.linalg.norm(np.array([table.l, table.w], dtype=float) / 2.0))
            bonus += 10.0 * float(np.clip(1.0 - dist_center / max_center_dist, 0.0, 1.0))

        # 白球靠近下一目标球（进球后更重要；无进球时也有一定意义）
        remaining_positions = []
        for bid in targets:
            ball = shot.balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            remaining_positions.append(np.array(ball.state.rvw[0][:2], dtype=float))
        if self.enable_positional_bonus and remaining_positions:
            min_dist = float(min(np.linalg.norm(cue_pos - pos) for pos in remaining_positions))
            if min_dist < self.cue_next_ball_radius:
                bonus += self.cue_next_ball_weight * (1 - min_dist / self.cue_next_ball_radius)

        # 防守：只有在“未打进己方球（将换对手）”时才计算对手威胁
        if self.enable_defense_threat and (not pocketed_own):
            opponent_threat = self._estimate_opponent_pot_threat(shot, table)
            bonus -= self.opponent_pot_threat_weight * opponent_threat

        return float(bonus)

    def _estimate_opponent_pot_threat(self, shot, table):
        """粗略估计对手下一杆直接进球威胁（0~1）。

        直觉：如果白球到对手球直线路径通畅，且对手球到某袋口也通畅，并且角度不极端，
        则威胁高。该指标用于走位杆/防守时的排序，不追求物理精确。
        """
        cue = shot.balls.get("cue")
        if cue is None or cue.state.s == 4:
            return 0.0
        cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)

        enemy_ids = self._enemy_target_ids()
        pockets = list(table.pockets.values())
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in pockets]

        # 轻量近似：只看距离白球最近的少数对手球，并且每球只看最近的少数袋口
        enemy_infos = []
        for eid in enemy_ids:
            ball = shot.balls.get(eid)
            if ball is None or ball.state.s == 4:
                continue
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            enemy_infos.append((float(np.linalg.norm(ball_pos - cue_pos)), eid, ball_pos))
        enemy_infos.sort(key=lambda x: x[0])
        enemy_infos = enemy_infos[:2]

        best = 0.0
        for d_cb, eid, ball_pos in enemy_infos:
            # cue -> enemy 必须通畅（只做这一条阻挡检查，避免昂贵的 enemy->pocket 全检查）
            if self._is_path_blocked_fast(cue_pos, ball_pos, shot.balls, ignore_id=eid):
                continue

            cue_to_ball = ball_pos - cue_pos
            if d_cb < 1e-3:
                continue
            u_cb = cue_to_ball / d_cb

            # 最近两个袋口即可
            pocket_dists = [(float(np.linalg.norm(ball_pos - ppos)), ppos) for ppos in pocket_positions]
            pocket_dists.sort(key=lambda x: x[0])
            for d_bp, ppos in pocket_dists[:2]:
                ball_to_pocket = ppos - ball_pos
                if d_bp < 1e-3:
                    continue
                u_bp = ball_to_pocket / d_bp
                align = float(np.clip(np.dot(u_cb, u_bp), -1.0, 1.0))
                if align <= 0.25:
                    continue
                dist_factor = 1.0 / (1.0 + 0.8 * d_cb + 0.6 * d_bp)
                best = max(best, align * dist_factor)

        return float(np.clip(best * 6.0, 0.0, 1.0))
    
    # ========== CMA-ES 快速优化 ==========
    
    def _cma_es_optimize_fast(self, initial_action, balls, table, last_state, targets, samples):
        """快速 CMA-ES 优化（减少种群和代数，添加超时保护）"""
        if not CMA_AVAILABLE:
            return initial_action, self._evaluate_action_fast(
                initial_action, balls, table, last_state, targets, samples
            )
        initial_action = self._sanitize_action(initial_action)
        if initial_action is None:
            return self._random_action(), -500.0
        
        # 评估计数器，防止无限循环
        eval_count = [0]
        max_evals = self.cma_population_size * self.cma_generations  # 6*3=18次
        
        def neg_reward(x):
            eval_count[0] += 1
            if eval_count[0] > max_evals:
                # 超过最大评估次数，返回一个很差的分数强制停止
                return 1e6
            
            action = {
                'V0': float(np.clip(x[0], 0.5, 8.0)),
                'phi': float(x[1] % 360),
                'theta': float(np.clip(x[2], 0, 90)),
                'a': float(np.clip(x[3], -0.5, 0.5)),
                'b': float(np.clip(x[4], -0.5, 0.5))
            }
            score = self._evaluate_action_fast(
                action, balls, table, last_state, targets, samples
            )
            return -score
        
        try:
            x0 = [initial_action['V0'], initial_action['phi'], initial_action['theta'],
                  initial_action['a'], initial_action['b']]
            
            bounds = [(0.5, 8.0), (0, 360), (0, 90), (-0.5, 0.5), (-0.5, 0.5)]
            
            es = cma.CMAEvolutionStrategy(
                x0, self.cma_sigma,
                {
                    'bounds': [[b[0] for b in bounds], [b[1] for b in bounds]],
                    'popsize': self.cma_population_size,
                    'maxiter': self.cma_generations,
                    'verbose': -9,
                    'verb_filenameprefix': '',  # 禁用文件输出
                    'seed': int(self._rng.integers(1_000_000)),
                    'tolx': 1e-3,  # 添加收敛阈值，提前停止
                    'tolfun': 1e-2  # 函数值变化阈值
                }
            )
            
            es.optimize(neg_reward)
            best_x = es.result.xbest
            
            best_action = {
                'V0': float(np.clip(best_x[0], 0.5, 8.0)),
                'phi': float(best_x[1] % 360),
                'theta': float(np.clip(best_x[2], 0, 90)),
                'a': float(np.clip(best_x[3], -0.5, 0.5)),
                'b': float(np.clip(best_x[4], -0.5, 0.5))
            }
            
            best_score = -es.result.fbest
            return best_action, best_score
        
        except Exception as e:
            return initial_action, self._evaluate_action_fast(
                initial_action, balls, table, last_state, targets, samples
            )
    
    # ========== 安全球快速规划 ==========
    
    def _plan_positional_shot_fast(self, balls, table, last_state, targets):
        """快速走位杆规划：保证合法首碰，尽量把白球走到更好下一杆位置。"""
        cue_ball = balls.get("cue")
        if cue_ball is None or cue_ball.state.s == 4:
            return None, -500.0

        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        candidates = []

        # 1) 以“碰到合法目标球”为前提：瞄准最近目标球，轻推+小角偏转 + 轻跟/拉
        target_infos = []
        for bid in targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            target_infos.append((float(np.linalg.norm(pos - cue_pos)), bid, pos))
        target_infos.sort(key=lambda x: x[0])
        target_infos = target_infos[: self.positional_max_targets]

        angle_offsets = self.positional_angle_offsets
        speeds = self.positional_speeds
        spin_variants = self.positional_spin_variants
        for dist, _, pos in target_infos:
            direction = pos - cue_pos
            if np.linalg.norm(direction) < 1e-6:
                continue
            base_phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            for v0 in speeds:
                # 距离极近时减小力度，避免误进袋/乱飞
                tuned_v0 = float(np.clip(v0 + 0.2 * dist, 1.6, 3.2))
                for off in angle_offsets:
                    for a, b in spin_variants:
                        candidates.append({
                            "V0": tuned_v0,
                            "phi": float((base_phi + off) % 360),
                            "theta": 2.0,
                            "a": float(a),
                            "b": float(b),
                        })

        # 2) 少量随机角度补充（极端拥挤/无明显目标时兜底）
        for angle in np.linspace(0, 360, self.positional_random_angles, endpoint=False):
            V0 = float(self._rng.uniform(*self.safe_speed_bounds))
            candidates.append({
                "V0": V0,
                "phi": float(angle),
                "theta": 2.0,
                "a": 0.0,
                "b": 0.0,
            })
        
        best_action = None
        best_score = -500
        
        for candidate in candidates:
            if self._is_degenerate_params(candidate):
                continue
            score, st = self._evaluate_action_fast(
                candidate, balls, table, last_state, targets,
                self.safety_samples, return_stats=True
            )
            # 安全球：硬过滤（避免把“安全”打成直接判负/回滚送机会）
            if st.get("catastrophic", 0) > 0:
                continue
            if st.get("no_hit", 0) > 0:
                continue
            # 首球犯规在防守中尤其致命：对手得到干净局面
            if st.get("foul_first_hit", 0) > 0:
                continue
            if score > best_score:
                best_score = score
                best_action = candidate
                if best_score >= self.positional_early_stop_score:
                    break
        
        # 如果全被过滤，退回到“保证碰球”的兜底候选中挑一个
        if best_action is None:
            fallback = self._generate_contact_fallback_candidates(balls, table, targets, max_targets=2)
            for cand in fallback:
                score, st = self._evaluate_action_fast(
                    cand, balls, table, last_state, targets,
                    max(1, self.safety_samples), return_stats=True
                )
                if st.get("catastrophic", 0) > 0 or st.get("no_hit", 0) > 0 or st.get("foul_first_hit", 0) > 0:
                    continue
                if score > best_score:
                    best_score = score
                    best_action = cand

        return best_action, best_score

    # 向后兼容：旧名仍可用
    def _plan_safety_shot_fast(self, balls, table, last_state, targets):
        return self._plan_positional_shot_fast(balls, table, last_state, targets)
    
    # ========== 辅助函数 ==========
    
    def _prepare_targets(self, balls, my_targets):
        """准备目标球列表"""
        if my_targets is None:
            return []
        
        # 推断球型
        if self.my_target_type is None:
            if any(t in self.SOLID_IDS for t in my_targets):
                self.my_target_type = 'solid'
            elif any(t in self.STRIPE_IDS for t in my_targets):
                self.my_target_type = 'stripe'
        
        valid = [t for t in my_targets if t in balls and balls[t].state.s != 4]
        if not valid:
            return ['8']
        return valid
    
    def _enemy_target_ids(self):
        """推断对手目标球"""
        if self.my_target_type == 'solid':
            return self.STRIPE_IDS
        elif self.my_target_type == 'stripe':
            return self.SOLID_IDS
        return self.SOLID_IDS + self.STRIPE_IDS
    
    def _sample_noisy_params(self, V0, phi, theta, a, b):
        """采样噪声参数"""
        return {
            'V0': float(V0 + self._rng.normal(0, 0.1)),
            'phi': float(phi + self._rng.normal(0, 0.1)),
            'theta': float(theta + self._rng.normal(0, 0.1)),
            'a': float(a + self._rng.normal(0, 0.003)),
            'b': float(b + self._rng.normal(0, 0.003))
        }
    
    def _is_degenerate_params(self, params):
        """检查参数是否退化"""
        return (params['V0'] < 0.3 or params['V0'] > 10.0 or
                params['theta'] < 0 or params['theta'] > 90 or
                abs(params['a']) > 0.6 or abs(params['b']) > 0.6)
    
    def _random_action(self):
        """随机动作"""
        return {
            'V0': float(self._rng.uniform(2.0, 5.0)),
            'phi': float(self._rng.uniform(0, 360)),
            'theta': 2.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def _conservative_action(self, balls, table):
        """保守动作（轻推白球）"""
        cue_pos = balls['cue'].state.rvw[0][:2]
        center = np.array([table.l / 2, table.w / 2])
        to_center = center - cue_pos
        phi = math.degrees(math.atan2(to_center[1], to_center[0])) % 360
        
        return {
            'V0': 2.0,
            'phi': float(phi),
            'theta': 2.0,
            'a': 0.0,
            'b': 0.0
        }
    
    def _shot_alarm_handler(self, signum, frame):
        """超时处理"""
        raise TimeoutError("单杆搜索超时")


# 为了兼容性，创建别名
NewAgent = OptimizedNewAgent
