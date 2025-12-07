"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
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
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )

            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(BasicAgent):
    """增强版 Agent：噪声鲁棒 + 局面控制"""
    
    SOLID_IDS = tuple(str(i) for i in range(1, 8))
    STRIPE_IDS = tuple(str(i) for i in range(9, 16))
    
    def __init__(self):
        super().__init__()
        # 搜索与评估配置
        self.INITIAL_SEARCH = 14
        self.OPT_SEARCH = 8
        self.robust_samples = 4
        self.enable_noise = False  # 使用自定义噪声采样
        
        # 策略超参数
        self.cue_next_ball_radius = 1.4
        self.cue_next_ball_weight = 22.0
        self.enemy_threat_radius = 1.0
        self.enemy_distance_weight = 18.0
        self.eight_guard_radius = 0.13
        self.eight_guard_weight = 45.0
        self.safety_trigger_score = 30.0
        self.safety_prefer_margin = 8.0
        self.safe_speed_bounds = (2.4, 3.2)
        self.no_rail_penalty = 80.0
        self.white_scratch_penalty = 220.0
        self.illegal_black_penalty = 300.0
        self.cue_edge_margin = 0.18
        self.cue_edge_weight = 20.0
        self.shot_timeout = 5 * 60
        self.fallback_score_threshold = 30.0
        self.attack_success_weight = 70.0
        self.attack_cue_bonus_weight = 20.0
        
        # 记录自己是实心还是条纹，方便推断对手
        self.my_target_type = None  # 'solid' / 'stripe'
        
        print("NewAgent (robust + strategic) 已初始化。")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """噪声鲁棒 + 局面控制的决策方法"""
        if balls is None or table is None:
            print("[NewAgent] 缺少关键观测，使用随机动作。")
            return self._random_action()
        
        try:
            prepared_targets = self._prepare_targets(balls, my_targets)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            def reward_fn_wrapper(V0, phi, theta, a, b):
                return self._evaluate_action(
                    V0=V0,
                    phi=phi,
                    theta=theta,
                    a=a,
                    b=b,
                    balls=balls,
                    table=table,
                    last_state_snapshot=last_state_snapshot,
                    player_targets=prepared_targets
                )
            
            print(f"[NewAgent] 搜索击球方案 (targets={prepared_targets}) ...")
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            timed_out = False
            prev_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, self._shot_alarm_handler)
                signal.alarm(self.shot_timeout)
                optimizer.maximize(
                    init_points=self.INITIAL_SEARCH,
                    n_iter=self.OPT_SEARCH
                )
            except TimeoutError:
                timed_out = True
                print(f"[NewAgent] 由于耗时超过{self.shot_timeout}s，提前终止搜索。")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)

            best_result = optimizer.max
            best_score = best_result['target']
            best_params = best_result['params']
            if best_score is None or np.isnan(best_score):
                print("[NewAgent] 搜索失败，返回随机动作。")
                return self._random_action()
            
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b'])
            }
            print(f"[NewAgent] 最佳动作得分 {best_score:.2f}: "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            # 评估安全球候选，必要时切换
            allow_safety = self._allow_safety_play(prepared_targets)
            safe_action = None
            safe_score = -500
            if allow_safety:
                safety_candidates = self._plan_safety_shot(balls, table)
                safe_action, safe_score = self._select_best_candidate(
                    safety_candidates, reward_fn_wrapper
                )
                if safe_action is not None:
                    print(f"[NewAgent] 安全球候选得分 {safe_score:.2f}")
            safe_validation = None
            if safe_action is not None:
                safe_validation = self._simulate_action_outcome(
                    safe_action, balls, table, prepared_targets, last_state_snapshot
                )
            
            if safe_action is not None:
                prefer_safe = (
                    best_score < self.safety_trigger_score
                    and safe_score >= best_score + self.safety_prefer_margin
                )
                force_safe = safe_score >= best_score + self.safety_prefer_margin * 2.0
                if prefer_safe or force_safe:
                    safe_ok = safe_validation is not None and safe_validation[0]
                    safe_danger = safe_validation and self._action_is_dangerous(
                        safe_validation[1], prepared_targets
                    )
                    if safe_ok and not safe_danger:
                        print("[NewAgent] 选择安全球方案。")
                        return safe_action
                    print("[NewAgent] 安全球候选验证失败，继续尝试进攻。")

            fallback_needed = timed_out or best_score < self.fallback_score_threshold
            if fallback_needed:
                fallback_action = self._prepare_fallback_action(
                    safe_action=safe_action,
                    safe_validation=safe_validation,
                    balls=balls,
                    table=table,
                    player_targets=prepared_targets,
                    last_state_snapshot=last_state_snapshot
                )
            if fallback_action is not None:
                print("[NewAgent] 使用启发式保守策略替代原始搜索结果。")
                return fallback_action

            attack_action, attack_score = self._hierarchical_attack_search(
                balls=balls,
                table=table,
                player_targets=prepared_targets,
                last_state_snapshot=last_state_snapshot,
                current_score=best_score
            )
            if attack_action is not None and attack_score > best_score + 5:
                print("[NewAgent] 启发式进攻获得更高评分，采用该方案。")
                return attack_action

            if best_score < 10:
                print("[NewAgent] 得分过低，使用随机动作兜底。")
                return self._random_action()
            action_ok, action_info = self._simulate_action_outcome(
                action, balls, table, prepared_targets, last_state_snapshot
            )
            action_danger = action_info if action_info else {}
            if (not action_ok) or self._action_is_dangerous(action_danger, prepared_targets):
                reason = []
                if action_info:
                    if action_info.get('WHITE_BALL_INTO_POCKET'):
                        reason.append("白球落袋")
                    if action_info.get('ILLEGAL_BLACK'):
                        reason.append("非法黑8")
                    if action_info.get('NO_POCKET_NO_RAIL'):
                        reason.append("无进球且未碰库")
                msg = "、".join(reason) if reason else "模拟失败"
                print(f"[NewAgent] 决策被否决：{msg}，尝试安全兜底。")
                fallback = self._fallback_safe_action(
                    safe_action=safe_action,
                    safe_validation=safe_validation,
                    balls=balls,
                    table=table,
                    player_targets=prepared_targets,
                    last_state_snapshot=last_state_snapshot,
                    scorer=reward_fn_wrapper
                )
                if fallback is not None:
                    return fallback
                print("[NewAgent] 无可行安全球，使用随机动作兜底。")
                return self._random_action()
            return action
        
        except Exception as exc:
            print(f"[NewAgent] 决策错误，改用随机动作：{exc}")
            import traceback
            traceback.print_exc()
            return self._random_action()
    
    def _prepare_targets(self, balls, my_targets):
        """规范化目标球列表，并更新己方球型"""
        if not my_targets:
            return ['8']
        valid = [bid for bid in my_targets if bid in balls]
        if not valid:
            return ['8']
        if not (len(valid) == 1 and valid[0] == '8'):
            self._update_target_type(valid)
            remaining = [bid for bid in valid if balls[bid].state.s != 4]
            if remaining:
                return remaining
        return ['8']
    
    def _allow_safety_play(self, prepared_targets):
        """判断当前是否允许执行保守安全球"""
        if not prepared_targets:
            return False
        if len(prepared_targets) <= 3:
            return False
        if len(prepared_targets) == 1 and prepared_targets[0] == '8':
            return False
        return True
    
    def _update_target_type(self, target_ids):
        """根据当前可见目标球推断我方球型"""
        if not target_ids or (len(target_ids) == 1 and target_ids[0] == '8'):
            return
        if any(tid in self.SOLID_IDS for tid in target_ids):
            self.my_target_type = 'solid'
        elif any(tid in self.STRIPE_IDS for tid in target_ids):
            self.my_target_type = 'stripe'

    def _shot_alarm_handler(self, signum, frame):
        """信号处理，达到单杆时间限制时抛出 TimeoutError。"""
        raise TimeoutError("单杆搜索超时")
    
    def _evaluate_action(self, V0, phi, theta, a, b, balls, table, last_state_snapshot, player_targets, robust_samples=None):
        """带噪声 Monte Carlo 的动作评估"""
        samples = int(robust_samples) if robust_samples is not None else self.robust_samples
        scores = []
        for _ in range(samples):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            noisy_params = self._sample_noisy_params(V0, phi, theta, a, b)
            if self._is_degenerate_params(noisy_params):
                # Skip samples that would trigger unstable root solving.
                return -500.0
            shot.cue.set_state(**noisy_params)
            try:
                pt.simulate(shot, inplace=True)
            except Exception:
                return -500.0
            
            pocketed = self._pocketed_since_last(shot, last_state_snapshot)
            base_score = analyze_shot_for_reward(
                shot=shot,
                last_state=last_state_snapshot,
                player_targets=player_targets
            )
            strategic_bonus = self._strategic_bonus(
                shot=shot,
                table=sim_table,
                player_targets=player_targets
            )
            rail_bonus = self._rail_awareness_bonus(shot, pocketed)
            cue_edge_bonus = self._cue_edge_bonus(shot, table)
            scratch_penalty = self._white_ball_penalty(shot, pocketed)
            black_penalty = self._illegal_black_penalty(pocketed, player_targets)
            candidate_score = (
                base_score + strategic_bonus + rail_bonus + cue_edge_bonus + scratch_penalty + black_penalty
            )
            if not np.isfinite(candidate_score):
                continue
            scores.append(candidate_score)
        
        if not scores:
            return -500.0
        return float(np.mean(scores))
    
    def _sample_noisy_params(self, V0, phi, theta, a, b):
        """按照评测噪声标准差采样动作"""
        noisy = {
            'V0': V0 + np.random.normal(0, self.noise_std['V0']),
            'phi': phi + np.random.normal(0, self.noise_std['phi']),
            'theta': theta + np.random.normal(0, self.noise_std['theta']),
            'a': a + np.random.normal(0, self.noise_std['a']),
            'b': b + np.random.normal(0, self.noise_std['b'])
        }
        noisy['V0'] = float(np.clip(noisy['V0'], *self.pbounds['V0']))
        noisy['phi'] = float(noisy['phi'] % 360)
        noisy['theta'] = float(np.clip(noisy['theta'], *self.pbounds['theta']))
        noisy['a'] = float(np.clip(noisy['a'], *self.pbounds['a']))
        noisy['b'] = float(np.clip(noisy['b'], *self.pbounds['b']))
        return noisy

    def _is_degenerate_params(self, params):
        """快速判断动作是否会造成数值退化/卡住。"""
        theta = params['theta']
        V0 = params['V0']
        # 几乎平行于库边的高能量平射常触发 divide-by-zero
        if theta < 1.0 and V0 > 7.0:
            return True
        if theta < 0.5 and V0 > 6.5:
            return True
        # 过于垂直的高能量也容易打回自己，影响 root solver
        if theta > 88.0 and V0 > 6.5:
            return True
        return False
    
    def _strategic_bonus(self, shot, table, player_targets):
        """根据局面优劣增加额外奖励"""
        cue_pos = np.array(shot.balls['cue'].state.rvw[0][:2], dtype=float)
        bonus = 0.0
        
        remaining_targets = self._remaining_targets_after_shot(shot, player_targets)
        own_positions = [
            np.array(shot.balls[bid].state.rvw[0][:2], dtype=float)
            for bid in remaining_targets
            if bid in shot.balls and shot.balls[bid].state.s != 4
        ]
        if own_positions:
            min_dist = min(np.linalg.norm(cue_pos - pos) for pos in own_positions)
            bonus += max(0.0, self.cue_next_ball_radius - min_dist) * self.cue_next_ball_weight
        
        enemy_positions = [
            np.array(shot.balls[bid].state.rvw[0][:2], dtype=float)
            for bid in self._enemy_target_ids()
            if bid in shot.balls and shot.balls[bid].state.s != 4
        ]
        if enemy_positions:
            near_enemy = min(np.linalg.norm(cue_pos - pos) for pos in enemy_positions)
            bonus -= max(0.0, self.enemy_threat_radius - near_enemy) * self.enemy_distance_weight
        
        if not (len(player_targets) == 1 and player_targets[0] == '8'):
            black_ball = shot.balls.get('8')
            if black_ball is not None and black_ball.state.s != 4:
                black_pos = np.array(black_ball.state.rvw[0][:2], dtype=float)
                min_pocket_dist = min(
                    np.linalg.norm(black_pos - np.array(pocket.center[:2]))
                    for pocket in table.pockets.values()
                )
                if min_pocket_dist < self.eight_guard_radius:
                    bonus -= (self.eight_guard_radius - min_pocket_dist) * self.eight_guard_weight
        
        return bonus
    
    def _pocketed_since_last(self, shot, last_state):
        """返回相对于上一杆新进袋的球"""
        pocketed = []
        for bid, ball in shot.balls.items():
            if bid not in last_state:
                continue
            if ball.state.s == 4 and last_state[bid].state.s != 4:
                pocketed.append(bid)
        return pocketed
    
    def _rail_awareness_bonus(self, shot, pocketed_ids):
        """鼓励至少碰库，避免 NO_POCKET_NO_RAIL"""
        if pocketed_ids:
            return 0.0
        for event in shot.events:
            et = str(event.event_type).lower()
            if 'cushion' in et:
                return 0.0
        return -self.no_rail_penalty
    
    def _white_ball_penalty(self, shot, pocketed_ids):
        """白球落袋时的额外惩罚"""
        if 'cue' in pocketed_ids:
            return -self.white_scratch_penalty
        for event in shot.events:
            et = str(event.event_type).lower()
            if 'pocket' in et and any(i == 'cue' for i in getattr(event, 'ids', [])):
                return -self.white_scratch_penalty
        return 0.0
    
    def _illegal_black_penalty(self, pocketed_ids, player_targets):
        """非法打进黑8时施加额外惩罚"""
        if '8' not in pocketed_ids:
            return 0.0
        legal = len(player_targets) == 1 and player_targets[0] == '8'
        if legal:
            return 0.0
        return -self.illegal_black_penalty

    def _remaining_targets_after_shot(self, shot, player_targets):
        """计算该杆结束后仍需击打的球"""
        valid = [bid for bid in player_targets if bid in shot.balls]
        if not valid:
            return ['8']
        remaining = [bid for bid in valid if shot.balls[bid].state.s != 4]
        if remaining:
            return remaining
        return ['8']
    
    def _enemy_target_ids(self):
        """根据记录的我方球型推断对方球集合"""
        if self.my_target_type == 'solid':
            return self.STRIPE_IDS
        if self.my_target_type == 'stripe':
            return self.SOLID_IDS
        return self.SOLID_IDS + self.STRIPE_IDS
    
    def _cue_edge_bonus(self, shot, table):
        """奖励母球贴边，限制对手直线进攻"""
        cue_pos = np.array(shot.balls['cue'].state.rvw[0][:2], dtype=float)
        distances = [
            cue_pos[0],
            table.l - cue_pos[0],
            cue_pos[1],
            table.w - cue_pos[1]
        ]
        min_dist = min(distances)
        if min_dist < self.cue_edge_margin:
            return (self.cue_edge_margin - min_dist) * self.cue_edge_weight
        return 0.0
    
    def _plan_safety_shot(self, balls, table):
        """生成多个安全球候选（确保有明显的碰库或藏球倾向）"""
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        margin = 0.07
        clamp_x = np.clip(cue_pos[0], margin, table.l - margin)
        clamp_y = np.clip(cue_pos[1], margin, table.w - margin)
        cushion_targets = [
            np.array([margin, clamp_y]),
            np.array([table.l - margin, clamp_y]),
            np.array([clamp_x, margin]),
            np.array([clamp_x, table.w - margin])
        ]
        two_bank_dirs = [
            math.degrees(math.atan2(clamp_y - cue_pos[1], (table.l - margin) - cue_pos[0])),
            math.degrees(math.atan2((table.w - margin) - cue_pos[1], clamp_x - cue_pos[0])),
            math.degrees(math.atan2(margin - cue_pos[1], margin - cue_pos[0])),
            math.degrees(math.atan2((table.w - margin) - cue_pos[1], (table.l - margin) - cue_pos[0]))
        ]
        candidates = []
        base_speeds = np.linspace(self.safe_speed_bounds[0], self.safe_speed_bounds[1], num=3)
        spin_options = [(-0.18, -0.08), (0.18, 0.06), (0.0, -0.12)]
        for target_point in cushion_targets:
            direction = target_point - cue_pos
            if np.linalg.norm(direction) < 1e-6:
                continue
            phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            for speed in base_speeds:
                for spin in spin_options:
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 2.0,
                        'a': float(spin[0]),
                        'b': float(spin[1])
                    })
        for phi in two_bank_dirs:
            for speed in base_speeds:
                candidates.append({
                    'V0': float(speed),
                    'phi': float(phi % 360),
                    'theta': 3.0,
                    'a': 0.0,
                    'b': -0.05
                })
        nearest_own = self._nearest_own_ball(balls)
        if nearest_own is not None:
            own_vec = nearest_own - cue_pos
            if np.linalg.norm(own_vec) > 1e-3:
                phi = math.degrees(math.atan2(own_vec[1], own_vec[0])) % 360
                candidates.append({
                    'V0': float(self.safe_speed_bounds[0]),
                    'phi': float(phi),
                    'theta': 0.5,
                    'a': -0.2,
                    'b': -0.08
                })
        return candidates
    
    def _nearest_own_ball(self, balls):
        """返回最近己方球的位置"""
        if self.my_target_type is None:
            return None
        ids = self.SOLID_IDS if self.my_target_type == 'solid' else self.STRIPE_IDS
        cue_pos = np.array(balls['cue'].state.rvw[0][:2], dtype=float)
        best = None
        best_dist = None
        for bid in ids:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            dist = np.linalg.norm(pos - cue_pos)
            if best is None or dist < best_dist:
                best = pos
                best_dist = dist
        return best
    
    def _select_best_candidate(self, candidates, scorer):
        """在候选动作中选取得分最高者"""
        best_score = -np.inf
        best_action = None
        for action in candidates:
            try:
                score = scorer(**action)
            except Exception:
                score = -500
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, best_score

    def _prepare_fallback_action(self, *, safe_action, safe_validation, balls, table,
                                 player_targets, last_state_snapshot):
        """在超时或低分时返回可用安全动作"""
        if safe_action is not None and safe_validation is not None:
            safe_ok = safe_validation[0] and not self._action_is_dangerous(
                safe_validation[1], player_targets
            )
            if safe_ok:
                return safe_action
        heuristic_action = self._heuristic_safety_action(
            balls=balls,
            table=table,
            player_targets=player_targets,
            last_state_snapshot=last_state_snapshot
        )
        return heuristic_action

    def _heuristic_safety_action(self, *, balls, table, player_targets, last_state_snapshot):
        """快速生成保守候选并验证，避免长时间搜索"""
        candidates = self._plan_safety_shot(balls, table)
        for action in candidates:
            valid, info = self._simulate_action_outcome(
                action, balls, table, player_targets, last_state_snapshot
            )
            if valid and not self._action_is_dangerous(info, player_targets):
                return action
        return None

    def _hierarchical_attack_search(self, *, balls, table, player_targets,
                                    last_state_snapshot, current_score):
        """生成启发式进攻候选并在本地优化层面选出最优解。"""
        candidates = self._generate_attack_candidates(balls, table, player_targets)
        best_action = None
        best_score = current_score
        for action in candidates:
            if self._is_degenerate_params(action):
                continue
            robust_score = self._evaluate_action(
                action['V0'], action['phi'], action['theta'], action['a'], action['b'],
                balls, table, last_state_snapshot, player_targets, robust_samples=2
            )
            if robust_score < -400:
                continue
            success_rate = self._estimate_success_rate(
                action, balls, table, player_targets, last_state_snapshot, trials=4
            )
            cue_bonus = self._cue_control_bonus(balls, player_targets)
            combined = robust_score + success_rate * self.attack_success_weight + cue_bonus
            if combined > best_score:
                best_score = combined
                best_action = action
        return best_action, best_score

    def _generate_attack_candidates(self, balls, table, player_targets, max_candidates=12):
        """基于几何启发式（ghost-ball / pocket）生成进攻初始种子。"""
        candidates = []
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return candidates
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        pockets = list(table.pockets.values())
        for bid in player_targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            distance = np.linalg.norm(ball_pos - cue_pos)
            speed = float(np.clip(distance * 0.6 + 1.8, 1.5, 6.0))
            aim_angle = math.atan2(ball_pos[1] - cue_pos[1], ball_pos[0] - cue_pos[0])
            for pocket in pockets:
                pocket_pos = np.array(pocket.center[:2], dtype=float)
                if np.linalg.norm(ball_pos - pocket_pos) < 1e-3:
                    continue
                pocket_angle = math.atan2(pocket_pos[1] - ball_pos[1], pocket_pos[0] - ball_pos[0])
                adjusted = aim_angle + 0.25 * self._angle_diff(pocket_angle, aim_angle)
                phi = math.degrees(adjusted) % 360
                theta = float(np.clip(5.0 + np.random.uniform(-1.0, 1.0), 2.0, 8.0))
                spin_a = 0.0
                spin_b = 0.0
                candidates.append({
                    'V0': speed,
                    'phi': phi,
                    'theta': theta,
                    'a': spin_a,
                    'b': spin_b
                })
                if len(candidates) >= max_candidates:
                    return candidates
        return candidates

    def _estimate_success_rate(self, action, balls, table, player_targets, last_state_snapshot, trials=4):
        """Monte Carlo 估计动作的实战进球概率（带噪声）。"""
        hits = 0
        for _ in range(trials):
            noisy = self._sample_noisy_params(
                action['V0'], action['phi'], action['theta'], action['a'], action['b']
            )
            valid, info = self._simulate_action_outcome(
                noisy, balls, table, player_targets, last_state_snapshot
            )
            if valid and info.get('ME_INTO_POCKET'):
                hits += 1
        return hits / trials if trials else 0.0

    def _cue_control_bonus(self, balls, player_targets):
        """根据白球与近期目标球的距离评估布局价值（距离越小越好）。"""
        cue_ball = balls.get('cue')
        if cue_ball is None or not player_targets:
            return 0.0
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        targets = [
            np.array(balls[bid].state.rvw[0][:2], dtype=float)
            for bid in player_targets
            if bid in balls and balls[bid].state.s != 4
        ]
        if not targets:
            return 0.0
        min_dist = min(np.linalg.norm(cue_pos - pos) for pos in targets)
        reward = max(0.0, self.cue_next_ball_radius - min_dist)
        return reward * self.attack_cue_bonus_weight

    def _angle_diff(self, target, source):
        """返回两个弧度角度的最小差值（-pi ~ pi）。"""
        delta = target - source
        return (delta + math.pi) % (2 * math.pi) - math.pi

    def _fallback_safe_action(self, *, safe_action, safe_validation, balls, table,
                              player_targets, last_state_snapshot, scorer):
        """当主决策危险时尝试使用（或重新搜索）安全球"""
        if safe_action is not None:
            validation = safe_validation
            if validation is None:
                validation = self._simulate_action_outcome(
                    safe_action, balls, table, player_targets, last_state_snapshot
                )
            if validation[0] and not self._action_is_dangerous(validation[1], player_targets):
                print("[NewAgent] fallback: 使用先前评估的安全球。")
                return safe_action
        candidates = self._plan_safety_shot(balls, table)
        if not candidates:
            return None
        alt_action, _ = self._select_best_candidate(candidates, scorer)
        if alt_action is None:
            return None
        valid, info = self._simulate_action_outcome(
            alt_action, balls, table, player_targets, last_state_snapshot
        )
        if valid and not self._action_is_dangerous(info, player_targets):
            print("[NewAgent] fallback: 重新规划安全球成功。")
            return alt_action
        return None

    def _simulate_action_outcome(self, action, balls, table, player_targets, last_state_snapshot):
        """单次模拟，用于验证动作是否存在危险（白球、黑8、未碰库等）"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        try:
            shot.cue.set_state(**action)
            pt.simulate(shot, inplace=True)
        except Exception:
            return False, {'SIM_FAIL': True}
        pocketed = self._pocketed_since_last(shot, last_state_snapshot)
        info = {}
        if pocketed:
            own = [bid for bid in pocketed if bid in player_targets]
            enemy = [
                bid for bid in pocketed
                if bid not in player_targets and bid not in ['cue']
            ]
            if own:
                info['ME_INTO_POCKET'] = own
            if enemy:
                info['ENEMY_INTO_POCKET'] = enemy
        if 'cue' in pocketed:
            info['WHITE_BALL_INTO_POCKET'] = True
        if '8' in pocketed:
            info['BLACK_BALL_INTO_POCKET'] = True
            legal = len(player_targets) == 1 and player_targets[0] == '8'
            if not legal:
                info['ILLEGAL_BLACK'] = True
        non_white_pocket = [b for b in pocketed if b != 'cue']
        if not non_white_pocket and not self._shot_hit_cushion(shot):
            info['NO_POCKET_NO_RAIL'] = True
        return True, info

    def _shot_hit_cushion(self, shot):
        """检查本杆是否有任何球触库"""
        for event in shot.events:
            et = str(event.event_type).lower()
            if 'cushion' in et:
                return True
        return False

    def _action_is_dangerous(self, info, player_targets):
        """根据 step_info 判断动作是否危险"""
        if not info:
            return False
        if info.get('WHITE_BALL_INTO_POCKET'):
            return True
        if info.get('ILLEGAL_BLACK'):
            return True
        if info.get('NO_POCKET_NO_RAIL'):
            return True
        return False
