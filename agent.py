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
    """增强版 Agent：Ghost Ball 几何瞄准 + CMA-ES 局部优化 + 噪声鲁棒"""
    
    SOLID_IDS = tuple(str(i) for i in range(1, 8))
    STRIPE_IDS = tuple(str(i) for i in range(9, 16))
    BALL_RADIUS = 0.028575  # 标准台球半径 (米)
    
    def __init__(self):
        super().__init__()
        # 搜索与评估配置
        self.INITIAL_SEARCH = 18
        self.OPT_SEARCH = 12
        self.robust_samples = 6  # 增加采样次数提高鲁棒性
        self.enable_noise = False  # 使用自定义噪声采样
        
        # CMA-ES 配置
        self.use_cma_es = True
        self.cma_population_size = 12
        self.cma_generations = 8
        self.cma_sigma = 0.3  # 初始步长
        
        # 策略超参数 - 调优后的权重
        self.cue_next_ball_radius = 1.2
        self.cue_next_ball_weight = 30.0  # 增加走位权重
        self.enemy_threat_radius = 0.8
        self.enemy_distance_weight = 15.0
        self.eight_guard_radius = 0.15
        self.eight_guard_weight = 50.0
        self.safety_trigger_score = 25.0  # 降低阈值，更早考虑防守
        self.safety_prefer_margin = 10.0
        self.safe_speed_bounds = (1.8, 2.8)  # 降低安全球速度
        self.no_rail_penalty = 100.0
        self.white_scratch_penalty = 250.0
        self.illegal_black_penalty = 350.0
        self.cue_edge_margin = 0.15
        self.cue_edge_weight = 15.0
        self.shot_timeout = 5 * 60
        self.fallback_score_threshold = 25.0
        self.attack_success_weight = 80.0
        self.attack_cue_bonus_weight = 25.0
        
        # 进球概率阈值
        self.min_pocket_probability = 0.4  # 低于此概率考虑防守
        self.high_confidence_threshold = 0.7  # 高于此概率优先进攻
        
        # 记录自己是实心还是条纹，方便推断对手
        self.my_target_type = None  # 'solid' / 'stripe'
        
        print("NewAgent (Ghost Ball + CMA-ES + robust) 已初始化。")
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Ghost Ball 几何瞄准 + CMA-ES 局部优化 + 噪声鲁棒的决策方法"""
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
            
            # Step 1: 使用 Ghost Ball 几何方法生成高质量初始种子
            ghost_ball_candidates = self._generate_ghost_ball_candidates(
                balls, table, prepared_targets
            )
            print(f"[NewAgent] Ghost Ball 生成了 {len(ghost_ball_candidates)} 个候选动作")
            
            # Step 2: 评估所有候选并选出最佳
            best_action = None
            best_score = -500
            
            # 评估 Ghost Ball 候选
            for candidate in ghost_ball_candidates:
                if self._is_degenerate_params(candidate):
                    continue
                score = reward_fn_wrapper(**candidate)
                if score > best_score:
                    best_score = score
                    best_action = candidate
            
            # Step 3: 使用 CMA-ES 对最佳候选进行局部优化
            if best_action is not None and self.use_cma_es and best_score > -100:
                optimized_action, optimized_score = self._cma_es_optimize(
                    initial_action=best_action,
                    reward_fn=reward_fn_wrapper,
                    balls=balls,
                    table=table
                )
                if optimized_score > best_score:
                    best_action = optimized_action
                    best_score = optimized_score
                    print(f"[NewAgent] CMA-ES 优化后得分提升至 {best_score:.2f}")
            
            # Step 4: 如果 Ghost Ball + CMA-ES 效果不好，回退到贝叶斯优化
            if best_score < self.fallback_score_threshold:
                print("[NewAgent] Ghost Ball 方案得分较低，尝试贝叶斯优化...")
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

                bayes_result = optimizer.max
                bayes_score = bayes_result['target']
                bayes_params = bayes_result['params']
                if bayes_score is not None and not np.isnan(bayes_score) and bayes_score > best_score:
                    best_score = bayes_score
                    best_action = {
                        'V0': float(bayes_params['V0']),
                        'phi': float(bayes_params['phi']),
                        'theta': float(bayes_params['theta']),
                        'a': float(bayes_params['a']),
                        'b': float(bayes_params['b'])
                    }
                    print(f"[NewAgent] 贝叶斯优化得分 {best_score:.2f}")
            
            if best_action is None or best_score < -400:
                print("[NewAgent] 搜索失败，返回随机动作。")
                return self._random_action()
            
            print(f"[NewAgent] 最佳动作得分 {best_score:.2f}: "
                  f"V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, "
                  f"θ={best_action['theta']:.2f}, a={best_action['a']:.3f}, b={best_action['b']:.3f}")
            
            # Step 5: 评估进球概率，决定是否防守
            pocket_probability = self._estimate_pocket_probability(
                best_action, balls, table, prepared_targets, last_state_snapshot
            )
            print(f"[NewAgent] 预估进球概率: {pocket_probability:.1%}")
            
            # Step 6: 评估安全球候选
            allow_safety = self._allow_safety_play(prepared_targets)
            safe_action = None
            safe_score = -500
            if allow_safety or pocket_probability < self.min_pocket_probability:
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
            
            # Step 7: 决策：进攻 vs 防守
            if safe_action is not None:
                # 进球概率低时优先防守
                prefer_safe_low_prob = (
                    pocket_probability < self.min_pocket_probability
                    and safe_score > -50
                )
                prefer_safe = (
                    best_score < self.safety_trigger_score
                    and safe_score >= best_score + self.safety_prefer_margin
                )
                force_safe = safe_score >= best_score + self.safety_prefer_margin * 2.0
                
                if prefer_safe_low_prob or prefer_safe or force_safe:
                    safe_ok = safe_validation is not None and safe_validation[0]
                    safe_danger = safe_validation and self._action_is_dangerous(
                        safe_validation[1], prepared_targets
                    )
                    if safe_ok and not safe_danger:
                        print("[NewAgent] 选择安全球方案。")
                        return safe_action
                    print("[NewAgent] 安全球候选验证失败，继续尝试进攻。")

            # Step 8: 最终验证
            if best_score < 10:
                print("[NewAgent] 得分过低，使用随机动作兜底。")
                return self._random_action()
            
            action_ok, action_info = self._simulate_action_outcome(
                best_action, balls, table, prepared_targets, last_state_snapshot
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
            
            return best_action
        
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

    def _generate_ghost_ball_candidates(self, balls, table, player_targets):
        """使用 Ghost Ball 方法生成高质量的击球候选
        
        Ghost Ball 原理：
        - 计算目标球到袋口的方向向量
        - 在目标球后方放置一个"幽灵球"（与目标球相切）
        - 瞄准幽灵球中心即可将目标球打入袋口
        """
        candidates = []
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return candidates
        
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        pockets = list(table.pockets.values())
        
        for bid in player_targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            
            for pocket in pockets:
                pocket_pos = np.array(pocket.center[:2], dtype=float)
                
                # 计算目标球到袋口的方向
                ball_to_pocket = pocket_pos - ball_pos
                dist_to_pocket = np.linalg.norm(ball_to_pocket)
                if dist_to_pocket < 1e-3:
                    continue
                
                ball_to_pocket_unit = ball_to_pocket / dist_to_pocket
                
                # Ghost Ball 位置：在目标球后方，距离为两球直径
                ghost_ball_pos = ball_pos - ball_to_pocket_unit * (2 * self.BALL_RADIUS)
                
                # 检查路径是否被阻挡
                if self._is_path_blocked(cue_pos, ghost_ball_pos, balls, bid):
                    continue
                
                # 检查目标球到袋口路径是否被阻挡
                if self._is_path_blocked(ball_pos, pocket_pos, balls, bid):
                    continue
                
                # 计算瞄准角度
                cue_to_ghost = ghost_ball_pos - cue_pos
                dist_to_ghost = np.linalg.norm(cue_to_ghost)
                if dist_to_ghost < 1e-3:
                    continue
                
                phi = math.degrees(math.atan2(cue_to_ghost[1], cue_to_ghost[0])) % 360
                
                # 根据距离计算合适的力度
                # 短距离用小力，长距离用大力
                base_speed = self._calculate_optimal_speed(dist_to_ghost, dist_to_pocket)
                
                # 生成多个力度和旋转的变体
                speed_variants = [base_speed * 0.85, base_speed, base_speed * 1.15]
                spin_variants = [
                    (0.0, 0.0),      # 无旋转
                    (0.0, -0.15),    # 低杆（拉杆）
                    (0.0, 0.15),     # 高杆（跟进）
                    (-0.1, 0.0),     # 左塞
                    (0.1, 0.0),      # 右塞
                ]
                
                for speed in speed_variants:
                    for spin_a, spin_b in spin_variants:
                        candidates.append({
                            'V0': float(np.clip(speed, 0.5, 8.0)),
                            'phi': float(phi),
                            'theta': 2.0,  # 小仰角
                            'a': float(spin_a),
                            'b': float(spin_b)
                        })
        
        # 添加一些直接瞄准目标球的候选（用于近距离球）
        for bid in player_targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            cue_to_ball = ball_pos - cue_pos
            dist = np.linalg.norm(cue_to_ball)
            
            if dist < 0.3:  # 近距离球
                phi = math.degrees(math.atan2(cue_to_ball[1], cue_to_ball[0])) % 360
                for speed in [1.5, 2.0, 2.5]:
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 1.0,
                        'a': 0.0,
                        'b': 0.0
                    })
        
        return candidates
    
    def _is_path_blocked(self, start_pos, end_pos, balls, exclude_ball_id):
        """检查两点之间的路径是否被其他球阻挡"""
        direction = end_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return False
        
        direction_unit = direction / dist
        
        for bid, ball in balls.items():
            if bid == 'cue' or bid == exclude_ball_id:
                continue
            if ball.state.s == 4:  # 已进袋
                continue
            
            ball_pos = np.array(ball.state.rvw[0][:2], dtype=float)
            
            # 计算球心到路径的距离
            to_ball = ball_pos - start_pos
            proj_length = np.dot(to_ball, direction_unit)
            
            if proj_length < 0 or proj_length > dist:
                continue
            
            closest_point = start_pos + direction_unit * proj_length
            dist_to_path = np.linalg.norm(ball_pos - closest_point)
            
            # 如果距离小于两球直径，则路径被阻挡
            if dist_to_path < 2 * self.BALL_RADIUS + 0.005:
                return True
        
        return False
    
    def _calculate_optimal_speed(self, dist_to_target, dist_to_pocket):
        """根据距离计算最优击球速度"""
        total_dist = dist_to_target + dist_to_pocket
        
        # 基础速度公式：考虑摩擦损耗
        # 短距离：1.5-2.5 m/s
        # 中距离：2.5-4.0 m/s
        # 长距离：4.0-6.0 m/s
        if total_dist < 0.5:
            speed = 1.8 + total_dist * 1.5
        elif total_dist < 1.0:
            speed = 2.5 + (total_dist - 0.5) * 2.0
        elif total_dist < 1.5:
            speed = 3.5 + (total_dist - 1.0) * 2.0
        else:
            speed = 4.5 + (total_dist - 1.5) * 1.5
        
        return float(np.clip(speed, 1.5, 6.5))
    
    def _cma_es_optimize(self, initial_action, reward_fn, balls, table):
        """使用 CMA-ES 对初始动作进行局部优化"""
        try:
            import cma
        except ImportError:
            # 如果没有 cma 库，使用简单的局部搜索
            return self._simple_local_search(initial_action, reward_fn)
        
        # 初始点
        x0 = [
            initial_action['V0'],
            initial_action['phi'],
            initial_action['theta'],
            initial_action['a'],
            initial_action['b']
        ]
        
        # 搜索范围（局部优化，范围较小）
        bounds = [
            [max(0.5, x0[0] - 1.5), min(8.0, x0[0] + 1.5)],  # V0
            [x0[1] - 5.0, x0[1] + 5.0],  # phi (允许 ±5 度)
            [max(0, x0[2] - 3.0), min(90, x0[2] + 3.0)],  # theta
            [max(-0.5, x0[3] - 0.2), min(0.5, x0[3] + 0.2)],  # a
            [max(-0.5, x0[4] - 0.2), min(0.5, x0[4] + 0.2)],  # b
        ]
        
        # CMA-ES 优化（最大化 reward，所以取负）
        def neg_reward(x):
            V0, phi, theta, a, b = x
            # 处理 phi 的周期性
            phi = phi % 360
            params = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
            if self._is_degenerate_params(params):
                return 500  # 惩罚退化参数
            return -reward_fn(V0, phi, theta, a, b)
        
        try:
            es = cma.CMAEvolutionStrategy(
                x0,
                self.cma_sigma,
                {
                    'bounds': [
                        [b[0] for b in bounds],
                        [b[1] for b in bounds]
                    ],
                    'popsize': self.cma_population_size,
                    'maxiter': self.cma_generations,
                    'verbose': -9,  # 静默模式
                    'seed': np.random.randint(1e6)
                }
            )
            
            es.optimize(neg_reward)
            best_x = es.result.xbest
            best_score = -es.result.fbest
            
            best_action = {
                'V0': float(np.clip(best_x[0], 0.5, 8.0)),
                'phi': float(best_x[1] % 360),
                'theta': float(np.clip(best_x[2], 0, 90)),
                'a': float(np.clip(best_x[3], -0.5, 0.5)),
                'b': float(np.clip(best_x[4], -0.5, 0.5))
            }
            
            return best_action, best_score
            
        except Exception as e:
            print(f"[NewAgent] CMA-ES 优化失败: {e}")
            return initial_action, reward_fn(**initial_action)
    
    def _simple_local_search(self, initial_action, reward_fn):
        """简单的局部搜索（当 CMA-ES 不可用时）"""
        best_action = initial_action.copy()
        best_score = reward_fn(**initial_action)
        
        # 在初始点附近进行网格搜索
        for dV0 in [-0.5, 0, 0.5]:
            for dphi in [-2, -1, 0, 1, 2]:
                for da in [-0.1, 0, 0.1]:
                    for db in [-0.1, 0, 0.1]:
                        candidate = {
                            'V0': float(np.clip(initial_action['V0'] + dV0, 0.5, 8.0)),
                            'phi': float((initial_action['phi'] + dphi) % 360),
                            'theta': initial_action['theta'],
                            'a': float(np.clip(initial_action['a'] + da, -0.5, 0.5)),
                            'b': float(np.clip(initial_action['b'] + db, -0.5, 0.5))
                        }
                        if self._is_degenerate_params(candidate):
                            continue
                        score = reward_fn(**candidate)
                        if score > best_score:
                            best_score = score
                            best_action = candidate
        
        return best_action, best_score
    
    def _estimate_pocket_probability(self, action, balls, table, player_targets, last_state_snapshot, trials=8):
        """估计动作的进球概率"""
        successes = 0
        for _ in range(trials):
            noisy = self._sample_noisy_params(
                action['V0'], action['phi'], action['theta'], action['a'], action['b']
            )
            valid, info = self._simulate_action_outcome(
                noisy, balls, table, player_targets, last_state_snapshot
            )
            if valid and info.get('ME_INTO_POCKET'):
                successes += 1
        return successes / trials if trials > 0 else 0.0

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
        """根据局面优劣增加额外奖励 - 增强版走位评估"""
        cue_ball = shot.balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return -200.0  # 白球落袋
        
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        bonus = 0.0
        
        remaining_targets = self._remaining_targets_after_shot(shot, player_targets)
        
        # 1. 白球与下一个目标球的距离奖励
        own_positions = [
            np.array(shot.balls[bid].state.rvw[0][:2], dtype=float)
            for bid in remaining_targets
            if bid in shot.balls and shot.balls[bid].state.s != 4
        ]
        if own_positions:
            min_dist = min(np.linalg.norm(cue_pos - pos) for pos in own_positions)
            # 距离越近奖励越高
            bonus += max(0.0, self.cue_next_ball_radius - min_dist) * self.cue_next_ball_weight
            
            # 额外奖励：白球在理想击球位置（目标球与袋口连线的延长线上）
            ideal_position_bonus = self._evaluate_ideal_position(
                cue_pos, own_positions, table
            )
            bonus += ideal_position_bonus * 15.0
        
        # 2. 远离对手球的奖励
        enemy_positions = [
            np.array(shot.balls[bid].state.rvw[0][:2], dtype=float)
            for bid in self._enemy_target_ids()
            if bid in shot.balls and shot.balls[bid].state.s != 4
        ]
        if enemy_positions:
            near_enemy = min(np.linalg.norm(cue_pos - pos) for pos in enemy_positions)
            bonus -= max(0.0, self.enemy_threat_radius - near_enemy) * self.enemy_distance_weight
        
        # 3. 保护黑8（避免意外打进）
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
        
        # 4. 白球位置安全性（远离袋口）
        pocket_safety = self._evaluate_cue_safety(cue_pos, table)
        bonus += pocket_safety * 10.0
        
        # 5. 白球在台面中央区域的奖励（更多击球选择）
        center_bonus = self._evaluate_center_position(cue_pos, table)
        bonus += center_bonus * 8.0
        
        return bonus
    
    def _evaluate_ideal_position(self, cue_pos, target_positions, table):
        """评估白球是否在理想击球位置"""
        if not target_positions:
            return 0.0
        
        best_score = 0.0
        pockets = list(table.pockets.values())
        
        for target_pos in target_positions:
            for pocket in pockets:
                pocket_pos = np.array(pocket.center[:2], dtype=float)
                
                # 理想位置：目标球与袋口连线的延长线上
                target_to_pocket = pocket_pos - target_pos
                dist = np.linalg.norm(target_to_pocket)
                if dist < 1e-3:
                    continue
                
                direction = target_to_pocket / dist
                # 理想白球位置在目标球后方 0.3-0.6 米
                ideal_pos = target_pos - direction * 0.4
                
                # 计算白球与理想位置的距离
                dist_to_ideal = np.linalg.norm(cue_pos - ideal_pos)
                score = max(0.0, 1.0 - dist_to_ideal / 0.5)
                best_score = max(best_score, score)
        
        return best_score
    
    def _evaluate_cue_safety(self, cue_pos, table):
        """评估白球位置的安全性（远离袋口）"""
        min_pocket_dist = float('inf')
        for pocket in table.pockets.values():
            pocket_pos = np.array(pocket.center[:2], dtype=float)
            dist = np.linalg.norm(cue_pos - pocket_pos)
            min_pocket_dist = min(min_pocket_dist, dist)
        
        # 距离袋口越远越安全
        if min_pocket_dist < 0.1:
            return -1.0  # 危险
        elif min_pocket_dist < 0.2:
            return 0.0
        else:
            return min(1.0, (min_pocket_dist - 0.2) / 0.3)
    
    def _evaluate_center_position(self, cue_pos, table):
        """评估白球是否在台面中央区域"""
        center_x = table.l / 2
        center_y = table.w / 2
        
        dist_from_center = np.sqrt(
            (cue_pos[0] - center_x) ** 2 + (cue_pos[1] - center_y) ** 2
        )
        
        # 距离中心越近越好
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        return max(0.0, 1.0 - dist_from_center / max_dist)
    
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
        """生成多个安全球候选 - 增强版：更智能的防守策略"""
        cue_ball = balls.get('cue')
        if cue_ball is None:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        margin = 0.07
        clamp_x = np.clip(cue_pos[0], margin, table.l - margin)
        clamp_y = np.clip(cue_pos[1], margin, table.w - margin)
        
        candidates = []
        base_speeds = np.linspace(self.safe_speed_bounds[0], self.safe_speed_bounds[1], num=4)
        
        # 策略1：将白球推向远端库边（增加对手难度）
        far_corners = [
            np.array([margin, margin]),  # 左下角
            np.array([margin, table.w - margin]),  # 左上角
            np.array([table.l - margin, margin]),  # 右下角
            np.array([table.l - margin, table.w - margin])  # 右上角
        ]
        
        for corner in far_corners:
            direction = corner - cue_pos
            dist = np.linalg.norm(direction)
            if dist < 0.1:
                continue
            phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            
            # 根据距离调整力度
            for speed_factor in [0.8, 1.0, 1.2]:
                speed = min(dist * 1.5 * speed_factor, self.safe_speed_bounds[1])
                speed = max(speed, self.safe_speed_bounds[0])
                
                for spin in [(0.0, -0.1), (-0.1, 0.0), (0.1, 0.0), (0.0, 0.0)]:
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 1.5,
                        'a': float(spin[0]),
                        'b': float(spin[1])
                    })
        
        # 策略2：轻触己方球后藏到库边
        nearest_own = self._nearest_own_ball(balls)
        if nearest_own is not None:
            own_vec = nearest_own - cue_pos
            dist_to_own = np.linalg.norm(own_vec)
            if dist_to_own > 0.05:
                phi = math.degrees(math.atan2(own_vec[1], own_vec[0])) % 360
                
                # 轻触后拉杆
                for speed in [1.2, 1.5, 1.8]:
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 1.0,
                        'a': 0.0,
                        'b': -0.2  # 低杆拉回
                    })
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 1.0,
                        'a': -0.15,
                        'b': -0.15  # 左下塞
                    })
                    candidates.append({
                        'V0': float(speed),
                        'phi': float(phi),
                        'theta': 1.0,
                        'a': 0.15,
                        'b': -0.15  # 右下塞
                    })
        
        # 策略3：制造斯诺克（将白球藏到障碍球后面）
        snooker_candidates = self._generate_snooker_candidates(balls, table, cue_pos)
        candidates.extend(snooker_candidates)
        
        # 策略4：简单碰库（确保合法）
        cushion_targets = [
            np.array([margin, clamp_y]),
            np.array([table.l - margin, clamp_y]),
            np.array([clamp_x, margin]),
            np.array([clamp_x, table.w - margin])
        ]
        
        for target_point in cushion_targets:
            direction = target_point - cue_pos
            if np.linalg.norm(direction) < 1e-6:
                continue
            phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            for speed in base_speeds:
                candidates.append({
                    'V0': float(speed),
                    'phi': float(phi),
                    'theta': 2.0,
                    'a': 0.0,
                    'b': 0.0
                })
        
        return candidates
    
    def _generate_snooker_candidates(self, balls, table, cue_pos):
        """生成斯诺克候选（将白球藏到障碍球后面）"""
        candidates = []
        
        # 找到对手的球
        enemy_ids = self._enemy_target_ids()
        enemy_positions = [
            np.array(balls[bid].state.rvw[0][:2], dtype=float)
            for bid in enemy_ids
            if bid in balls and balls[bid].state.s != 4
        ]
        
        if not enemy_positions:
            return candidates
        
        # 找到可以用来遮挡的球（己方球或其他球）
        blocker_positions = []
        for bid, ball in balls.items():
            if bid == 'cue' or ball.state.s == 4:
                continue
            if bid not in enemy_ids:  # 己方球或黑8
                blocker_positions.append(np.array(ball.state.rvw[0][:2], dtype=float))
        
        if not blocker_positions:
            return candidates
        
        # 对于每个遮挡球，计算可以藏白球的位置
        for blocker_pos in blocker_positions:
            for enemy_pos in enemy_positions:
                # 计算遮挡方向
                blocker_to_enemy = enemy_pos - blocker_pos
                dist = np.linalg.norm(blocker_to_enemy)
                if dist < 0.1:
                    continue
                
                direction = blocker_to_enemy / dist
                
                # 理想藏球位置：在遮挡球的另一侧
                hide_pos = blocker_pos - direction * 0.15
                
                # 检查位置是否在台面内
                if (hide_pos[0] < 0.1 or hide_pos[0] > table.l - 0.1 or
                    hide_pos[1] < 0.1 or hide_pos[1] > table.w - 0.1):
                    continue
                
                # 计算击球参数
                cue_to_hide = hide_pos - cue_pos
                dist_to_hide = np.linalg.norm(cue_to_hide)
                if dist_to_hide < 0.1:
                    continue
                
                phi = math.degrees(math.atan2(cue_to_hide[1], cue_to_hide[0])) % 360
                speed = min(dist_to_hide * 2.0, 2.5)
                speed = max(speed, 1.5)
                
                candidates.append({
                    'V0': float(speed),
                    'phi': float(phi),
                    'theta': 1.5,
                    'a': 0.0,
                    'b': -0.1
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
        if best_action is not None:
            sim_valid, sim_info = self._simulate_action_outcome(
                best_action, balls, table, player_targets, last_state_snapshot
            )
            if sim_valid and sim_info.get('NO_POCKET_NO_RAIL'):
                return None, current_score
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
