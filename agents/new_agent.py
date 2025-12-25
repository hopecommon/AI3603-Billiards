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
import cma  # CMA-ES优化器

from .agent import Agent

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



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
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
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
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
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

class NewAgent(Agent):
    """CMA-ES Sniper - 狙击手优化版"""
    
    def __init__(self,
                 n_simulations=50,
                 c_puct=1.414):
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }
        
        # CMA-ES 配置
        self.use_cmaes_sniper = True      # 启用CMA-ES狙击模式
        self.cmaes_max_eval = 20          # CMA-ES最大评估次数
        self.geometric_top_k = 3          # 几何筛选Top-K
        self.simple_shot_threshold = 5    # 剩余球<5个时用CMA-ES
        
        # 优化配置
        self.enable_adaptive_sims = True
        self.enable_safety_penalty = True
        self.cue_danger_dist = 0.08
        self.black_risk_penalty = 50.0
        self.cue_danger_penalty = 20.0
        
        print("[NewAgent] CMA-ES Sniper Mode - 狙击手优化版已初始化")

    def fast_geometric_filter(self, balls, my_targets, table):
        """快速几何筛选：找出理论可进的Top-K球（纯numpy，<0.01秒）"""
        cue_ball = balls.get('cue')
        if not cue_ball:
            return []
        
        cue_pos = np.array(cue_ball.state.rvw[0][:2])
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        if not target_ids:
            target_ids = ['8']
        
        candidates = []
        
        for tid in target_ids:
            obj_pos = np.array(balls[tid].state.rvw[0][:2])
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array(pocket.center[:2])
                
                # 计算角度
                vec_obj_to_pocket = pocket_pos - obj_pos
                vec_cue_to_obj = obj_pos - cue_pos
                
                dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
                dist_cue_to_obj = np.linalg.norm(vec_cue_to_obj)
                
                if dist_obj_to_pocket < 0.01 or dist_cue_to_obj < 0.01:
                    continue
                
                # 计算进球角度
                angle_rad = np.arccos(np.clip(
                    np.dot(-vec_cue_to_obj, vec_obj_to_pocket) / 
                    (dist_cue_to_obj * dist_obj_to_pocket), -1, 1
                ))
                angle_deg = np.degrees(angle_rad)
                
                # 筛选条件：角度<80度，距离合理
                if angle_deg < 80 and dist_cue_to_obj < 3.0:
                    # 精确遮挡检测：只检查白球到目标球之间的路径
                    is_blocked = False
                    for other_id, other_ball in balls.items():
                        if other_id in [tid, 'cue'] or other_ball.state.s == 4:
                            continue
                        other_pos = np.array(other_ball.state.rvw[0][:2])
                        
                        # 计算other_pos在击球路径上的投影
                        vec_cue_to_other = other_pos - cue_pos
                        projection_length = np.dot(vec_cue_to_other, vec_cue_to_obj) / dist_cue_to_obj
                        
                        # 只有当投影在[0, dist_cue_to_obj]范围内，才可能遮挡
                        if 0 < projection_length < dist_cue_to_obj:
                            # 计算垂直距离
                            dist_to_line = np.abs(np.cross(vec_cue_to_obj, cue_pos - other_pos)) / dist_cue_to_obj
                            if dist_to_line < self.ball_radius * 2.5:
                                is_blocked = True
                                break
                    
                    if not is_blocked:
                        # 评分：角度越小、距离越近越好
                        score = 100 - angle_deg - dist_cue_to_obj * 5
                        candidates.append({
                            'target_id': tid,
                            'pocket_id': pocket_id,
                            'score': score,
                            'angle': angle_deg,
                            'distance': dist_cue_to_obj
                        })
        
        # 按得分排序，返回Top-K
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.geometric_top_k]

    def cmaes_optimize_shot(self, balls, table, target_id, pocket_id, my_targets):
        """CMA-ES优化单个击球：找到鲁棒性最大的参数"""
        cue_pos = balls['cue'].state.rvw[0]
        obj_pos = balls[target_id].state.rvw[0]
        pocket_pos = table.pockets[pocket_id].center
        
        # 初始猜测
        phi_init, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
        v0_init = np.clip(1.5 + dist * 1.5, 1.0, 6.0)
        
        # CMA-ES初始参数 [V0, phi]
        x0 = np.array([v0_init, phi_init])
        sigma0 = 0.5  # 初始步长
        
        # 定义目标函数（负数，因为CMA-ES最小化）
        def objective(x):
            v0, phi = x
            # 边界约束
            if v0 < 0.5 or v0 > 8.0 or phi < 0 or phi > 360:
                return 1000.0  # 惩罚越界
            
            action = {'V0': v0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
            
            # 多样本评估（鲁棒性）- 减少到3次采样以加速
            rewards = []
            for _ in range(3):  # 从5次减少到3次
                shot = self.simulate_action(balls, table, action, my_targets)
                if shot is None:
                    rewards.append(-500)
                else:
                    reward = analyze_shot_for_reward(shot, balls, my_targets)
                    reward += self.calc_safety_penalty(shot, table, my_targets)
                    rewards.append(reward)
            
            # 鲁棒性得分 = 均值 - 标准差（惩罚不稳定）
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            robust_score = mean_reward - 0.5 * std_reward
            
            return -robust_score  # 负数，CMA-ES求最小
        
        # 运行CMA-ES
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': [[0.5, 0], [8.0, 360]],
                'maxfevals': self.cmaes_max_eval,
                'verbose': -1,  # 静默模式
                'verb_filenameprefix': ''  # 禁用日志文件输出
            })
            es.optimize(objective)
            best_x = es.result.xbest
            best_score = -es.result.fbest
            
            return {
                'V0': float(best_x[0]),
                'phi': float(best_x[1]) % 360,
                'theta': 0,
                'a': 0,
                'b': 0
            }, best_score
        except Exception as e:
            # CMA-ES失败，返回初始猜测
            return {
                'V0': v0_init,
                'phi': phi_init,
                'theta': 0,
                'a': 0,
                'b': 0
            }, 0.0

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        """生成候选动作列表（增强版：更多角度微调）"""
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 精准一击
                actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})
                # 力度稍大
                actions.append({'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})
                # 角度微调 ±0.5, ±1.0, ±2.0度
                for delta in [0.5, 1.0, 2.0]:
                    actions.append({'V0': v_base, 'phi': (phi_ideal + delta) % 360, 'theta': 0, 'a': 0, 'b': 0})
                    actions.append({'V0': v_base, 'phi': (phi_ideal - delta) % 360, 'theta': 0, 'a': 0, 'b': 0})

        if len(actions) == 0:
            for _ in range(10):
                actions.append(self._random_action())
        
        random.shuffle(actions)
        return actions[:40]

    def calc_safety_penalty(self, shot, table, my_targets):
        """计算安全惩罚（轻量级）"""
        if not self.enable_safety_penalty:
            return 0.0
        
        penalty = 0.0
        
        # 白球危险位置
        cue_ball = shot.balls.get('cue')
        if cue_ball and cue_ball.state.s != 4:
            cue_pos = cue_ball.state.rvw[0]
            min_dist = float('inf')
            for pocket_id, pocket in table.pockets.items():
                dist = np.linalg.norm(np.array(cue_pos[:2]) - np.array(pocket.center[:2]))
                min_dist = min(min_dist, dist)
            
            if min_dist < self.cue_danger_dist:
                penalty -= self.cue_danger_penalty * (1.0 - min_dist / self.cue_danger_dist)
        
        # 黑8风险
        if '8' not in my_targets:
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                    valid_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
                    other_ids = [i for i in ids if i != 'cue' and i in valid_ids]
                    if other_ids and other_ids[0] == '8':
                        penalty -= self.black_risk_penalty
                        break
        
        return penalty

    def simulate_action(self, balls, table, action, my_targets=None):
        """执行带噪声的物理仿真（带超时保护）"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # 注入高斯噪声
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            
            # 使用超时保护模拟
            success = simulate_with_timeout(shot, timeout=2)
            if not success:
                return None
            
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0: my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # === CMA-ES狙击模式 ===
        if self.use_cmaes_sniper and len(remaining) <= self.simple_shot_threshold:
            print(f"[NewAgent] CMA-ES狙击模式 (剩余{len(remaining)}球)")
            
            # 快速几何筛选
            candidates = self.fast_geometric_filter(balls, my_targets, table)
            
            if len(candidates) > 0:
                best_action = None
                best_score = -float('inf')
                
                # 对Top-K候选运行CMA-ES
                for cand in candidates:
                    action, score = self.cmaes_optimize_shot(
                        balls, table, cand['target_id'], cand['pocket_id'], my_targets
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                if best_action is not None:
                    print(f"[NewAgent] CMA-ES最优解: Score={best_score:.1f}, V0={best_action['V0']:.2f}, phi={best_action['phi']:.1f}")
                    return best_action
        
        # === MCTS备用模式 ===
        # 自适应模拟次数
        if self.enable_adaptive_sims:
            if len(remaining) < 3:
                n_sims = 30
            elif len(remaining) >= 5:
                n_sims = 70
            else:
                n_sims = 50
        else:
            n_sims = self.n_simulations

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(n_sims):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation
            shot = self.simulate_action(balls, table, candidate_actions[idx], my_targets)

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                raw_reward += self.calc_safety_penalty(shot, table, my_targets)
            
            # Normalization: [-600, 150]
            normalized_reward = (raw_reward - (-600)) / 750.0
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward

        # Final Decision
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        print(f"[NewAgent] MCTS: Score={avg_rewards[best_idx]:.3f}, Sims={n_sims}, Remain={len(remaining)}")
        
        return best_action

