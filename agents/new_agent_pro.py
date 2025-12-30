import math
import copy
import time
import numpy as np
import pooltool as pt

from .new_agent import analyze_shot_for_reward
from .agent import Agent


class OptimizedNewAgentPro(Agent):
    """
    纯 MCTS 版本：不使用 new_agent.py 的启发式评分，只用物理仿真结果 + 规则奖励。
    """

    def __init__(self,
                 n_simulations=80,
                 n_simulations_critical=120,
                 c_puct=1.1,
                 max_actions_active=24,
                 max_actions_pool=48,
                 initial_active=16,
                 pw_c=2.5,
                 pw_alpha=0.5,
                 rave_k=100.0):
        super().__init__()

        self.n_simulations = n_simulations
        self.n_simulations_critical = n_simulations_critical
        self.c_puct = c_puct
        self.max_actions_active = max_actions_active
        self.max_actions_pool = max_actions_pool
        self.initial_active = initial_active
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.rave_k = rave_k

        # 噪声与惩罚参数（复用 new_agent 里的风险权重）
        self.mcts_noise = {'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005}
        self.catastrophic_foul_penalty = 8000.0
        self.scratch_extra_penalty = 600.0
        self.first_hit_extra_penalty = 120.0
        self.no_rail_extra_penalty = 80.0
        self.no_hit_extra_penalty = 200.0

        print("OptimizedNewAgentPro 已初始化 (纯MCTS版)")

    # ========== 主决策（纯 MCTS） ==========
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None:
            return self._random_action()

        prepared_targets = self._prepare_targets(balls, my_targets)
        is_critical = ('8' in prepared_targets) or (len(prepared_targets) <= 3)
        actions = self._enumerate_actions(balls, table, prepared_targets)
        if not actions:
            return self._random_action()

        sims = self.n_simulations_critical if is_critical else self.n_simulations
        best_action = self._mcts(actions, balls, table, prepared_targets, sims)
        best_action = self._sanitize_action(best_action)
        return best_action if best_action else self._random_action()

    # ========== 动作枚举（无启发式打分，只生成候选） ==========
    def _enumerate_actions(self, balls, table, targets):
        candidates = []
        cue = balls.get("cue")
        if cue is None or cue.state.s == 4:
            return candidates
        cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]

        # 1) ghost ball 基础候选
        BALL_RADIUS = 0.028575
        target_infos = []
        for bid in targets:
            b = balls.get(bid)
            if b is None or b.state.s == 4:
                continue
            pos = np.array(b.state.rvw[0][:2], dtype=float)
            dist = np.linalg.norm(pos - cue_pos)
            target_infos.append((dist, bid, pos))
        target_infos.sort(key=lambda x: x[0])
        target_infos = target_infos[:4]  # 只取最近 4 个目标

        for _, bid, pos in target_infos:
            # 最近两个袋口
            pocket_dists = [(np.linalg.norm(pos - ppos), ppos) for ppos in pocket_positions]
            pocket_dists.sort(key=lambda x: x[0])
            for _, ppos in pocket_dists[:2]:
                vec_obj_to_pocket = ppos - pos
                dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
                if dist_obj_to_pocket < 1e-6:
                    continue
                unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
                ghost_pos = pos - unit_vec * (2 * BALL_RADIUS)
                vec_cue_to_ghost = ghost_pos - cue_pos
                dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
                if dist_cue_to_ghost < 1e-6:
                    continue
                # 路径被其他球严重遮挡则跳过
                if self._is_path_blocked_simple(cue_pos, ghost_pos, balls, ignore_ids={bid, 'cue'}):
                    continue
                phi = math.degrees(math.atan2(vec_cue_to_ghost[1], vec_cue_to_ghost[0])) % 360
                base_v0 = float(np.clip(1.8 + dist_cue_to_ghost * 1.8, 1.6, 5.5))

                for scale in (0.92, 1.0, 1.08):
                    for dphi in (-0.8, 0.0, 0.8):
                        candidates.append({
                            "V0": float(np.clip(base_v0 * scale, 0.5, 6.0)),
                            "phi": float((phi + dphi) % 360),
                            "theta": 2.0,
                            "a": 0.0,
                            "b": 0.0,
                            "meta_target": bid
                        })

        # 2) 保证碰球的兜底（最近目标，角度轻微偏移）
        candidates.extend(self._generate_contact_fallback_candidates(balls, table, targets, max_targets=2))

        # 3) 简易防守球：将母球送向远端库边，降低对手直接进攻角度
        candidates.extend(self._generate_safety_candidates(balls, table, max_cnt=3))

        # 去重并截断
        uniq = {}
        for c in candidates:
            key = (
                round(c["V0"], 2),
                round(c["phi"] % 360, 2),
                round(c["theta"], 2),
                round(c["a"], 3),
                round(c["b"], 3),
            )
            if key not in uniq:
                uniq[key] = c
            if len(uniq) >= self.max_actions_pool:
                break
        return list(uniq.values())

    # ========== MCTS ==========
    def _mcts(self, actions, balls, table, targets, n_simulations):
        n = len(actions)
        # Progressive widening: 初始激活子集，访问次数增加后逐步放开
        active = list(range(min(self.initial_active, n)))
        inactive = list(range(len(active), n))

        N = np.zeros(n, dtype=float)
        Q = np.zeros(n, dtype=float)
        RAVE_N = np.zeros(n, dtype=float)
        RAVE_Q = np.zeros(n, dtype=float)

        for sim in range(n_simulations):
            # progressive widening: 根据访问次数决定是否扩展新动作
            total_n = np.sum(N)
            max_active_allow = min(n, int(self.max_actions_active + self.pw_c * (total_n ** self.pw_alpha)))
            while len(active) < max_active_allow and inactive:
                active.append(inactive.pop(0))

            # selection
            if sim < len(active):
                idx = active[sim]
            else:
                total_n = np.sum(N[active])
                q_mean = Q / (N + 1e-6)
                rave_mean = RAVE_Q / (RAVE_N + 1e-6)
                beta = np.sqrt(self.rave_k / (3.0 * (N + 1e-6) + self.rave_k))
                mixed_q = (1.0 - beta) * q_mean + beta * rave_mean
                ucb = mixed_q + self.c_puct * np.sqrt(np.log(total_n + 1.0) / (N + 1e-6))
                ucb_active = [(ucb[i], i) for i in active]
                idx = max(ucb_active, key=lambda x: x[0])[1]

            reward, first_contact = self._rollout(actions[idx], balls, table, targets)
            N[idx] += 1
            Q[idx] += reward

            # RAVE 更新：对同一预期目标（meta_target）或同一首碰球的动作累积
            target_key = first_contact or actions[idx].get("meta_target")
            for j in active:
                if target_key is not None and target_key == actions[j].get("meta_target"):
                    RAVE_N[j] += 1
                    RAVE_Q[j] += reward
            # 也确保被选动作自身纳入 RAVE
            if target_key is None:
                RAVE_N[idx] += 1
                RAVE_Q[idx] += reward

        avg = Q / (N + 1e-6)
        best_idx = int(np.argmax(avg))
        return actions[best_idx]

    # ========== Rollout ==========
    def _rollout(self, action, balls, table, targets):
        # 单次噪声仿真 + 规则打分 + 走位奖励
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        noisy = {
            "V0": float(np.clip(action["V0"] + np.random.normal(0, self.mcts_noise["V0"]), 0.5, 8.0)),
            "phi": float((action["phi"] + np.random.normal(0, self.mcts_noise["phi"])) % 360),
            "theta": float(np.clip(action["theta"] + np.random.normal(0, self.mcts_noise["theta"]), 0, 90)),
            "a": float(np.clip(action["a"] + np.random.normal(0, self.mcts_noise["a"]), -0.5, 0.5)),
            "b": float(np.clip(action["b"] + np.random.normal(0, self.mcts_noise["b"]), -0.5, 0.5)),
        }
        try:
            shot.cue.set_state(**noisy)
            pt.simulate(shot, inplace=True)
        except Exception:
            return -500.0, None

        base_score = analyze_shot_for_reward(shot, balls, targets)
        foul, first_contact = self._analyze_fouls(shot, balls, targets)

        reward = base_score
        # 强硬风险压制：致命犯规直接极大惩罚，首犯/白球/未碰给半灾难级惩罚
        if foul["WHITE_AND_EIGHT"] or foul["ILLEGAL_EIGHT"]:
            return -self.catastrophic_foul_penalty, first_contact
        if foul["CUE_POCKETED"] or foul["FOUL_FIRST_HIT"] or foul["NO_HIT"]:
            return -0.5 * self.catastrophic_foul_penalty, first_contact
        if foul["FOUL_NO_RAIL"]:
            reward -= self.no_rail_extra_penalty
        # 走位奖励：若本杆有效且有进球，鼓励母球停在下杆易进球位置
        reward += self._estimate_position_bonus(shot, targets)
        return float(reward), first_contact

    # ========== 简易遮挡检测 ==========
    def _is_path_blocked_simple(self, start, end, balls, ignore_ids=None):
        ignore_ids = ignore_ids or set()
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return False
        direction = direction / dist
        BALL_RADIUS = 0.028575
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            to_ball = pos - start
            proj = np.dot(to_ball, direction)
            if proj < 0 or proj > dist:
                continue
            perp = np.linalg.norm(to_ball - proj * direction)
            if perp < 2.2 * BALL_RADIUS:
                return True
        return False

    # ========== 规则辅助 ==========
    def _prepare_targets(self, balls, my_targets):
        if my_targets is None:
            return []
        valid = [t for t in my_targets if t in balls and balls[t].state.s != 4]
        return valid if valid else ['8']

    def _analyze_fouls(self, shot, last_state, targets):
        new_pocketed = [
            bid for bid, b in shot.balls.items()
            if bid in last_state and b.state.s == 4 and last_state[bid].state.s != 4
        ]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        legal_eight = (len(targets) == 1 and targets[0] == "8")
        illegal_eight = eight_pocketed and (not legal_eight)
        white_and_eight = cue_pocketed and eight_pocketed

        # 首球接触
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
        }, first_contact_ball_id

    def _generate_contact_fallback_candidates(self, balls, table, targets, max_targets=1):
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)

        targets_list = []
        for bid in targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4:
                continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            targets_list.append((float(np.linalg.norm(pos - cue_pos)), bid, pos))
        if not targets_list:
            return []
        targets_list.sort(key=lambda x: x[0])
        targets_list = targets_list[:max_targets]

        candidates = []
        angle_offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
        for dist, _, pos in targets_list:
            direction = pos - cue_pos
            if np.linalg.norm(direction) < 1e-6:
                continue
            base_phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            base_v0 = float(np.clip(1.6 + dist * 1.8, 1.8, 4.2))
            for off in angle_offsets:
                candidates.append({
                    'V0': base_v0,
                    'phi': float((base_phi + off) % 360),
                    'theta': 2.0,
                    'a': 0.0,
                    'b': 0.0,
                    "meta_target": bid if bid in targets else None
                })
        return candidates

    def _generate_safety_candidates(self, balls, table, max_cnt=3):
        """简单防守球：将母球推向远端库边，降低对手进攻角度"""
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4:
            return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        # 选取四个库中心方向
        rail_points = []
        if hasattr(table, "boundaries") and table.boundaries:
            try:
                xmin, xmax = table.boundaries["x"]
                ymin, ymax = table.boundaries["y"]
                rail_points = [
                    np.array([xmin + 0.05, cue_pos[1]]),
                    np.array([xmax - 0.05, cue_pos[1]]),
                    np.array([cue_pos[0], ymin + 0.05]),
                    np.array([cue_pos[0], ymax - 0.05]),
                ]
            except Exception:
                pass
        if not rail_points:
            return []

        cand = []
        for pt in rail_points:
            vec = pt - cue_pos
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue
            phi = math.degrees(math.atan2(vec[1], vec[0])) % 360
            v0 = float(np.clip(1.5 + dist * 1.2, 1.4, 4.0))
            cand.append({
                "V0": v0,
                "phi": phi,
                "theta": 2.0,
                "a": 0.0,
                "b": 0.0,
                "meta_target": None
            })
            if len(cand) >= max_cnt:
                break
        return cand

    def _estimate_position_bonus(self, shot, targets):
        """在合法击球后，鼓励母球靠近下一颗易打球的幽灵点"""
        cue_ball = shot.balls.get("cue")
        if cue_ball is None or cue_ball.state.s == 4:
            return 0.0
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        pockets = list(getattr(shot, "table", None).pockets.values()) if getattr(shot, "table", None) else []
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in pockets]
        if not pocket_positions:
            return 0.0

        BALL_RADIUS = 0.028575
        remaining = []
        for bid in targets:
            b = shot.balls.get(bid)
            if b is None or b.state.s == 4:
                continue
            pos = np.array(b.state.rvw[0][:2], dtype=float)
            remaining.append((bid, pos))
        if not remaining:
            return 0.0

        best_dist = None
        for bid, pos in remaining:
            pocket_dists = [(np.linalg.norm(pos - ppos), ppos) for ppos in pocket_positions]
            pocket_dists.sort(key=lambda x: x[0])
            for _, ppos in pocket_dists[:2]:
                vec_obj_to_pocket = ppos - pos
                dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
                if dist_obj_to_pocket < 1e-6:
                    continue
                unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
                ghost_pos = pos - unit_vec * (2 * BALL_RADIUS)
                # 路径需不被其他球遮挡
                if self._is_path_blocked_simple(cue_pos, ghost_pos, shot.balls, ignore_ids={bid, 'cue'}):
                    continue
                dist_cue_to_ghost = np.linalg.norm(ghost_pos - cue_pos)
                if best_dist is None or dist_cue_to_ghost < best_dist:
                    best_dist = dist_cue_to_ghost
        if best_dist is None:
            return 0.0
        # 距离越近越好，设置一个平滑上限
        return float(max(0.0, 80.0 - best_dist * 800.0))

    def _sanitize_action(self, action):
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
        V0 = float(np.clip(V0, 0.5, 8.0))
        phi = float(phi % 360.0)
        theta = float(np.clip(theta, 0.0, 90.0))
        a = float(np.clip(a, -0.5, 0.5))
        b = float(np.clip(b, -0.5, 0.5))
        return {"V0": V0, "phi": phi, "theta": theta, "a": a, "b": b}

    def _random_action(self):
        return {
            'V0': float(np.random.uniform(2.0, 5.0)),
            'phi': float(np.random.uniform(0, 360)),
            'theta': 2.0,
            'a': 0.0,
            'b': 0.0
        }

# 向后兼容别名
NewAgentPro = OptimizedNewAgentPro

