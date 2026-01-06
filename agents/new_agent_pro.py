import math
import copy
import time
import numpy as np
import pooltool as pt

from .new_agent import analyze_shot_for_reward
from .agent import Agent


class OptimizedNewAgentPro(Agent):
    """
    纯 MCTS 版本 V4：极限进攻版。
    针对 120 局 43.3% 胜率进行的深度优化：
    1. 增加模拟次数 (120/200) 与动作池 (64)
    2. 进一步压低 lambda (0.03/0.20)，释放进攻火力
    3. 候选球扩展至 8 颗，并引入纵向旋转 (b) 变体
    4. 强化“未碰球”惩罚，解决 Player B 高犯规率问题
    """

    def __init__(self,
                 n_simulations=140,
                 n_simulations_critical=240,
                 c_puct=1.2,
                 max_actions_active=28,
                 max_actions_pool=64,
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

        # 噪声与惩罚参数
        self.mcts_noise = {'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003}
        self.catastrophic_foul_penalty = 8000.0
        self.scratch_extra_penalty = 600.0
        self.first_hit_extra_penalty = 120.0
        self.no_rail_extra_penalty = 1000.0  # 大幅提升碰库惩罚常数
        self.no_hit_extra_penalty = 300.0

        # 对手建模：极限进攻版 V4.1
        self.opponent_value_weight_early = 0.02  # 近乎零防御，专注连杆
        self.opponent_value_weight_late = 0.18   # 后期轻度干扰
        self.opponent_max_actions_pool = 18
        self.opponent_eval_k = 6
        self.opponent_simulate_threshold = 18.0  # 对手只有“必进球”时才考虑防守
        
        self.enable_safety_candidates = False
        self.offense_theta_variants = (0.0, 2.0)
        self.offense_b_variants = (0.0, 0.12, -0.12)
        self.risk_eps = 1e-9
        self.scratch_tolerance = 0.06

        print("OptimizedNewAgentPro 已初始化 (V4: 极限进攻版)")

    # ========== 主决策（纯 MCTS） ==========
    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or table is None:
            return self._random_action()

        prepared_targets = self._prepare_targets(balls, my_targets)
        is_critical = ('8' in prepared_targets) or (len(prepared_targets) <= 3)
        
        if self._is_break_state(balls):
            actions = self._enumerate_break_actions(balls, table, prepared_targets)
            sims = 180
        else:
            actions = self._enumerate_actions(balls, table, prepared_targets)
            sims = self.n_simulations_critical if is_critical else self.n_simulations
            
        if not actions:
            return self._random_action()

        best_action = self._mcts(actions, balls, table, prepared_targets, sims, is_late=is_critical)
        best_action = self._sanitize_action(best_action)
        return best_action if best_action else self._random_action()

    # ========== 动作枚举 ==========
    def _enumerate_actions(self, balls, table, targets, max_pool=None):
        candidates = []
        cue = balls.get("cue")
        if cue is None or cue.state.s == 4:
            return candidates
        cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in table.pockets.values()]

        # 1) ghost ball 分层采样
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
        
        # 扩展范围：看最近 8 颗球
        top_targets = target_infos[:8]

        for _, bid, pos in top_targets:
            cue_to_ball = pos - cue_pos
            d_cb = float(np.linalg.norm(cue_to_ball))
            if d_cb < 1e-6: continue
            u_cb = cue_to_ball / d_cb

            pocket_scores = []
            for ppos in pocket_positions:
                ball_to_pocket = ppos - pos
                d_bp = float(np.linalg.norm(ball_to_pocket))
                if d_bp < 1e-6: continue
                u_bp = ball_to_pocket / d_bp
                align = float(np.clip(np.dot(u_cb, u_bp), -1.0, 1.0))
                if align <= 0.05: continue
                if self._is_path_blocked_simple(pos, ppos, balls, ignore_ids={bid, 'cue'}):
                    continue
                score = align / (1.0 + 0.4 * d_bp)
                pocket_scores.append((score, ppos))

            pocket_scores.sort(key=lambda x: x[0], reverse=True)
            for _, ppos in pocket_scores[:2]:
                vec_obj_to_pocket = ppos - pos
                dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
                unit_vec = vec_obj_to_pocket / (dist_obj_to_pocket + 1e-8)
                ghost_pos = pos - unit_vec * (2 * BALL_RADIUS)
                vec_cue_to_ghost = ghost_pos - cue_pos
                dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
                if self._is_path_blocked_simple(cue_pos, ghost_pos, balls, ignore_ids={bid, 'cue'}):
                    continue
                
                phi = math.degrees(math.atan2(vec_cue_to_ghost[1], vec_cue_to_ghost[0])) % 360
                base_v0 = float(np.clip(1.7 + dist_cue_to_ghost * 1.6, 1.6, 5.8))

                for scale in (0.96, 1.0, 1.04):
                    for theta in self.offense_theta_variants:
                        for b_val in self.offense_b_variants:
                            candidates.append({
                                "V0": float(np.clip(base_v0 * scale, 0.5, 6.2)),
                                "phi": phi,
                                "theta": float(theta),
                                "a": 0.0, "b": float(b_val),
                                "meta_target": bid,
                                "meta_pocket": (float(ppos[0]), float(ppos[1])),
                            })

        # 2) 保证碰球的兜底
        candidates.extend(self._generate_contact_fallback_candidates(balls, table, targets, max_targets=3))

        # 去重并分层截断
        max_pool = int(max_pool) if max_pool is not None else int(self.max_actions_pool)
        grouped = {}
        for c in candidates:
            tid = c.get("meta_target", "none")
            grouped.setdefault(tid, []).append(c)
        
        uniq_list = []
        uniq_keys = set()
        group_keys = list(grouped.keys())
        group_keys.sort(key=lambda k: 0 if k != "none" else 1)
        
        while len(uniq_list) < max_pool:
            added_in_round = 0
            for k in group_keys:
                if not grouped[k]: continue
                c = grouped[k].pop(0)
                key = (round(c["V0"], 2), round(c["phi"] % 360, 2), round(c["theta"], 2), round(c["b"], 3))
                if key not in uniq_keys:
                    uniq_keys.add(key)
                    uniq_list.append(c)
                    added_in_round += 1
                if len(uniq_list) >= max_pool: break
            if added_in_round == 0: break
            
        return uniq_list

    def _is_break_state(self, balls):
        try:
            obj = []
            for i in range(1, 16):
                bid = str(i)
                b = balls.get(bid)
                if b is None or b.state.s == 4: return False
                obj.append(np.array(b.state.rvw[0][:2], dtype=float))
            cue = balls.get("cue")
            if cue is None or cue.state.s == 4: return False
            cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)
        except Exception: return False
        obj = np.stack(obj, axis=0)
        center = np.mean(obj, axis=0)
        max_pair = 0.0
        for i in range(obj.shape[0]):
            for j in range(i + 1, obj.shape[0]):
                d = float(np.linalg.norm(obj[i] - obj[j]))
                if d > max_pair: max_pair = d
        cue_to_center = float(np.linalg.norm(cue_pos - center))
        return (max_pair < 0.30) and (cue_to_center > 0.40)

    def _enumerate_break_actions(self, balls, table, targets):
        cue = balls.get("cue")
        if cue is None or cue.state.s == 4: return []
        cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)
        obj_pos = []
        for i in range(1, 16):
            bid = str(i)
            b = balls.get(bid)
            if b is None or b.state.s == 4: continue
            obj_pos.append(np.array(b.state.rvw[0][:2], dtype=float))
        if not obj_pos: return []
        center = np.mean(np.stack(obj_pos, axis=0), axis=0)
        direction = center - cue_pos
        if np.linalg.norm(direction) < 1e-6: return []
        base_phi = float(math.degrees(math.atan2(direction[1], direction[0])) % 360)
        v0_list = (2.8, 3.1, 3.4)
        dphi_list = (-0.8, -0.4, 0.0, 0.4, 0.8)
        candidates = []
        for v0 in v0_list:
            for dphi in dphi_list:
                candidates.append({
                    "V0": float(np.clip(v0, 0.5, 5.6)),
                    "phi": float((base_phi + dphi) % 360),
                    "theta": 0.0, "a": 0.0, "b": 0.0,
                    "meta_target": None, "meta_pocket": None,
                })
        uniq = {}
        for c in candidates:
            key = (round(c["V0"], 2), round(c["phi"] % 360, 2), round(c["theta"], 2))
            if key not in uniq: uniq[key] = c
        return list(uniq.values())[:20]

    # ========== MCTS ==========
    def _mcts(self, actions, balls, table, targets, n_simulations, is_late=False):
        n = len(actions)
        active = list(range(min(self.initial_active, n)))
        inactive = list(range(len(active), n))
        N = np.zeros(n, dtype=float)
        Q = np.zeros(n, dtype=float)
        RAVE_N = np.zeros(n, dtype=float)
        RAVE_Q = np.zeros(n, dtype=float)
        catastrophic = np.zeros(n, dtype=float)
        scratch = np.zeros(n, dtype=float)
        no_hit = np.zeros(n, dtype=float)
        no_rail = np.zeros(n, dtype=float)

        # 确定本手对手权重
        if len(targets) == 1 and targets[0] == '8':
            opp_weight = 0.0
        else:
            opp_weight = self.opponent_value_weight_late if is_late else self.opponent_value_weight_early

        for sim in range(n_simulations):
            total_n = np.sum(N)
            max_active_allow = min(n, int(self.max_actions_active + self.pw_c * (total_n ** self.pw_alpha)))
            while len(active) < max_active_allow and inactive:
                active.append(inactive.pop(0))

            if sim < len(active):
                idx = active[sim]
            else:
                total_n_active = np.sum(N[active])
                q_mean = Q / (N + 1e-6)
                rave_mean = RAVE_Q / (RAVE_N + 1e-6)
                beta = np.sqrt(self.rave_k / (3.0 * (N + 1e-6) + self.rave_k))
                mixed_q = (1.0 - beta) * q_mean + beta * rave_mean
                ucb = mixed_q + self.c_puct * np.sqrt(np.log(total_n_active + 1.0) / (N + 1e-6))
                idx = active[np.argmax(ucb[active])]

            reward, first_contact, foul = self._rollout(actions[idx], balls, table, targets, opp_weight=opp_weight)
            N[idx] += 1
            Q[idx] += reward
            if foul:
                if foul.get("WHITE_AND_EIGHT") or foul.get("ILLEGAL_EIGHT"):
                    catastrophic[idx] += 1
                if foul.get("CUE_POCKETED"):
                    scratch[idx] += 1
                if foul.get("NO_HIT"):
                    no_hit[idx] += 1
                if foul.get("FOUL_NO_RAIL"):
                    no_rail[idx] += 1

            target_key = first_contact or actions[idx].get("meta_target")
            for j in active:
                if target_key is not None and target_key == actions[j].get("meta_target"):
                    RAVE_N[j] += 1
                    RAVE_Q[j] += reward
            if target_key is None:
                RAVE_N[idx] += 1
                RAVE_Q[idx] += reward

        avg = Q / (N + 1e-6)
        cat_rate = catastrophic / (N + 1e-6)
        scr_rate = scratch / (N + 1e-6)
        nohit_rate = no_hit / (N + 1e-6)
        norail_rate = no_rail / (N + 1e-6)
        
        min_cat = float(np.min(cat_rate)) if n else 1.0
        cand = [i for i in range(n) if float(cat_rate[i]) <= min_cat + self.risk_eps]
        
        # 压制无碰球与不碰库风险
        min_nohit = min(float(nohit_rate[i]) for i in cand) if cand else 1.0
        cand = [i for i in cand if float(nohit_rate[i]) <= min_nohit + self.risk_eps]
        
        min_norail = min(float(norail_rate[i]) for i in cand) if cand else 1.0
        cand = [i for i in cand if float(norail_rate[i]) <= min_norail + self.risk_eps]

        min_scr = min(float(scr_rate[i]) for i in cand) if cand else 1.0
        final_cand = [i for i in cand if float(scr_rate[i]) <= min_scr + self.scratch_tolerance + self.risk_eps]
        
        best_idx = max(final_cand, key=lambda i: float(avg[i]))

        if N[best_idx] > 0 and (cat_rate[best_idx] > 0 or scr_rate[best_idx] > 0.02 or nohit_rate[best_idx] > 0.05 or norail_rate[best_idx] > 0.05):
            for _ in range(40):
                reward, _, _ = self._rollout(actions[best_idx], balls, table, targets, opp_weight=opp_weight)
                N[best_idx] += 1
                Q[best_idx] += reward

        return actions[best_idx]

    # ========== Rollout ==========
    def _rollout(self, action, balls, table, targets, opp_weight=0.25):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = table
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
        except Exception: return -500.0, None, None

        base_score = analyze_shot_for_reward(shot, balls, targets)
        foul, first_contact = self._analyze_fouls(shot, balls, targets)
        reward = float(base_score)
        if foul["WHITE_AND_EIGHT"] or foul["ILLEGAL_EIGHT"]:
            return -self.catastrophic_foul_penalty, first_contact, foul
        # 修正：将碰球不碰库 (FOUL_NO_RAIL) 纳入高惩罚项，彻底解决 Player B 的犯规黑洞
        if foul["CUE_POCKETED"] or foul["FOUL_FIRST_HIT"] or foul["NO_HIT"] or foul["FOUL_NO_RAIL"]:
            p = self.catastrophic_foul_penalty * 0.5
            if foul["NO_HIT"]: p += self.no_hit_extra_penalty
            if foul["FOUL_NO_RAIL"]: p += self.no_rail_extra_penalty
            return -p, first_contact, foul
        
        reward += 1.25 * self._estimate_position_bonus(shot, targets)
        if opp_weight > 1e-9:
            if self._turn_ends_after_shot(shot, balls, targets):
                opp_targets = self._infer_opponent_targets(shot.balls, targets)
                if opp_targets:
                    opp_best = self._estimate_opponent_best_value(shot.balls, table, opp_targets)
                    reward -= float(opp_weight) * float(opp_best)
        return float(reward), first_contact, foul

    def _turn_ends_after_shot(self, shot, last_state, targets):
        try:
            for bid in targets:
                if bid in last_state and bid in shot.balls:
                    if shot.balls[bid].state.s == 4 and last_state[bid].state.s != 4: return False
        except Exception: pass
        return True

    def _infer_opponent_targets(self, balls_after, my_targets_now):
        remaining = []
        for i in range(1, 16):
            bid = str(i)
            b = balls_after.get(bid)
            if b is None or b.state.s == 4: continue
            remaining.append(bid)
        if my_targets_now == ['8'] or (len(my_targets_now) == 1 and my_targets_now[0] == '8'):
            return [bid for bid in remaining if bid != '8']
        has_low = any(t in {'1', '2', '3', '4', '5', '6', '7'} for t in my_targets_now)
        has_high = any(t in {'9', '10', '11', '12', '13', '14', '15'} for t in my_targets_now)
        open_table = (len(my_targets_now) >= 9) or (has_low and has_high)
        if open_table: return [bid for bid in remaining if bid != '8']
        my_set = set([t for t in my_targets_now if t != '8'])
        return [bid for bid in remaining if bid != '8' and bid not in my_set]

    def _estimate_opponent_best_value(self, opp_balls, table, opp_targets):
        opp_actions = self._enumerate_actions(opp_balls, table, opp_targets, max_pool=self.opponent_max_actions_pool)
        if not opp_actions: return 0.0
        sampled = self._stratified_pick_actions(opp_actions, k=self.opponent_eval_k)
        if not sampled: return 0.0
        best_a = None
        best_h = None
        for a in sampled:
            h = float(self._heuristic_shot_value(a, opp_balls, opp_targets))
            if best_h is None or h > best_h: best_h, best_a = h, a
        if best_a is None or best_h is None or best_h < 1e-6: return 0.0
        if best_h >= float(self.opponent_simulate_threshold):
            val = float(self._rollout_leaf_value(best_a, opp_balls, table, opp_targets))
            return float(max(0.0, val))
        return float(max(0.0, best_h))

    def _heuristic_shot_value(self, action, balls, targets):
        cue = balls.get("cue")
        if cue is None or cue.state.s == 4: return 0.0
        cue_pos = np.array(cue.state.rvw[0][:2], dtype=float)
        bid = action.get("meta_target")
        if bid is None or bid not in targets: return 0.0
        b = balls.get(bid)
        if b is None or b.state.s == 4: return 0.0
        obj_pos = np.array(b.state.rvw[0][:2], dtype=float)
        p = action.get("meta_pocket")
        if p is None: return 0.0
        pocket_pos = np.array([float(p[0]), float(p[1])], dtype=float)
        cue_to_obj = obj_pos - cue_pos
        d_cb = float(np.linalg.norm(cue_to_obj))
        if d_cb < 1e-6: return 0.0
        obj_to_pocket = pocket_pos - obj_pos
        d_bp = float(np.linalg.norm(obj_to_pocket))
        if d_bp < 1e-6: return 0.0
        u_cb = cue_to_obj / d_cb
        u_bp = obj_to_pocket / d_bp
        align = float(np.clip(np.dot(u_cb, u_bp), -1.0, 1.0))
        BALL_RADIUS = 0.028575
        ghost_pos = obj_pos - u_bp * (2 * BALL_RADIUS)
        if self._is_path_blocked_simple(obj_pos, pocket_pos, balls, ignore_ids={bid, 'cue'}): return 0.0
        if self._is_path_blocked_simple(cue_pos, ghost_pos, balls, ignore_ids={bid, 'cue'}): return 0.0
        d_cg = float(np.linalg.norm(ghost_pos - cue_pos))
        v0 = float(action.get("V0", 2.5))
        score = 40.0 * max(0.0, align) + 25.0 / (1.0 + 1.2 * d_bp) + 15.0 / (1.0 + 1.2 * d_cg) - 3.0 * max(0.0, v0 - 4.0)
        return float(score)

    def _stratified_pick_actions(self, actions, k=8):
        k = int(k)
        if k <= 0 or len(actions) <= k: return list(actions)
        groups = {}
        for a in actions: groups.setdefault(a.get("meta_target"), []).append(a)
        keys = list(groups.keys())
        np.random.shuffle(keys)
        picked = []
        for key in keys:
            g = groups[key]
            picked.append(g[np.random.randint(0, len(g))])
            if len(picked) >= k: return picked
        if len(picked) < k:
            remain = [a for a in actions if a not in picked]
            if remain:
                np.random.shuffle(remain)
                picked.extend(remain[: (k - len(picked))])
        return picked[:k]

    def _rollout_leaf_value(self, action, balls, table, targets):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = table
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
        except Exception: return -500.0
        base_score = analyze_shot_for_reward(shot, balls, targets)
        foul, _ = self._analyze_fouls(shot, balls, targets)
        reward = float(base_score)
        if foul["WHITE_AND_EIGHT"] or foul["ILLEGAL_EIGHT"]: return -self.catastrophic_foul_penalty
        if foul["CUE_POCKETED"] or foul["FOUL_FIRST_HIT"] or foul["NO_HIT"]: return -0.5 * self.catastrophic_foul_penalty
        if foul["FOUL_NO_RAIL"]: reward -= self.no_rail_extra_penalty
        reward += self._estimate_position_bonus(shot, targets)
        return float(reward)

    def _is_path_blocked_simple(self, start, end, balls, ignore_ids=None):
        ignore_ids = ignore_ids or set()
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return False
        direction = direction / dist
        BALL_RADIUS = 0.028575
        for bid, ball in balls.items():
            if bid in ignore_ids or ball.state.s == 4: continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            to_ball = pos - start
            proj = np.dot(to_ball, direction)
            if proj < 0 or proj > dist: continue
            perp = np.linalg.norm(to_ball - proj * direction)
            if perp < 2.2 * BALL_RADIUS: return True
        return False

    def _prepare_targets(self, balls, my_targets):
        if my_targets is None: return []
        valid = [t for t in my_targets if t in balls and balls[t].state.s != 4]
        return valid if valid else ['8']

    def _analyze_fouls(self, shot, last_state, targets):
        new_pocketed = [bid for bid, b in shot.balls.items() if bid in last_state and b.state.s == 4 and last_state[bid].state.s != 4]
        cue_pocketed = "cue" in new_pocketed
        eight_pocketed = "8" in new_pocketed
        legal_eight = (len(targets) == 1 and targets[0] == "8")
        illegal_eight = eight_pocketed and (not legal_eight)
        white_and_eight = cue_pocketed and eight_pocketed
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
            if len(last_state) > 2 or targets != ['8']: foul_first_hit = True
        else:
            if first_contact_ball_id not in targets: foul_first_hit = True
        cue_hit_cushion = False
        target_hit_cushion = False
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids: cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids: target_hit_cushion = True
        foul_no_rail = (len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion))
        return {
            "CUE_POCKETED": cue_pocketed, "EIGHT_POCKETED": eight_pocketed,
            "ILLEGAL_EIGHT": illegal_eight, "WHITE_AND_EIGHT": white_and_eight,
            "FOUL_FIRST_HIT": foul_first_hit, "FOUL_NO_RAIL": foul_no_rail, "NO_HIT": no_hit,
        }, first_contact_ball_id

    def _generate_contact_fallback_candidates(self, balls, table, targets, max_targets=1):
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4: return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        targets_list = []
        for bid in targets:
            ball = balls.get(bid)
            if ball is None or ball.state.s == 4: continue
            pos = np.array(ball.state.rvw[0][:2], dtype=float)
            targets_list.append((float(np.linalg.norm(pos - cue_pos)), bid, pos))
        if not targets_list: return []
        targets_list.sort(key=lambda x: x[0])
        targets_list = targets_list[:max_targets]
        candidates = []
        angle_offsets = (-2.0, -1.0, 0.0, 1.0, 2.0)
        for dist, bid, pos in targets_list:
            direction = pos - cue_pos
            if np.linalg.norm(direction) < 1e-6: continue
            base_phi = math.degrees(math.atan2(direction[1], direction[0])) % 360
            base_v0 = float(np.clip(1.6 + dist * 1.8, 1.8, 4.2))
            for off in angle_offsets:
                candidates.append({
                    'V0': base_v0, 'phi': float((base_phi + off) % 360),
                    'theta': 2.0, 'a': 0.0, 'b': 0.0,
                    "meta_target": bid if bid in targets else None
                })
        return candidates

    def _generate_safety_candidates(self, balls, table, max_cnt=3):
        cue_ball = balls.get('cue')
        if cue_ball is None or cue_ball.state.s == 4: return []
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        rail_points = []
        if hasattr(table, "boundaries") and table.boundaries:
            try:
                xmin, xmax = table.boundaries["x"]
                ymin, ymax = table.boundaries["y"]
                rail_points = [
                    np.array([xmin + 0.05, cue_pos[1]]), np.array([xmax - 0.05, cue_pos[1]]),
                    np.array([cue_pos[0], ymin + 0.05]), np.array([cue_pos[0], ymax - 0.05]),
                ]
            except Exception: pass
        if not rail_points: return []
        cand = []
        for pt in rail_points:
            vec = pt - cue_pos
            dist = np.linalg.norm(vec)
            if dist < 1e-6: continue
            phi = math.degrees(math.atan2(vec[1], vec[0])) % 360
            v0 = float(np.clip(1.5 + dist * 1.2, 1.4, 4.0))
            cand.append({"V0": v0, "phi": phi, "theta": 2.0, "a": 0.0, "b": 0.0, "meta_target": None})
            if len(cand) >= max_cnt: break
        return cand

    def _estimate_position_bonus(self, shot, targets):
        cue_ball = shot.balls.get("cue")
        if cue_ball is None or cue_ball.state.s == 4: return 0.0
        cue_pos = np.array(cue_ball.state.rvw[0][:2], dtype=float)
        pockets = list(getattr(shot, "table", None).pockets.values()) if getattr(shot, "table", None) else []
        pocket_positions = [np.array(p.center[:2], dtype=float) for p in pockets]
        if not pocket_positions: return 0.0
        BALL_RADIUS = 0.028575
        remaining = []
        for bid in targets:
            b = shot.balls.get(bid)
            if b is None or b.state.s == 4: continue
            pos = np.array(b.state.rvw[0][:2], dtype=float)
            remaining.append((bid, pos))
        if not remaining: return 0.0
        best_dist = None
        for bid, pos in remaining:
            pocket_dists = [(np.linalg.norm(pos - ppos), ppos) for ppos in pocket_positions]
            pocket_dists.sort(key=lambda x: x[0])
            for _, ppos in pocket_dists[:2]:
                vec_obj_to_pocket = ppos - pos
                dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
                if dist_obj_to_pocket < 1e-6: continue
                unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
                ghost_pos = pos - unit_vec * (2 * BALL_RADIUS)
                if self._is_path_blocked_simple(cue_pos, ghost_pos, shot.balls, ignore_ids={bid, 'cue'}): continue
                dist_cue_to_ghost = np.linalg.norm(ghost_pos - cue_pos)
                if best_dist is None or dist_cue_to_ghost < best_dist: best_dist = dist_cue_to_ghost
        if best_dist is None: return 0.0
        return float(max(0.0, 80.0 - best_dist * 800.0))

    def _sanitize_action(self, action):
        if action is None or not isinstance(action, dict): return None
        required = ("V0", "phi", "theta", "a", "b")
        if any(k not in action for k in required): return None
        try:
            V0, phi, theta, a, b = float(action["V0"]), float(action["phi"]), float(action["theta"]), float(action["a"]), float(action["b"])
        except Exception: return None
        return {"V0": float(np.clip(V0, 0.5, 8.0)), "phi": float(phi % 360.0), "theta": float(np.clip(theta, 0.0, 90.0)), "a": float(np.clip(a, -0.5, 0.5)), "b": float(np.clip(b, -0.5, 0.5))}

    def _random_action(self):
        return {'V0': float(np.random.uniform(2.0, 5.0)), 'phi': float(np.random.uniform(0, 360)), 'theta': 2.0, 'a': 0.0, 'b': 0.0}

# 向后兼容别名
NewAgentPro = OptimizedNewAgentPro
