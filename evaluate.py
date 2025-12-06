"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为40局来计算胜率）
3. 运行脚本查看结果
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import argparse

from poolenv import PoolEnv
from agent import Agent, BasicAgent, NewAgent

parser = argparse.ArgumentParser(description="Evaluate BasicAgent vs NewAgent")
parser.add_argument("--games", type=int, default=40, help="对战局数（默认40，与评分一致）")
parser.add_argument("--logdir", default="logs", help="日志目录（默认为 logs/，传空字符串即可禁用）")
args = parser.parse_args()

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = args.games

agent_a = BasicAgent()
agent_b = NewAgent()

players = [agent_a, agent_b]  # 用于切换先后手
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # 轮换球型

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
log_path = ""
if args.logdir:
    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"eval-{timestamp}.log")
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write(f"[INFO] Evaluate at {timestamp}, games={n_games}\n")
        log_fp.write(f"[INFO] AgentA={agent_a.__class__.__name__}, AgentB={agent_b.__class__.__name__}\n")
        log_fp.flush()

def log_write(line: str) -> None:
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(line)
        if not line.endswith("\n"):
            fp.write("\n")

for i in range(n_games): 
    print()
    print(f"------- 第 {i} 局比赛开始 -------")
    env.reset(target_ball=target_ball_choice[i % 4])
    player_a_agent = players[i % 2]
    player_b_agent = players[(i + 1) % 2]
    print(f"本局 Player A: {player_a_agent.__class__.__name__}, 目标球型: {target_ball_choice[i % 4]}")
    if log_path:
        log_write(f"[GAME {i}] PlayerA={player_a_agent.__class__.__name__}, targets={target_ball_choice[i % 4]}\n")
    while True:
        player = env.get_curr_player()
        acting_agent = player_a_agent if player == 'A' else player_b_agent
        agent_label = acting_agent.__class__.__name__
        print(f"[第{env.hit_count}次击球] player: {player} ({agent_label})")
        if log_path:
            log_write(f"[SHOT {env.hit_count}] player={player} agent={agent_label}\n")
        obs = env.get_observation(player)
        action = acting_agent.decision(*obs)
        action_log = {k: round(float(v), 3) for k, v in action.items()}
        if log_path:
            log_write(f"    action={action_log}\n")
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            log_notes = []
            if step_info.get('FOUL_FIRST_HIT'):
                msg = "本杆判罚：首次接触对方球或黑8，直接交换球权。"
                print(msg)
                log_notes.append(msg)
            if step_info.get('NO_POCKET_NO_RAIL'):
                msg = "本杆判罚：无进球且母球或目标球未碰库，直接交换球权。"
                print(msg)
                log_notes.append(msg)
            if step_info.get('NO_HIT'):
                msg = "本杆判罚：白球未接触任何球，直接交换球权。"
                print(msg)
                log_notes.append(msg)
            if step_info.get('ME_INTO_POCKET'):
                msg = f"我方球入袋：{step_info['ME_INTO_POCKET']}"
                print(msg)
                log_notes.append(msg)
            if step_info.get('ENEMY_INTO_POCKET'):
                msg = f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}"
                print(msg)
                log_notes.append(msg)
            if log_path:
                summary_keys = [
                    'ME_INTO_POCKET',
                    'ENEMY_INTO_POCKET',
                    'WHITE_BALL_INTO_POCKET',
                    'BLACK_BALL_INTO_POCKET',
                    'FOUL_FIRST_HIT',
                    'NO_POCKET_NO_RAIL',
                    'NO_HIT'
                ]
                clean_info = {}
                for key in summary_keys:
                    if key not in step_info:
                        continue
                    value = step_info[key]
                    if isinstance(value, list):
                        if value:
                            clean_info[key] = value
                    elif isinstance(value, bool):
                        if value:
                            clean_info[key] = True
                    elif value:
                        clean_info[key] = value
                if clean_info:
                    log_write(f"    step_info={clean_info}\n")
                for note in log_notes:
                    log_write(f"    note={note}\n")
        if done:
            # 统计结果（player A/B 转换为 agent A/B） 
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            if log_path:
                log_write(f"[GAME {i} DONE] info={info}\n")
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\n最终结果：", results)
if log_path:
    log_write(f"[RESULT] {results}\n")
