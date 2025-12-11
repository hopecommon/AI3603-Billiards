"""
evaluate.py - Agent 评估脚本

功能：
- 让两个 Agent 进行多局对战
- 统计胜负和得分
- 支持切换先后手和球型分配

使用方式：
1. 修改 agent_b 为你设计的待测试的 Agent， 与课程提供的BasicAgent对打
2. 调整 n_games 设置对战局数（评分时设置为120局来计算胜率）
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
import time

# 导入必要的模块
from utils import set_random_seed
from poolenv import PoolEnv
from agent import BasicAgent, NewAgent       
from agent_optimized import OptimizedNewAgent  # 使用优化版

# 设置随机种子，enable=True 时使用固定种子，enable=False 时使用完全随机
# 根据需求，我们在这里统一设置随机种子，确保 agent 双方的全局击球扰动使用相同的随机状态
set_random_seed(enable=True, seed=42)

parser = argparse.ArgumentParser(description="Evaluate BasicAgent vs NewAgent")
parser.add_argument("--games", type=int, default=120, help="对战局数（默认40，与评分一致）")
parser.add_argument("--logdir", default="logs", help="日志目录（默认为 logs/，传空字符串即可禁用）")
args = parser.parse_args()

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = args.games
game_times = []  # 记录每局游戏时间

# 失误统计
foul_stats = {
    'white_ball_pocketed': 0,
    'illegal_eight_ball': 0,
    'white_and_eight': 0,
    'first_hit_foul': 0,
    'no_rail_foul': 0,
    'no_hit_foul': 0
}

agent_a = BasicAgent()
# agent_b = NewAgent()
agent_b = OptimizedNewAgent()

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
    game_start_time = time.time()  # 记录本局开始时间
    env.reset(target_ball=target_ball_choice[i % 4])
    player_a_agent = players[i % 2]
    player_b_agent = players[(i + 1) % 2]
    player_class = players[i % 2].__class__.__name__
    ball_type = target_ball_choice[i % 4]
    print(f"本局 Player A: {player_class}, 目标球型: {ball_type}")
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
        
        # 统计失误（无论是否done）
        if step_info.get('WHITE_BALL_INTO_POCKET'):
            if step_info.get('BLACK_BALL_INTO_POCKET'):
                foul_stats['white_and_eight'] += 1
            else:
                foul_stats['white_ball_pocketed'] += 1
        elif step_info.get('BLACK_BALL_INTO_POCKET'):
            # 检查是否合法打黑八（需要从环境获取信息）
            # 简化：如果黑八进袋且游戏未结束或输了，则是非法
            if done and info.get('winner') != player:
                foul_stats['illegal_eight_ball'] += 1
        
        if step_info.get('FOUL_FIRST_HIT'):
            foul_stats['first_hit_foul'] += 1
        if step_info.get('NO_POCKET_NO_RAIL'):
            foul_stats['no_rail_foul'] += 1
        if step_info.get('NO_HIT'):
            foul_stats['no_hit_foul'] += 1
        
        if not done:
            log_notes = []
            # poolenv中已有打印，无需再输出
            # if step_info.get('FOUL_FIRST_HIT'):
            #     print("本杆判罚：首次接触对方球或黑8，直接交换球权。")
            # if step_info.get('NO_POCKET_NO_RAIL'):
            #     print("本杆判罚：无进球且母球或目标球未碰库，直接交换球权。")
            # if step_info.get('NO_HIT'):
            #     print("本杆判罚：白球未接触任何球，直接交换球权。")
            # if step_info.get('ME_INTO_POCKET'):
            #     print(f"我方球入袋：{step_info['ME_INTO_POCKET']}")
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
            
            # 记录本局游戏时间
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            game_times.append(game_duration)
            print(f"本局耗时: {game_duration:.2f}秒")
            
            if log_path:
                log_write(f"[GAME {i} DONE] info={info}, duration={game_duration:.2f}s\n")
            break

# 计算分数：胜1分，负0分，平局0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

# 计算时间统计
if game_times:
    avg_game_time = sum(game_times) / len(game_times)
    total_time = sum(game_times)
    min_game_time = min(game_times)
    max_game_time = max(game_times)
    
    print("\n========== 时间统计 ==========")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每局: {avg_game_time:.2f}秒")
    print(f"最快一局: {min_game_time:.2f}秒")
    print(f"最慢一局: {max_game_time:.2f}秒")
    print("=" * 30)

# 失误统计
print("\n========== 失误统计 ==========")
print(f"白球进袋: {foul_stats['white_ball_pocketed']}次")
print(f"非法黑八: {foul_stats['illegal_eight_ball']}次")
print(f"白球+黑八: {foul_stats['white_and_eight']}次")
print(f"首球犯规: {foul_stats['first_hit_foul']}次")
print(f"无碰库犯规: {foul_stats['no_rail_foul']}次")
print(f"未击中犯规: {foul_stats['no_hit_foul']}次")
total_critical_fouls = (foul_stats['white_ball_pocketed'] + 
                        foul_stats['illegal_eight_ball'] + 
                        foul_stats['white_and_eight'])
print(f"致命失误总计: {total_critical_fouls}次")
if n_games > 0:
    print(f"致命失误率: {total_critical_fouls/n_games*100:.1f}%")
print("=" * 30)

print("\n最终结果：", results)
if log_path:
    if game_times:
        log_write(f"[TIME STATS] total={total_time:.2f}s, avg={avg_game_time:.2f}s, min={min_game_time:.2f}s, max={max_game_time:.2f}s\n")
    log_write(f"[FOUL STATS] {foul_stats}\n")
    log_write(f"[RESULT] {results}\n")
