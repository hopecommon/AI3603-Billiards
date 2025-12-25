#!/usr/bin/env python3
import re
import statistics

# 读取日志
with open('debug1.log', 'r') as f:
    lines = f.readlines()

games = []
current_game = None
current_player_a = None
current_winner = None
current_duration = None

for line in lines:
    line = line.strip()
    
    # 新游戏开始
    m = re.match(r'-+ 第 (\d+) 局比赛开始', line)
    if m:
        current_game = int(m.group(1))
        current_player_a = None
        current_winner = None
        current_duration = None
        continue
    
    # Player A信息
    m = re.match(r'本局 Player A: (\w+),', line)
    if m:
        current_player_a = m.group(1)
        continue
    
    # 胜者
    m = re.match(r'🏆 Player ([AB])', line)
    if m:
        current_winner = m.group(1)
        continue
    
    # 耗时
    m = re.match(r'本局耗时: ([0-9.]+)秒', line)
    if m and current_game is not None:
        current_duration = float(m.group(1))
        # 记录这局
        games.append({
            'game': current_game,
            'player_a': current_player_a,
            'winner': current_winner,
            'duration': current_duration
        })

# 统计
print(f"\n{'='*60}")
print("DEBUG1.LOG 分析结果")
print('='*60)
print(f"完成局数: {len(games)}\n")

# 按Player A分组统计
basicagentpro_wins = sum(1 for g in games if g['player_a'] == 'BasicAgentPro' and g['winner'] == 'A')
basicagentpro_loss = sum(1 for g in games if g['player_a'] == 'BasicAgentPro' and g['winner'] == 'B')
newagent_wins = sum(1 for g in games if g['player_a'] == 'NewAgent' and g['winner'] == 'A')
newagent_loss = sum(1 for g in games if g['player_a'] == 'NewAgent' and g['winner'] == 'B')

print("当Player A时的胜负:")
print(f"  BasicAgentPro: {basicagentpro_wins}胜 {basicagentpro_loss}负")
print(f"  NewAgent:      {newagent_wins}胜 {newagent_loss}负")
print()

# 根据奇偶轮换计算真实胜负
# 偶数局: Player A = BasicAgentPro (opponent)
# 奇数局: Player A = NewAgent (our agent)
opponent_wins = 0
our_wins = 0

for g in games:
    if g['game'] % 2 == 0:  # 偶数局
        if g['winner'] == 'A':
            opponent_wins += 1
        else:
            our_wins += 1
    else:  # 奇数局
        if g['winner'] == 'A':
            our_wins += 1
        else:
            opponent_wins += 1

total = opponent_wins + our_wins
print(f"实际胜负（考虑奇偶轮换）:")
print(f"  BasicAgentPro (对手):  {opponent_wins:2d}胜 ({opponent_wins/total*100:5.1f}%)")
print(f"  NewAgent (我方):       {our_wins:2d}胜 ({our_wins/total*100:5.1f}%)")
print()

# 时间统计
durations = [g['duration'] for g in games]
print(f"时间统计:")
print(f"  平均:   {statistics.mean(durations):6.2f}秒")
print(f"  中位数: {statistics.median(durations):6.2f}秒")
print(f"  最快:   {min(durations):6.2f}秒")
print(f"  最慢:   {max(durations):6.2f}秒")
print(f"  标准差: {statistics.stdev(durations):6.2f}秒")
print('='*60)
