#!/usr/bin/env python3
"""
Analyze evaluation logs produced by evaluate.py.

Usage:
    python eval/analyze_log.py logs/eval-20251206-220816.log
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
import os
import re


SHOT_RE = re.compile(r"\[SHOT (\d+)] player=([AB]) agent=([A-Za-z0-9_]+)")
GAME_START_RE = re.compile(r"\[GAME (\d+)] PlayerA=([A-Za-z0-9_]+)")
GAME_DONE_RE = re.compile(r"\[GAME (\d+) DONE] info=\{'winner': '([ABSAME]+)', 'hit_count': (\d+)\}")


def parse_log(path: str):
    games = {}
    shots_by_agent = Counter()
    events_by_agent = defaultdict(Counter)
    pocketed_balls_by_agent = defaultdict(lambda: Counter())
    step_summary_counts = Counter()
    hit_counts = []
    winners = Counter()

    current_game = None
    game_agents = defaultdict(lambda: {'A': None, 'B': None})
    last_shot = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')

            m = GAME_START_RE.match(line)
            if m:
                game_id = int(m.group(1))
                player_a_name = m.group(2)
                current_game = game_id
                game_agents[game_id]['A'] = player_a_name
                continue

            m = GAME_DONE_RE.match(line)
            if m:
                game_id = int(m.group(1))
                winner = m.group(2)
                hit_count = int(m.group(3))
                hit_counts.append(hit_count)

                player_map = game_agents.get(game_id, {})
                if winner == 'A':
                    winners[player_map.get('A', 'PlayerA')] += 1
                elif winner == 'B':
                    winners[player_map.get('B', 'PlayerB')] += 1
                elif winner == 'SAME':
                    winners['SAME'] += 1
                games[game_id] = {
                    'winner': winner,
                    'hit_count': hit_count,
                    'agents': dict(player_map)
                }
                current_game = None
                last_shot = None
                continue

            m = SHOT_RE.match(line)
            if m:
                shot_id = int(m.group(1))
                player = m.group(2)
                agent_name = m.group(3)
                if current_game is None:
                    # If log order is unexpected, skip.
                    continue
                game_agents[current_game][player] = agent_name
                shots_by_agent[agent_name] += 1
                last_shot = {
                    'game': current_game,
                    'shot': shot_id,
                    'player': player,
                    'agent': agent_name
                }
                continue

            striped = line.strip()
            if striped.startswith("step_info=") and last_shot is not None:
                try:
                    data = ast.literal_eval(striped.split("=", 1)[1].strip())
                except Exception:
                    continue

                agent = last_shot['agent']
                for key, value in data.items():
                    step_summary_counts[key] += 1
                    if isinstance(value, list):
                        if value:
                            events_by_agent[key][agent] += 1
                            pocketed_balls_by_agent[key][agent] += len(value)
                    elif value:
                        events_by_agent[key][agent] += 1
                continue

    return {
        'games': games,
        'shots_by_agent': shots_by_agent,
        'events_by_agent': events_by_agent,
        'pocketed_balls_by_agent': pocketed_balls_by_agent,
        'step_summary_counts': step_summary_counts,
        'hit_counts': hit_counts,
        'winners': winners
    }


def build_summary(stats, path):
    return {
        'log': os.path.basename(path),
        'total_games': len(stats['games']),
        'hit_counts': list(stats['hit_counts']),
        'winners': dict(stats['winners']),
        'shots_by_agent': dict(stats['shots_by_agent']),
        'events': {event: dict(agent_counts) for event, agent_counts in stats['events_by_agent'].items()}
    }


def format_summary(summary):
    total_games = summary['total_games']
    winners = summary['winners']
    hit_counts = summary['hit_counts']
    shots_by_agent = summary['shots_by_agent']

    lines = []
    lines.append(f"# Log: {summary['log']}")
    lines.append(f"总局数: {total_games}")
    if total_games:
        avg_hits = sum(hit_counts) / total_games
        lines.append(f"平均击球数: {avg_hits:.2f} (最少 {min(hit_counts)}, 最多 {max(hit_counts)})")
    if winners:
        lines.append("胜场统计:")
        for agent, cnt in winners.items():
            lines.append(f"  - {agent}: {cnt}")

    lines.append("击球次数（所有 shots 行）:")
    for agent, cnt in shots_by_agent.items():
        lines.append(f"  - {agent}: {cnt}")

    events = summary['events']
    if events:
        lines.append("事件统计（按出杆方）：")
        for event, agent_counts in events.items():
            lines.append(f"  * {event}:")
            for agent, cnt in agent_counts.items():
                lines.append(f"      - {agent}: {cnt}")

    return "\n".join(lines)


def resolve_logfile(path: str | None) -> str:
    if path:
        return path
    logs_dir = "logs"
    try:
        candidates = sorted(
            (os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")),
            key=os.path.getmtime,
            reverse=True
        )
    except FileNotFoundError:
        raise SystemExit("未找到 logs 目录。请指定日志文件路径。")
    if not candidates:
        raise SystemExit("logs 目录为空，请先运行 evaluate.py 生成日志。")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="分析 evaluate.py 生成的日志文件。")
    parser.add_argument("logfile", nargs="?", help="日志路径（默认自动取 logs/ 下最新的）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    logfile = resolve_logfile(args.logfile)
    stats = parse_log(logfile)
    summary = build_summary(stats, logfile)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
