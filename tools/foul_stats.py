#!/usr/bin/env python3
"""统计 debug 日志中 BasicAgentPro 与 NewAgentPro 的犯规次数与类型。"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterable, Tuple


def canonical_agent(name: str) -> str:
    """将日志中的多种写法映射为统一的代理名称。"""
    name_lower = name.lower()
    if "basicagentpro" in name_lower:
        return "BasicAgentPro"
    if "optimizednewagentpro" in name_lower or "newagentpro" in name_lower:
        return "NewAgentPro"
    return name.strip()


FOUL_PATTERNS: Iterable[Tuple[str, str, str]] = (
    ("first_collision", "首次碰撞为对方球或黑八", "首次碰撞非己方/黑8"),
    ("no_rail", "无进球且母球和目标球均未碰库", "无进球且未碰库"),
    ("scratch", "白球落袋", "白球落袋"),
    ("wrong_8", "误打黑8", "误打黑8判负"),
)


def detect_foul_type(line: str) -> Tuple[str, str] | tuple[None, None]:
    """返回犯规类型 ID 与描述，若非犯规则返回 (None, None)。"""
    for key, pattern, label in FOUL_PATTERNS:
        if pattern in line:
            return key, label
    if "犯规" in line:
        return "other", "其他犯规"
    return None, None


def main(log_path: str) -> None:
    # stats[agent][role]["type"]，role ∈ {"A", "B"}
    stats: DefaultDict[str, Dict[str, Counter]] = defaultdict(lambda: {"A": Counter(), "B": Counter()})

    agent_a = agent_b = None
    current_player = None  # "A" 或 "B"

    game_re = re.compile(r"本局 Player A: ([^,，]+)")
    shot_re = re.compile(r"\[第\d+次击球\] player: ([AB])")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            match_game = game_re.search(line)
            if match_game:
                agent_a = canonical_agent(match_game.group(1))
                # 假设两名代理互为对手，B 即另一方
                agent_b = "NewAgentPro" if agent_a == "BasicAgentPro" else "BasicAgentPro"
                current_player = None
                continue

            match_shot = shot_re.search(line)
            if match_shot:
                current_player = match_shot.group(1)
                continue

            foul_key, foul_label = detect_foul_type(line)
            if not foul_key:
                continue

            if current_player not in ("A", "B"):
                # 无当前击球方信息时无法归属，跳过
                continue

            agent = agent_a if current_player == "A" else agent_b
            agent_stats = stats[agent][current_player]
            agent_stats["total"] += 1
            agent_stats[foul_label] += 1

    if not stats:
        print("未找到犯规记录")
        return

    for agent, role_map in stats.items():
        print(f"{agent}:")
        for role in ("A", "B"):
            counter = role_map[role]
            print(f"  作为 Player {role}: 总犯规 {counter['total']}")
            for key in (label for _, _, label in FOUL_PATTERNS):
                if counter[key]:
                    print(f"    - {key}: {counter[key]}")
            if counter["其他犯规"]:
                print(f"    - 其他犯规: {counter['其他犯规']}")


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "debug.log"
    main(log_file)

