#!/usr/bin/env python3
# python analyze_eval_log.py debug.log 
import argparse
import re
from collections import defaultdict
from statistics import mean


SHOT_RE = re.compile(r"^\[第(\d+)次击球\] player: ([AB]) \(([^)]+)\)")
GAME_RE = re.compile(r"^------- 第 (\d+) 局比赛开始 -------")
PLAYER_A_RE = re.compile(r"^本局 Player A: ([^,]+),")
STEP_TIME_RE = re.compile(r"^\[(Step \d+)\].*耗时:\s*([0-9.]+)s")
SUMMARY_TIME_RE = re.compile(r"^\[总结\] 总耗时:\s*([0-9.]+)s")
GAME_DURATION_RE = re.compile(r"^本局耗时:\s*([0-9.]+)秒")
MAX_HIT_WINNER_RE = re.compile(r"胜者：\s*(A|B|SAME)")

FOUL_PATTERNS = {
    "WHITE_AND_EIGHT": re.compile(r"白球和黑8同时落袋"),
    "ILLEGAL_EIGHT": re.compile(r"误打黑8"),
    "CUE_POCKETED": re.compile(r"白球落袋"),
    "FOUL_FIRST_HIT": re.compile(r"首次碰撞为对方球或黑八"),
    "NO_POCKET_NO_RAIL": re.compile(r"无进球且母球和目标球均未碰库"),
    "NO_HIT": re.compile(r"白球未接触任何球"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze evaluate.py stdout log (tee output).")
    ap.add_argument("log", help="Path to debug_codex.log or similar")
    ap.add_argument(
        "--by-parity",
        action="store_true",
        help="Also attribute stats to AGENT_A/AGENT_B using evaluate.py parity swap (game_idx%%2).",
    )
    args = ap.parse_args()

    stats = defaultdict(lambda: defaultdict(int))
    bucket_stats = defaultdict(lambda: defaultdict(int))  # bucket_stats['AGENT_A'/'AGENT_B'][key] -> int
    step_times = defaultdict(lambda: defaultdict(list))  # step_times[agent][step] -> [sec]
    bucket_step_times = defaultdict(lambda: defaultdict(list))
    decision_times = defaultdict(list)  # decision_times[agent] -> [sec]
    bucket_decision_times = defaultdict(list)
    game_outcomes = []  # [(game_idx, winner_player, winner_agent, duration)]
    bucket_wins = defaultdict(int)  # AGENT_A/AGENT_B/SAME/UNKNOWN
    agent_wins = defaultdict(int)   # agent_name/UNKNOWN

    current_game = None
    current_shot = None
    current_player = None
    current_agent = None
    current_bucket = None
    current_player_agents = {}  # {'A': agent_name, 'B': agent_name}
    current_winner_player = None
    current_duration = None

    # Winner lines from poolenv.py
    winner_re = re.compile(r"^🏆 Player ([AB])")

    with open(args.log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = GAME_RE.match(line)
            if m:
                current_game = int(m.group(1))
                current_shot = None
                current_player = None
                current_agent = None
                current_bucket = None
                current_player_agents = {}
                current_winner_player = None
                current_duration = None
                continue

            m = PLAYER_A_RE.match(line)
            if m:
                current_player_agents["A"] = m.group(1).strip()
                continue

            m = SHOT_RE.match(line)
            if m:
                current_shot = int(m.group(1))
                current_player = m.group(2)
                current_agent = m.group(3)
                stats[current_agent]["SHOTS"] += 1
                if current_player not in current_player_agents:
                    current_player_agents[current_player] = current_agent
                if args.by_parity and current_game is not None:
                    # evaluate.py:
                    # player_a_agent = players[i % 2]
                    # player_b_agent = players[(i + 1) % 2]
                    if current_player == "A":
                        current_bucket = "AGENT_A" if (current_game % 2 == 0) else "AGENT_B"
                    else:
                        current_bucket = "AGENT_B" if (current_game % 2 == 0) else "AGENT_A"
                    bucket_stats[current_bucket]["SHOTS"] += 1
                continue

            if current_agent:
                if "无候选" in line:
                    stats[current_agent]["NO_CANDIDATE"] += 1
                    if current_bucket:
                        bucket_stats[current_bucket]["NO_CANDIDATE"] += 1

                m = STEP_TIME_RE.match(line)
                if m:
                    step_times[current_agent][m.group(1)].append(float(m.group(2)))
                    if current_bucket:
                        bucket_step_times[current_bucket][m.group(1)].append(float(m.group(2)))
                    continue

                m = SUMMARY_TIME_RE.match(line)
                if m:
                    decision_times[current_agent].append(float(m.group(1)))
                    if current_bucket:
                        bucket_decision_times[current_bucket].append(float(m.group(1)))
                    continue

                for key, pat in FOUL_PATTERNS.items():
                    if pat.search(line):
                        stats[current_agent][key] += 1
                        if current_bucket:
                            bucket_stats[current_bucket][key] += 1
                        break

            m = winner_re.match(line)
            if m and current_game is not None and current_winner_player is None:
                current_winner_player = m.group(1)
                continue

            m = MAX_HIT_WINNER_RE.search(line)
            if m and current_game is not None and current_winner_player is None:
                current_winner_player = m.group(1)
                continue

            m = GAME_DURATION_RE.match(line)
            if m and current_game is not None:
                current_duration = float(m.group(1))
                winner_agent = None
                if current_winner_player in ("A", "B"):
                    winner_agent = current_player_agents.get(current_winner_player)
                game_outcomes.append((current_game, current_winner_player, winner_agent, current_duration))
                if current_winner_player in ("A", "B", "SAME"):
                    agent_wins[winner_agent or "UNKNOWN"] += 1
                if args.by_parity and current_winner_player in ("A", "B", "SAME"):
                    if current_winner_player == "SAME":
                        bucket_wins["SAME"] += 1
                    else:
                        # winner bucket depends on (game_idx%2) swap:
                        # if winner is Player A: AGENT_A when i even else AGENT_B
                        # if winner is Player B: AGENT_B when i even else AGENT_A
                        if current_winner_player == "A":
                            bucket_wins["AGENT_A" if (current_game % 2 == 0) else "AGENT_B"] += 1
                        else:
                            bucket_wins["AGENT_B" if (current_game % 2 == 0) else "AGENT_A"] += 1
                continue

    print("== Per-agent foul counts ==")
    for agent in sorted(stats.keys()):
        s = stats[agent]
        print(
            f"- {agent}: shots={s['SHOTS']}, "
            f"illegal8={s['ILLEGAL_EIGHT']}, white+8={s['WHITE_AND_EIGHT']}, "
            f"scratch={s['CUE_POCKETED']}, first_hit={s['FOUL_FIRST_HIT']}, "
            f"no_rail={s['NO_POCKET_NO_RAIL']}, no_hit={s['NO_HIT']}, "
            f"no_candidate={s['NO_CANDIDATE']}"
        )

    if args.by_parity and bucket_stats:
        print("\n== By parity (AGENT_A/AGENT_B) ==")
        for bucket in ("AGENT_A", "AGENT_B"):
            s = bucket_stats[bucket]
            if not s:
                continue
            print(
                f"- {bucket}: shots={s['SHOTS']}, "
                f"illegal8={s['ILLEGAL_EIGHT']}, white+8={s['WHITE_AND_EIGHT']}, "
                f"scratch={s['CUE_POCKETED']}, first_hit={s['FOUL_FIRST_HIT']}, "
                f"no_rail={s['NO_POCKET_NO_RAIL']}, no_hit={s['NO_HIT']}, "
                f"no_candidate={s['NO_CANDIDATE']}"
            )

    print("\n== Optimized agent timing (from printed summaries) ==")
    for agent in sorted(decision_times.keys()):
        times = decision_times[agent]
        if not times:
            continue
        print(f"- {agent}: decisions={len(times)}, avg={mean(times):.3f}s, max={max(times):.3f}s")
        for step in sorted(step_times[agent].keys()):
            st = step_times[agent][step]
            if st:
                print(f"  - {step}: n={len(st)}, avg={mean(st):.3f}s, max={max(st):.3f}s")

    if args.by_parity and bucket_decision_times:
        print("\n== By parity timing ==")
        for bucket in ("AGENT_A", "AGENT_B"):
            times = bucket_decision_times.get(bucket, [])
            if not times:
                continue
            print(f"- {bucket}: decisions={len(times)}, avg={mean(times):.3f}s, max={max(times):.3f}s")
            for step in sorted(bucket_step_times[bucket].keys()):
                st = bucket_step_times[bucket][step]
                if st:
                    print(f"  - {step}: n={len(st)}, avg={mean(st):.3f}s, max={max(st):.3f}s")

    if game_outcomes:
        print("\n== Games ==")
        finished = len(game_outcomes)
        known_winner = sum(1 for _, w, _, _ in game_outcomes if w in ("A", "B", "SAME"))
        print(f"- games_finished={finished}, games_with_winner={known_winner}")

        by_player = defaultdict(int)
        by_agent = defaultdict(int)
        durations = []
        for _, w, wa, dur in game_outcomes:
            by_player[w] += 1
            by_agent[wa or "UNKNOWN"] += 1
            if dur is not None:
                durations.append(dur)

        print(f"- winners_by_player: {dict(sorted(by_player.items()))}")
        print(f"- winners_by_agent: {dict(sorted(by_agent.items()))}")
        if args.by_parity:
            print(f"- winners_by_parity: {dict(sorted(bucket_wins.items()))}")
        if durations:
            print(f"- game_duration: n={len(durations)}, avg={mean(durations):.2f}s, max={max(durations):.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
