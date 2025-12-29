#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import time as _time

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress common serial-mode warning noise from sklearn/joblib in restricted environments.
warnings.filterwarnings(
    "ignore",
    message=r".*joblib will operate in serial mode.*",
    category=UserWarning,
)

from poolenv import PoolEnv
from utils import set_random_seed
from agents import BasicAgent, BasicAgentPro, NewAgent


@dataclass
class GameRecord:
    game_index: int
    target_ball: str
    env_player_a_agent: str
    env_player_b_agent: str
    env_player_a_role: str
    env_player_b_role: str
    winner_env: str
    winner_agent: str
    winner_role: str
    duration_s: float
    n_shots: int
    decision_time_s_mean: float
    decision_time_s_p95: float
    decision_time_s_max: float


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round(0.95 * (len(values_sorted) - 1)))
    return float(values_sorted[idx])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

class _Tee:
    def __init__(self, *streams, flush_seconds: float = 10.0):
        self._streams = streams
        self._flush_seconds = float(flush_seconds)
        self._last_flush = _time.monotonic()

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        now = _time.monotonic()
        if (now - self._last_flush) >= self._flush_seconds:
            self.flush()
            self._last_flush = now
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


class _PeriodicFlush:
    """Wrap a stream and force flush every N seconds (useful for long-running logs)."""

    def __init__(self, stream, flush_seconds: float = 10.0):
        self._stream = stream
        self._flush_seconds = float(flush_seconds)
        self._last_flush = _time.monotonic()

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        now = _time.monotonic()
        if (now - self._last_flush) >= self._flush_seconds:
            try:
                self._stream.flush()
            except Exception:
                pass
            self._last_flush = now
        return n

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass


def _agent_from_name(name: str, config: dict[str, Any] | None) -> Any:
    name = name.strip()
    if name == "BasicAgent":
        return BasicAgent()
    if name == "BasicAgentPro":
        return BasicAgentPro()
    if name == "NewAgent":
        return NewAgent(config=config)
    raise ValueError(f"Unknown agent name: {name}")


def _run_one_game(
    env: PoolEnv,
    agent_env_a: Any,
    agent_env_b: Any,
    target_ball: str,
    role_env_a: str,
    role_env_b: str,
    on_shot: Callable[[int, str, float], None] | None = None,
) -> GameRecord:
    env.reset(target_ball=target_ball)
    mapping = {"A": agent_env_a, "B": agent_env_b}
    decision_times: list[float] = []
    n_shots = 0

    game_start = time.perf_counter()
    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        n_shots += 1
        if on_shot is not None:
            try:
                on_shot(n_shots, player, time.perf_counter() - game_start)
            except Exception:
                pass
        t0 = time.perf_counter()
        action = mapping[player].decision(*obs)
        decision_times.append(time.perf_counter() - t0)
        env.take_shot(action)
        done, info = env.get_done()
        if done:
            duration = time.perf_counter() - game_start
            winner_env = info["winner"]
            winner_agent = (
                mapping[winner_env].__class__.__name__ if winner_env in mapping else "SAME"
            )
            if winner_env == "A":
                winner_role = role_env_a
            elif winner_env == "B":
                winner_role = role_env_b
            else:
                winner_role = "SAME"
            return GameRecord(
                game_index=-1,
                target_ball=target_ball,
                env_player_a_agent=agent_env_a.__class__.__name__,
                env_player_b_agent=agent_env_b.__class__.__name__,
                env_player_a_role=role_env_a,
                env_player_b_role=role_env_b,
                winner_env=winner_env,
                winner_agent=winner_agent,
                winner_role=winner_role,
                duration_s=float(duration),
                n_shots=int(n_shots),
                decision_time_s_mean=float(sum(decision_times) / max(1, len(decision_times))),
                decision_time_s_p95=_p95(decision_times),
                decision_time_s_max=float(max(decision_times) if decision_times else 0.0),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tournament between two agents.")
    parser.add_argument("--agent-a", required=True, choices=["BasicAgent", "BasicAgentPro", "NewAgent"])
    parser.add_argument("--agent-b", required=True, choices=["BasicAgent", "BasicAgentPro", "NewAgent"])
    parser.add_argument("--agent-a-config", type=str, default=None, help="JSON config for NewAgent (if used).")
    parser.add_argument("--agent-b-config", type=str, default=None, help="JSON config for NewAgent (if used).")
    parser.add_argument("--n-games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-seed", action="store_true", help="Use deterministic RNG for reproducibility.")
    parser.add_argument(
        "--target-rotation",
        type=str,
        default="solid,solid,stripe,stripe",
        help="Comma-separated target_ball rotation per game.",
    )
    parser.add_argument(
        "--swap-every-game",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap which agent is env player A each game.",
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=180.0,
        help="Reference game-time budget in seconds (used to compute over-budget rate; not necessarily a hard cap).",
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress stdout/stderr during simulation.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=sys.stderr.isatty(),
        help="Show progress bars (requires tqdm; falls back gracefully).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path. When set, always write logs to the file; in non-quiet mode logs are tee'd to terminal.",
    )
    parser.add_argument(
        "--log-flush-seconds",
        type=float,
        default=10.0,
        help="Force flush log output at this interval (seconds).",
    )
    parser.add_argument("--out", required=True, type=str, help="Output directory (will be created).")
    args = parser.parse_args()

    # Suppress common serial-mode warning noise from sklearn/joblib in restricted environments.
    warnings.filterwarnings(
        "ignore",
        message=r".*joblib will operate in serial mode.*",
        category=UserWarning,
    )

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    config_a = _load_json(Path(args.agent_a_config)) if args.agent_a_config else None
    config_b = _load_json(Path(args.agent_b_config)) if args.agent_b_config else None

    if args.fixed_seed:
        set_random_seed(enable=True, seed=args.seed)
    else:
        set_random_seed(enable=False, seed=args.seed)

    env = PoolEnv()
    rotation = [s.strip() for s in args.target_rotation.split(",") if s.strip()]
    if not rotation:
        raise ValueError("Empty --target-rotation")

    records: list[GameRecord] = []
    wins_by_role = {"agent_a": 0, "agent_b": 0, "SAME": 0}
    over_budget = 0

    log_path = Path(args.log_file).expanduser().resolve() if args.log_file else None
    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write("\n" + "=" * 80 + "\n")
            log_handle.write(f"[run_match] {datetime.now().isoformat(timespec='seconds')}\n")
            log_handle.write(json.dumps({"argv": sys.argv}, ensure_ascii=False) + "\n")
            log_handle.flush()

        if args.quiet:
            if log_handle is not None:
                stdout_target = _PeriodicFlush(log_handle, flush_seconds=args.log_flush_seconds)
                stderr_target = _PeriodicFlush(log_handle, flush_seconds=args.log_flush_seconds)
            else:
                stdout_target = open(os.devnull, "w", encoding="utf-8")
                stderr_target = open(os.devnull, "w", encoding="utf-8")
        else:
            if log_handle is not None:
                stdout_target = _Tee(orig_stdout, log_handle, flush_seconds=args.log_flush_seconds)
                stderr_target = _Tee(orig_stderr, log_handle, flush_seconds=args.log_flush_seconds)
            else:
                stdout_target = orig_stdout
                stderr_target = orig_stderr

        redirect_out_ctx = contextlib.redirect_stdout(stdout_target) if stdout_target is not orig_stdout else contextlib.nullcontext()
        redirect_err_ctx = contextlib.redirect_stderr(stderr_target) if stderr_target is not orig_stderr else contextlib.nullcontext()

        with redirect_out_ctx, redirect_err_ctx:
            agent_a = _agent_from_name(args.agent_a, config_a)
            agent_b = _agent_from_name(args.agent_b, config_b)

            iterator = range(args.n_games)
            use_progress = bool(args.progress and tqdm is not None)
            bar = (
                tqdm(
                    iterator,
                    desc="Games",
                    unit="game",
                    dynamic_ncols=True,
                    mininterval=0.5,
                    file=orig_stderr,  # keep progress visible even if stderr is redirected
                )
                if use_progress
                else iterator
            )

            for i in bar:
                if args.fixed_seed:
                    # Keep deterministic but vary per game to reduce correlation.
                    set_random_seed(enable=True, seed=args.seed + i)

                target_ball = rotation[i % len(rotation)]
                if args.swap_every_game and (i % 2 == 1):
                    agent_env_a, agent_env_b = agent_b, agent_a
                    role_env_a, role_env_b = "agent_b", "agent_a"
                else:
                    agent_env_a, agent_env_b = agent_a, agent_b
                    role_env_a, role_env_b = "agent_a", "agent_b"

                def _on_shot(shot_idx: int, player: str, elapsed_s: float) -> None:
                    if not use_progress:
                        return
                    bar.set_postfix(
                        game=f"{i+1}/{args.n_games}",
                        shot=shot_idx,
                        turn=player,
                        elapsed=f"{elapsed_s:.0f}s",
                    )

                rec = _run_one_game(
                    env,
                    agent_env_a,
                    agent_env_b,
                    target_ball,
                    role_env_a,
                    role_env_b,
                    on_shot=_on_shot if use_progress else None,
                )
                rec.game_index = i
                records.append(rec)

                if use_progress:
                    if rec.duration_s > args.time_limit_s:
                        over_budget += 1
                    wins_by_role[rec.winner_role] = wins_by_role.get(rec.winner_role, 0) + 1
                    bar.set_postfix(
                        win_b=wins_by_role.get("agent_b", 0),
                        win_a=wins_by_role.get("agent_a", 0),
                        over=f"{over_budget}/{len(records)}",
                        avg=f"{sum(r.duration_s for r in records)/len(records):.1f}s",
                    )
    finally:
        try:
            if log_handle is not None:
                log_handle.flush()
                log_handle.close()
        except Exception:
            pass

    if not (args.progress and tqdm is not None):
        for rec in records:
            if rec.duration_s > args.time_limit_s:
                over_budget += 1
            wins_by_role[rec.winner_role] = wins_by_role.get(rec.winner_role, 0) + 1

    durations = [r.duration_s for r in records]
    summary = {
        "match": {
            "agent_a": args.agent_a,
            "agent_b": args.agent_b,
            "agent_a_config": args.agent_a_config,
            "agent_b_config": args.agent_b_config,
            "n_games": args.n_games,
            "target_rotation": rotation,
            "swap_every_game": args.swap_every_game,
            "time_limit_s": args.time_limit_s,
            "fixed_seed": args.fixed_seed,
            "seed": args.seed,
        },
        "results": {
            "wins_by_role": wins_by_role,
            "win_rate_agent_a": wins_by_role.get("agent_a", 0) / max(1, args.n_games),
            "win_rate_agent_b": wins_by_role.get("agent_b", 0) / max(1, args.n_games),
            "draw_rate": wins_by_role.get("SAME", 0) / max(1, args.n_games),
            "over_budget_rate": over_budget / max(1, args.n_games),
            "budget_s": float(args.time_limit_s),
            "avg_game_time_s": float(sum(durations) / max(1, len(durations))),
            "median_game_time_s": float(sorted(durations)[len(durations) // 2]) if durations else 0.0,
            "p95_game_time_s": _p95(durations),
            "max_game_time_s": float(max(durations) if durations else 0.0),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }

    (outdir / "games.jsonl").write_text(
        "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
