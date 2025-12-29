#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a suite of matches and aggregate summaries.")
    parser.add_argument("--suite", required=True, type=str, help="Path to suite JSON.")
    parser.add_argument("--out", required=True, type=str, help="Output directory.")
    parser.add_argument("--n-games", type=int, default=None, help="Override n_games for all matches.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for all matches.")
    parser.add_argument("--fixed-seed", action="store_true", help="Use deterministic RNG.")
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress simulation logs.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=sys.stderr.isatty(),
        help="Show progress over matches (requires tqdm; falls back gracefully).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path passed through to each match.",
    )
    parser.add_argument(
        "--log-flush-seconds",
        type=float,
        default=10.0,
        help="Force flush log output at this interval (seconds).",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite)
    suite = _load_json(suite_path)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    matches = suite.get("matches", [])
    if not matches:
        raise ValueError("Suite has no matches")

    aggregated: dict[str, Any] = {"suite": suite.get("name", suite_path.name), "matches": []}

    use_progress = bool(args.progress and tqdm is not None)
    outer = tqdm(matches, desc="Matches", unit="match", dynamic_ncols=True) if use_progress else matches

    for match in outer:
        match_id = match["id"]
        match_out = outdir / match_id
        match_out.mkdir(parents=True, exist_ok=True)

        argv = [
            sys.executable,
            str((Path(__file__).resolve().parent / "run_match.py").resolve()),
            "--agent-a",
            match["agent_a"],
            "--agent-b",
            match["agent_b"],
            "--out",
            str(match_out),
        ]

        if match.get("agent_a_config"):
            argv += ["--agent-a-config", match["agent_a_config"]]
        if match.get("agent_b_config"):
            argv += ["--agent-b-config", match["agent_b_config"]]

        n_games = args.n_games if args.n_games is not None else match.get("n_games", 120)
        seed = args.seed if args.seed is not None else match.get("seed", 42)
        argv += ["--n-games", str(n_games), "--seed", str(seed)]

        if args.fixed_seed or match.get("fixed_seed", False):
            argv += ["--fixed-seed"]
        if args.quiet or match.get("quiet", True):
            argv += ["--quiet"]
        if args.progress:
            argv += ["--progress"]
        else:
            argv += ["--no-progress"]

        if match.get("target_rotation"):
            argv += ["--target-rotation", match["target_rotation"]]
        if match.get("time_limit_s"):
            argv += ["--time-limit-s", str(match["time_limit_s"])]

        if args.log_file:
            argv += ["--log-file", args.log_file]
            argv += ["--log-flush-seconds", str(args.log_flush_seconds)]

        if use_progress:
            outer.set_postfix(match=match_id)
        subprocess.run(argv, check=True)

        summary_path = match_out / "summary.json"
        summary = _load_json(summary_path)
        aggregated["matches"].append({"id": match_id, "out": str(match_out), "summary": summary})

    _write_json(outdir / "summary.json", aggregated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
