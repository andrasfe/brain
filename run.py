#!/usr/bin/env python3
"""CLI entry point for the brain.

Usage:
    python run.py "your task here"
    python run.py                      # interactive prompt
    python run.py --quiet "task"       # suppress the cognitive trace
    python run.py --yes "task"         # auto-approve shell commands
"""
from __future__ import annotations

import argparse
import sys

from brain.config import load_config
from brain.orchestrator import Brain


def make_confirm(auto_yes: bool):
    def confirm(msg: str) -> bool:
        if auto_yes:
            print(f"[auto-approve] {msg}")
            return True
        try:
            ans = input(f"\n{msg}\nProceed? [y/N] ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")
    return confirm


def main() -> int:
    ap = argparse.ArgumentParser(description="brain — a multi-agent model of cognition")
    ap.add_argument("task", nargs="*", help="the task for the brain to perform")
    ap.add_argument("--quiet", action="store_true", help="hide the cognitive trace")
    ap.add_argument("--yes", action="store_true", help="auto-approve shell commands")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--persona", default=None,
                    help="override persona file path (default from config.yaml)")
    ap.add_argument("--scenario", default=None,
                    help="override scenario (neutral | calm_morning | deadline_night "
                         "| boring_afternoon | social_evening | sick_day)")
    ap.add_argument("--vanilla", action="store_true",
                    help="disable humanization (no affect, no world, no DMN) "
                         "for A/B comparison with the original brain")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for world ticker and DMN (reproducibility)")
    args = ap.parse_args()

    task = " ".join(args.task).strip() or input("Task: ").strip()
    if not task:
        print("No task given.", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    # CLI overrides go into raw so Brain.__init__ picks them up
    if args.persona is not None:
        cfg.raw["persona_path"] = args.persona
    if args.scenario is not None:
        cfg.raw["scenario"] = args.scenario
    humanize = (not args.vanilla) and bool(cfg.raw.get("humanize", True))

    log = (lambda _m: None) if args.quiet else (lambda m: print(m, flush=True))
    brain = Brain(cfg, confirm=make_confirm(args.yes), log=log,
                  humanize=humanize, seed=args.seed)
    try:
        answer = brain.run(task)
    finally:
        brain.close()

    print("\n" + "═" * 60)
    print("FINAL ANSWER")
    print("═" * 60)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
