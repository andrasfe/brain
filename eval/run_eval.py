#!/usr/bin/env python3
"""Run the MAP-paper comparison: multi-agent brain vs. qwen-alone (zero-shot)
on Tower of Hanoi and graph traversal.

Usage:
    python -m eval.run_eval [--n 10] [--disks 3] [--out eval/results]

Writes results.json and results.md to the output dir, and prints a table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from eval import hanoi, graph
from eval.solvers import solve_with_brain, solve_with_qwen


def _pct(xs):
    return round(100.0 * sum(xs) / len(xs), 1) if xs else 0.0


def run_hanoi(n_instances: int, disks: int, log):
    instances = hanoi.generate_instances(n=disks, k=n_instances)
    rows = []
    for i, init in enumerate(instances, 1):
        zp = hanoi.zero_shot_prompt(init, disks)
        bt = hanoi.brain_task(init, disks)

        log(f"[hanoi {i}/{len(instances)}] qwen…")
        q_text, q_calls = solve_with_qwen(zp)
        q = hanoi.score(init, disks, q_text)

        log(f"[hanoi {i}/{len(instances)}] brain…")
        b_text, b_calls = solve_with_brain(bt)
        b = hanoi.score(init, disks, b_text)

        rows.append({
            "instance": i, "initial": hanoi.render_state(init).replace("\n", " | "),
            "optimal": q.optimal,
            "qwen": {**asdict(q), "calls": q_calls},
            "brain": {**asdict(b), "calls": b_calls},
        })
        log(f"   qwen solved={q.solved} invalid={q.n_invalid} | "
            f"brain solved={b.solved} invalid={b.n_invalid}")
    return rows


def run_graph(n_instances: int, log):
    instances = graph.generate_instances(k=n_instances)
    rows = []
    for i, inst in enumerate(instances, 1):
        zp = graph.zero_shot_prompt(inst)
        bt = graph.brain_task(inst)

        log(f"[graph {i}/{len(instances)}] qwen…")
        q_text, q_calls = solve_with_qwen(zp)
        q = graph.score(inst, q_text)

        log(f"[graph {i}/{len(instances)}] brain…")
        b_text, b_calls = solve_with_brain(bt)
        b = graph.score(inst, b_text)

        rows.append({
            "instance": i, "start": inst.start, "goal": inst.goal,
            "optimal": inst.optimal,
            "qwen": {**asdict(q), "calls": q_calls},
            "brain": {**asdict(b), "calls": b_calls},
        })
        log(f"   qwen solved={q.solved} invalidEdges={q.n_invalid_edges} | "
            f"brain solved={b.solved} invalidEdges={b.n_invalid_edges}")
    return rows


def summarize(rows, task):
    out = {}
    for cond in ("qwen", "brain"):
        solved = [r[cond]["solved"] for r in rows]
        calls = [r[cond]["calls"] for r in rows]
        if task == "hanoi":
            inv_moves = sum(r[cond]["n_invalid"] for r in rows)
            tot_moves = sum(r[cond]["n_moves"] + r[cond]["n_invalid"] for r in rows)
            reached = [r[cond]["reached_goal"] for r in rows]
            out[cond] = {
                "pct_solved": _pct(solved),
                "pct_reached_goal": _pct(reached),
                "pct_invalid_moves": round(100.0 * inv_moves / tot_moves, 1) if tot_moves else 0.0,
                "avg_calls": round(statistics.mean(calls), 1),
            }
        else:
            inv = sum(r[cond]["n_invalid_edges"] for r in rows)
            valid = [r[cond]["valid_path"] for r in rows]
            out[cond] = {
                "pct_solved": _pct(solved),
                "pct_valid_path": _pct(valid),
                "total_invalid_edges": inv,
                "avg_calls": round(statistics.mean(calls), 1),
            }
    return out


def to_markdown(hsum, gsum, n, disks):
    L = []
    L.append("# Brain (multi-agent) vs. qwen-alone — MAP-paper planning tasks\n")
    L.append(f"_{n} instances per task. Model: qwen/qwen3.7-plus on both conditions "
             f"(brain = all modules; qwen = zero-shot single call). "
             f"Tower of Hanoi = {disks} disks._\n")
    L.append("## Tower of Hanoi\n")
    L.append("| Metric | qwen-alone | brain (multi-agent) |")
    L.append("|---|---|---|")
    L.append(f"| % solved (valid → goal, 0 invalid) | {hsum['qwen']['pct_solved']} | {hsum['brain']['pct_solved']} |")
    L.append(f"| % reached goal | {hsum['qwen']['pct_reached_goal']} | {hsum['brain']['pct_reached_goal']} |")
    L.append(f"| % invalid moves (↓) | {hsum['qwen']['pct_invalid_moves']} | {hsum['brain']['pct_invalid_moves']} |")
    L.append(f"| avg LLM calls / problem | {hsum['qwen']['avg_calls']} | {hsum['brain']['avg_calls']} |")
    L.append("\n## Graph traversal (shortest path)\n")
    L.append("| Metric | qwen-alone | brain (multi-agent) |")
    L.append("|---|---|---|")
    L.append(f"| % solved (valid shortest path) | {gsum['qwen']['pct_solved']} | {gsum['brain']['pct_solved']} |")
    L.append(f"| % valid path (reaches goal, real edges) | {gsum['qwen']['pct_valid_path']} | {gsum['brain']['pct_valid_path']} |")
    L.append(f"| total invalid edges used (↓) | {gsum['qwen']['total_invalid_edges']} | {gsum['brain']['total_invalid_edges']} |")
    L.append(f"| avg LLM calls / problem | {gsum['qwen']['avg_calls']} | {gsum['brain']['avg_calls']} |")
    L.append("\n_Reference (paper, GPT-4): ToH 3-disk zero-shot 11% vs MAP 74%; "
             "graph steppath zero-shot ~75/40/20% (2/3/4-step) vs MAP ~100%._\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--disks", type=int, default=3)
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    log = lambda m: print(m, flush=True)
    t0 = time.time()
    log("══ MAP-paper comparison: brain vs qwen-alone ══\n")

    hrows = run_hanoi(args.n, args.disks, log)
    grows = run_graph(args.n, log)

    hsum = summarize(hrows, "hanoi")
    gsum = summarize(grows, "graph")

    out_dir = Path(__file__).resolve().parent.parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"n": args.n, "disks": args.disks, "model": "qwen/qwen3.7-plus",
                   "elapsed_sec": round(time.time() - t0, 1)},
        "hanoi": {"summary": hsum, "rows": hrows},
        "graph": {"summary": gsum, "rows": grows},
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    md = to_markdown(hsum, gsum, args.n, args.disks)
    (out_dir / "results.md").write_text(md)

    print("\n" + md)
    print(f"\nWrote {out_dir/'results.json'} and {out_dir/'results.md'} "
          f"({payload['config']['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
