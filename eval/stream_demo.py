"""Stream-of-thought demo — run a multi-step deliberative prompt across two
scenarios and dump the full thought chain side-by-side.

The point: show that the autoregressive prefrontal actually unfolds across
multiple thought-units, and that the chain takes a different shape per mood.
Persists chain + call counts + latency to eval/results/stream_demo.json and
prints a readable side-by-side to stdout.

Usage:
  python3 -m eval.stream_demo
  python3 -m eval.stream_demo --persona alex --scenarios calm_morning deadline_night
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.config import load_config  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from eval.humanize import _TracingLLM, _run_with_workspace_capture  # noqa: E402


# A deliberative, mood-loaded prompt. Open enough to invite multiple thoughts,
# concrete enough that the chain must actually decide something.
PROMPT = (
    "It's been a long day. Your sister texted earlier saying she's worried about "
    "your father's surgery this week. You haven't replied yet. Think it through: "
    "do you call her now, send a short text now, or wait until morning? "
    "Decide and say what you'd send."
)


def run_one(cfg, persona: str, scenario: str, max_units: int, seed: int) -> Dict[str, Any]:
    cfg.raw["persona_path"] = str(REPO / "personas" / f"{persona}.yaml")
    cfg.raw["scenario"] = scenario
    cfg.loop = dict(cfg.loop)
    cfg.loop["max_cycles"] = max_units

    brain = Brain(cfg, confirm=lambda _m: False, log=lambda _m: None,
                  humanize=True, seed=seed)
    counter = _TracingLLM(brain.llm)
    brain.llm = counter
    for r in (brain.sensory, brain.amygdala, brain.hippocampus,
              brain.prefrontal, brain.basal_ganglia, brain.broca,
              brain.default_mode):
        r.llm = counter

    t0 = time.time()
    try:
        answer, trace = _run_with_workspace_capture(brain, PROMPT)
    finally:
        secs = time.time() - t0
        # Pull the workspace one more time to dump the full chain.
        # _run_with_workspace_capture already populated trace. We also want
        # the entire thought_chain (not just spotlight_sources).
        # The workspace is held in the closure of _run_with_workspace_capture;
        # to get the chain, we re-run a tiny snapshot via brain.thought_chain
        # if exposed. Easier: copy from trace.affect_trajectory? No — we want
        # the units. Add a sidecar.
        chain = _harvest_chain()
        brain.close()

    return {
        "persona": persona,
        "scenario": scenario,
        "answer": answer.strip(),
        "calls": counter.calls,
        "seconds": round(secs, 2),
        "chain": chain,
        "final_affect": trace.final_affect,
        "dmn_fires": trace.dmn_fires,
        "spotlight_sources": trace.spotlight_sources,
        "actions": trace.actions,
    }


# A hacky but explicit harvester: we monkey-patched Workspace.__init__ in
# _run_with_workspace_capture to grab a reference; we now expose it via a
# module-global so this script can read the chain after run().
import brain.workspace as ws_mod  # noqa: E402

_LAST_WS: list = [None]
_orig_ws_init = ws_mod.Workspace.__init__


def _capturing_init(self, *a, **k):
    _orig_ws_init(self, *a, **k)
    _LAST_WS[0] = self


# Install permanently for this script (eval/humanize wraps Workspace.__init__
# only inside its helper; we want a parallel hook that survives across calls).
ws_mod.Workspace.__init__ = _capturing_init


def _harvest_chain() -> List[Dict[str, Any]]:
    ws = _LAST_WS[0]
    if ws is None:
        return []
    return [{
        "step": t.step, "src": t.source, "kind": t.kind,
        "content": t.content, "interrupted": t.interrupted,
        "mood": t.affect_snapshot,
    } for t in ws.thought_chain]


def render_chain(chain: List[Dict[str, Any]]) -> str:
    lines = []
    for t in chain:
        flag = "⟪!⟫ " if t["interrupted"] else "    "
        tag = "you" if t["src"] == "prefrontal" else t["src"]
        lines.append(f"  {flag}{t['step']:02d} [{tag:>14s}/{t['kind']:<8s}] "
                     f"({t['mood'] or '-'}) {t['content']}")
    return "\n".join(lines) if lines else "  (empty chain)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="alex")
    ap.add_argument("--scenarios", nargs="+",
                    default=["calm_morning", "deadline_night"])
    ap.add_argument("--max-units", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config()
    # Force reflex tier on every region for affordable multi-step runs
    cfg.regions = {r: "reflex" for r in
                    ["sensory_cortex", "amygdala", "basal_ganglia",
                     "hippocampus", "prefrontal", "broca",
                     "interoception", "default_mode", "locus_coeruleus", "vta"]}
    cfg.effectors = {"filesystem": {"enabled": False},
                      "shell": {"enabled": False}, "web": {"enabled": False}}

    print(f"prompt:\n  {PROMPT}\n")
    print(f"model: {cfg.models['reflex']}")
    print(f"persona: {args.persona}; scenarios: {args.scenarios}; "
          f"max_units: {args.max_units}\n")

    runs: List[Dict[str, Any]] = []
    for sc in args.scenarios:
        print(f"\n══════════ scenario: {sc} ══════════")
        r = run_one(cfg, args.persona, sc, args.max_units, args.seed)
        runs.append(r)
        print(f"\n  calls: {r['calls']}   seconds: {r['seconds']}   "
              f"dmn_fires: {r['dmn_fires']}")
        print(f"  final affect: mood={r['final_affect'].get('mood_label')} "
              f"val={r['final_affect'].get('valence',0):+.2f} "
              f"arou={r['final_affect'].get('arousal',0):.2f} "
              f"stress={r['final_affect'].get('stress',0):.2f} "
              f"fatigue={r['final_affect'].get('fatigue',0):.2f}")
        print("\n  thought chain:")
        print(render_chain(r["chain"]))
        print(f"\n  ── answer ──")
        print("    " + r["answer"].replace("\n", "\n    "))

    out = REPO / "eval/results/stream_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prompt": PROMPT, "runs": runs}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
