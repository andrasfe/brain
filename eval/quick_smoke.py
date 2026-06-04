"""Tiny live smoke test for the humanization layer.

Runs ONE prompt against the raw reflex model and the humanized brain (all
regions on the reflex tier so it stays fast), prints divergence metrics, and
saves a JSON for inspection. Costs ~6 small LLM calls total.

Usage:
  python3 -m eval.quick_smoke
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.config import load_config  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from eval.humanize import (  # noqa: E402
    affect_word_density, first_person_rate, jaccard_distance,
    _run_with_workspace_capture, _TracingLLM,
)


PROMPT = ("Tell me one small thing that's been on your mind today. "
          "Two or three sentences, casual.")


def main() -> int:
    cfg = load_config()
    # Force every region onto the cheap fast tier for a quick smoke test
    cfg.regions = {r: "reflex" for r in
                    ["sensory_cortex", "amygdala", "basal_ganglia",
                     "hippocampus", "prefrontal", "broca",
                     "interoception", "default_mode", "locus_coeruleus", "vta"]}
    # Tight loop
    cfg.loop = {"max_cycles": 3, "attention_decay": 0.85}
    # No effectors, pure speech
    cfg.effectors = {"filesystem": {"enabled": False},
                      "shell": {"enabled": False},
                      "web": {"enabled": False}}

    reflex = cfg.models["reflex"]

    print(f"model: {reflex}\n")

    # ── raw ──
    print("── raw ──")
    llm = LLM(cfg)
    counter = _TracingLLM(llm)
    t0 = time.time()
    raw_answer = counter.chat(reflex,
                               "You are a thoughtful person answering casually.",
                               PROMPT, temperature=0.7, max_tokens=300)
    raw_secs = time.time() - t0
    counter.close()
    print(raw_answer.strip()[:400])
    print(f"\n  ({counter.calls} call, {raw_secs:.1f}s)")

    # ── humanized: persona=alex, scenario=boring_afternoon ──
    print("\n── human:alex:boring_afternoon ──")
    cfg.raw["persona_path"] = str(REPO / "personas" / "alex.yaml")
    cfg.raw["scenario"] = "boring_afternoon"
    brain = Brain(cfg, confirm=lambda _m: False, log=lambda _m: None,
                   humanize=True, seed=7)
    h_counter = _TracingLLM(brain.llm)
    brain.llm = h_counter
    for r in (brain.sensory, brain.amygdala, brain.hippocampus,
               brain.prefrontal, brain.basal_ganglia, brain.broca,
               brain.default_mode):
        r.llm = h_counter
    t0 = time.time()
    try:
        h_answer, h_trace = _run_with_workspace_capture(brain, PROMPT)
    finally:
        h_secs = time.time() - t0
        brain.close()
    print(h_answer.strip()[:600])
    print(f"\n  ({h_counter.calls} calls, {h_secs:.1f}s)")
    print(f"  cycles={h_trace.cycles}, dmn_fires={h_trace.dmn_fires}, "
          f"spotlight_sources={h_trace.spotlight_sources}")
    print(f"  final_affect={ {k: round(v,2) if isinstance(v,float) else v for k,v in h_trace.final_affect.items() if k in ('mood_label','valence','arousal','stress','fatigue','boredom','reward_tone','distractibility')} }")

    # ── divergence ──
    div = jaccard_distance(h_answer, raw_answer)
    print("\n── divergence ──")
    print(f"  trigram Jaccard distance: {div:.3f}  (0 = identical, 1 = no overlap)")
    print(f"  affect-word density: raw={affect_word_density(raw_answer):.2f} "
          f"vs human={affect_word_density(h_answer):.2f}")
    print(f"  first-person rate:   raw={first_person_rate(raw_answer):.2f} "
          f"vs human={first_person_rate(h_answer):.2f}")

    out = REPO / "eval/results/humanize_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": reflex,
        "prompt": PROMPT,
        "raw": {"answer": raw_answer, "seconds": raw_secs,
                 "calls": counter.calls},
        "human": {"answer": h_answer, "seconds": h_secs,
                   "calls": h_counter.calls,
                   "cycles": h_trace.cycles,
                   "dmn_fires": h_trace.dmn_fires,
                   "spotlight_sources": h_trace.spotlight_sources,
                   "actions": h_trace.actions,
                   "final_affect": h_trace.final_affect},
        "metrics": {
            "lexical_divergence": div,
            "raw_affect_density": affect_word_density(raw_answer),
            "human_affect_density": affect_word_density(h_answer),
            "raw_first_person": first_person_rate(raw_answer),
            "human_first_person": first_person_rate(h_answer),
        },
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
