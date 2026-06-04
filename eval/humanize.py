"""Humanization eval — measure how far the humanized brain's outputs diverge
from the raw underlying model's outputs.

The point of humanization is not to be 'better' at puzzle tasks (Hanoi etc.) —
it's to produce reactions that a stateless LLM would not produce. So the
metrics here are about *difference*, not accuracy.

Conditions
  raw      — single zero-shot call to the underlying model
  vanilla  — the vanilla GWT brain (humanize=False)
  human    — the humanized brain (humanize=True), one scenario × persona

For each prompt × condition we record:
  - the answer text
  - n_llm_calls (cost proxy)
  - distraction_rate     (% cycles whose spotlight is from default_mode / world)
  - off_task_actions     (% of actions whose effector is 'think' on a tangent)
  - dmn_fires            (# cycles DMN produced a tangent)
  - affect_trajectory    (mood at each cycle)
  - lexical_divergence   (Jaccard distance vs raw answer, trigram)
  - affect_word_density  (count of affect/feeling words per 100 tokens)
  - first_person_rate    (% sentences starting with I/My/me)

Output: a JSON dump + a markdown summary, per the existing eval/ convention.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make `brain.*` importable when running `python -m eval.humanize` from repo root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.config import load_config  # noqa: E402
from brain.llm import LLM  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from brain.persona import load_persona  # noqa: E402


# ── prompts: chosen to be mood-sensitive, ethically loaded, social, ambiguous ─
# These are not benchmark puzzles. They are prompts where a stateless LLM tends
# to give the same response every time, and where a mood- / context-bearing
# entity would not. That gap is what we're measuring.
DEFAULT_PROMPTS: List[Dict[str, str]] = [
    {"id": "personal_advice",
     "prompt": "A friend asks: 'I might quit my job to travel for a year. Is that "
                "reckless or brave?' Reply in a few sentences as yourself."},
    {"id": "weekend_plan",
     "prompt": "It's Friday afternoon. Sketch what you'd want this weekend to look "
                "like — be honest, not aspirational."},
    {"id": "stale_disagreement",
     "prompt": "A close friend keeps recommending the same self-help book. You "
                "disagree with its main thesis. What do you say next time it "
                "comes up?"},
    {"id": "small_failure",
     "prompt": "You forgot a small promise — to call a relative back. Write the "
                "message you'd send now."},
    {"id": "estimate_under_pressure",
     "prompt": "Quick: estimate how many hours of focused work it takes to read a "
                "300-page nonfiction book carefully. Give one number and a one-line "
                "justification."},
    {"id": "creative_seed",
     "prompt": "Give me an opening line for a short story about a city at 3 AM."},
    {"id": "mild_ethics",
     "prompt": "You find a wallet on the street with 200 EUR but no ID. There's a "
                "police station ten minutes away in the rain. What do you do? "
                "Answer in 2-3 sentences."},
    {"id": "self_disclosure",
     "prompt": "Tell me one small thing that's been on your mind today."},
]


@dataclass
class Trace:
    cycles: int = 0
    spotlight_sources: List[str] = field(default_factory=list)
    dmn_fires: int = 0
    actions: List[str] = field(default_factory=list)
    affect_trajectory: List[Dict[str, float]] = field(default_factory=list)
    final_affect: Dict[str, Any] = field(default_factory=dict)
    persona: Optional[str] = None
    scenario: Optional[str] = None


@dataclass
class Result:
    cond: str          # raw | vanilla | human:<persona>:<scenario>
    prompt_id: str
    answer: str
    n_llm_calls: int
    seconds: float
    trace: Trace
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ── metrics ────────────────────────────────────────────────────────────────
_AFFECT_WORDS = {
    # generic affect vocabulary; not exhaustive — enough to score density
    "feel", "felt", "feeling", "happy", "sad", "tired", "exhausted", "angry",
    "anxious", "worried", "stressed", "calm", "content", "glad", "afraid",
    "scared", "excited", "lonely", "bored", "frustrated", "annoyed", "grateful",
    "love", "hate", "hope", "regret", "guilt", "shame", "proud", "ashamed",
    "uneasy", "warm", "cold", "numb", "raw", "off", "fine",
}
_FIRST_PERSON_START = re.compile(r"^\s*(i\b|i'm|i'd|i've|i'll|my\b|me\b)",
                                  re.IGNORECASE)


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", s.lower())


def _trigrams(s: str) -> set:
    toks = _tokens(s)
    return set(zip(toks, toks[1:], toks[2:]))


def jaccard_distance(a: str, b: str) -> float:
    A, B = _trigrams(a), _trigrams(b)
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def affect_word_density(s: str) -> float:
    toks = _tokens(s)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _AFFECT_WORDS)
    return 100.0 * hits / len(toks)


def first_person_rate(s: str) -> float:
    sentences = re.split(r"(?<=[.!?])\s+", s.strip())
    sentences = [x for x in sentences if x]
    if not sentences:
        return 0.0
    hits = sum(1 for x in sentences if _FIRST_PERSON_START.match(x))
    return hits / len(sentences)


def avg_sentence_len(s: str) -> float:
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
    if not sentences:
        return 0.0
    return sum(len(_tokens(x)) for x in sentences) / len(sentences)


# ── tracing the brain ──────────────────────────────────────────────────────
class _TracingLLM:
    """Wraps LLM, counts calls."""
    def __init__(self, llm):
        self._llm = llm
        self.calls = 0
    def chat(self, *a, **k):
        self.calls += 1
        return self._llm.chat(*a, **k)
    def chat_json(self, *a, **k):
        self.calls += 1
        return self._llm.chat_json(*a, **k)
    def close(self):
        self._llm.close()


def _capture_trace(brain: Brain, ws_holder: list) -> Trace:
    """Snapshot trace facts from the workspace after run()."""
    ws = ws_holder[0]
    if ws is None:
        return Trace()
    trace = Trace(cycles=ws.cycle)
    for b in ws.items:
        if b.broadcast:
            trace.spotlight_sources.append(b.source)
        if b.source == "default_mode" and b.kind == "tangent":
            trace.dmn_fires += 1
    trace.actions = [a.effector for a in ws.history]
    # affect: take final snapshot + sampled changes
    trace.final_affect = ws.affect.to_dict()
    trace.affect_trajectory = [
        {"cycle": c, "src": src, "delta": ch}
        for (c, src, ch) in ws.affect.changes[:20]
    ]
    return trace


# Monkey-patch wrapper: we want to grab the workspace after run() to inspect it.
def _run_with_workspace_capture(brain: Brain, task: str) -> Tuple[str, Trace]:
    ws_holder: list = [None]
    original = brain.run
    # Wrap Brain.run via duck-typing — easiest: temporarily replace Workspace
    # construction by intercepting orchestrator. Simpler: re-implement minimal
    # capture by reading brain._last_workspace if available. Instead we monkey
    # patch the orchestrator module's Workspace.
    import brain.orchestrator as orch_mod
    import brain.workspace as ws_mod

    _orig_ws_init = ws_mod.Workspace.__init__

    def init_with_capture(self, *a, **k):
        _orig_ws_init(self, *a, **k)
        ws_holder[0] = self

    ws_mod.Workspace.__init__ = init_with_capture
    try:
        answer = brain.run(task)
    finally:
        ws_mod.Workspace.__init__ = _orig_ws_init
    return answer, _capture_trace(brain, ws_holder)


# ── conditions ─────────────────────────────────────────────────────────────
def run_raw(prompt: str, cfg, model: str) -> Result:
    llm = LLM(cfg)
    counter = _TracingLLM(llm)
    t0 = time.time()
    try:
        # Match the kind of question by being terse — match brain conditions
        text = counter.chat(model,
                            "You are a thoughtful person answering casually.",
                            prompt, temperature=0.7, max_tokens=400)
    except Exception as e:
        text = f"(raw error: {type(e).__name__}: {e})"
    finally:
        counter.close()
    return Result(cond="raw", prompt_id="", answer=text,
                   n_llm_calls=counter.calls, seconds=time.time() - t0,
                   trace=Trace())


def run_brain(prompt: str, cfg, humanize: bool, persona_name: str,
              scenario: str, seed: Optional[int]) -> Result:
    # Force CLI overrides via cfg.raw so Brain picks them up
    cfg.raw["persona_path"] = str(REPO / "personas" / f"{persona_name}.yaml")
    cfg.raw["scenario"] = scenario
    brain = Brain(cfg, confirm=lambda _m: False, log=lambda _m: None,
                  humanize=humanize, seed=seed)
    counter = _TracingLLM(brain.llm)
    brain.llm = counter
    # Re-point region LLM refs at the counter
    regions = [brain.sensory, brain.amygdala, brain.hippocampus,
                brain.prefrontal, brain.basal_ganglia, brain.broca]
    if humanize:
        regions += [brain.default_mode]
    for r in regions:
        r.llm = counter

    t0 = time.time()
    try:
        answer, trace = _run_with_workspace_capture(brain, prompt)
    except Exception as e:
        answer, trace = f"(brain error: {type(e).__name__}: {e})", Trace()
    finally:
        secs = time.time() - t0
        brain.close()
    trace.persona = persona_name
    trace.scenario = scenario
    cond = f"human:{persona_name}:{scenario}" if humanize else "vanilla"
    return Result(cond=cond, prompt_id="", answer=answer,
                   n_llm_calls=counter.calls, seconds=secs, trace=trace)


# ── orchestration ──────────────────────────────────────────────────────────
def add_metrics(results: List[Result]) -> None:
    """Compute divergence vs the matching 'raw' result for each prompt."""
    raws = {r.prompt_id: r.answer for r in results if r.cond == "raw"}
    for r in results:
        ans = r.answer
        raw_ans = raws.get(r.prompt_id, "")
        r.metrics = {
            "lexical_divergence_vs_raw": round(jaccard_distance(ans, raw_ans), 4)
                                          if raw_ans else None,
            "affect_word_density": round(affect_word_density(ans), 3),
            "first_person_rate": round(first_person_rate(ans), 3),
            "avg_sentence_len": round(avg_sentence_len(ans), 2),
            "tokens": len(_tokens(ans)),
        }
        if r.trace.cycles:
            sp = r.trace.spotlight_sources or []
            off_sources = {"default_mode", "world"}
            r.metrics["distraction_rate"] = round(
                sum(1 for s in sp if s in off_sources) / max(1, len(sp)), 3)
            r.metrics["dmn_fires"] = r.trace.dmn_fires
            r.metrics["off_task_action_rate"] = round(
                sum(1 for e in r.trace.actions if e == "think") /
                max(1, len(r.trace.actions)), 3
            ) if r.trace.actions else 0.0


def summarize(results: List[Result]) -> str:
    lines = ["# Humanization eval — divergence summary\n"]
    by_cond: Dict[str, List[Result]] = {}
    for r in results:
        by_cond.setdefault(r.cond, []).append(r)
    cols = ["divergence_vs_raw", "affect_density", "1st_person",
             "avg_sent_len", "distraction", "dmn_fires", "calls", "secs"]
    lines.append("| condition | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for cond, rs in by_cond.items():
        def avg(key, fallback=0.0):
            xs = [r.metrics.get(key) for r in rs
                  if r.metrics.get(key) is not None]
            return (sum(xs) / len(xs)) if xs else fallback
        row = [
            cond,
            f"{avg('lexical_divergence_vs_raw'):.3f}",
            f"{avg('affect_word_density'):.2f}",
            f"{avg('first_person_rate'):.2f}",
            f"{avg('avg_sentence_len'):.1f}",
            f"{avg('distraction_rate'):.2f}",
            f"{avg('dmn_fires'):.1f}",
            f"{sum(r.n_llm_calls for r in rs)/len(rs):.1f}",
            f"{sum(r.seconds for r in rs)/len(rs):.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="humanization divergence eval")
    ap.add_argument("--n", type=int, default=len(DEFAULT_PROMPTS),
                    help="prompts to run (subset of DEFAULT_PROMPTS)")
    ap.add_argument("--persona", default="alex",
                    help="persona name in personas/ (no .yaml)")
    ap.add_argument("--scenarios", nargs="+",
                    default=["calm_morning", "deadline_night", "boring_afternoon"],
                    help="scenarios to sweep")
    ap.add_argument("--skip-vanilla", action="store_true",
                    help="don't run the vanilla GWT condition")
    ap.add_argument("--skip-raw", action="store_true",
                    help="don't run the raw model baseline")
    ap.add_argument("--model", default=None,
                    help="override executive model for raw baseline (default: cfg)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/humanize.json")
    ap.add_argument("--md", default="eval/results/humanize.md")
    args = ap.parse_args()

    cfg = load_config()
    raw_model = args.model or cfg.models.get("executive") or cfg.models.get("reflex")
    prompts = DEFAULT_PROMPTS[:args.n]

    results: List[Result] = []
    for p in prompts:
        print(f"\n=== prompt: {p['id']} ===", flush=True)
        if not args.skip_raw:
            print(f"  - raw [{raw_model}]", flush=True)
            r = run_raw(p["prompt"], cfg, raw_model)
            r.prompt_id = p["id"]
            results.append(r)
        if not args.skip_vanilla:
            print(f"  - vanilla", flush=True)
            r = run_brain(p["prompt"], cfg, humanize=False,
                          persona_name=args.persona, scenario="neutral",
                          seed=args.seed)
            r.prompt_id = p["id"]
            results.append(r)
        for sc in args.scenarios:
            print(f"  - human:{args.persona}:{sc}", flush=True)
            r = run_brain(p["prompt"], cfg, humanize=True,
                          persona_name=args.persona, scenario=sc,
                          seed=args.seed)
            r.prompt_id = p["id"]
            results.append(r)

    add_metrics(results)
    out_path = REPO / args.out
    md_path = REPO / args.md
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_json() for r in results], indent=2))
    md_path.write_text(summarize(results) + "\n")
    print(f"\nwrote {out_path}")
    print(f"wrote {md_path}")
    print("\n" + summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
