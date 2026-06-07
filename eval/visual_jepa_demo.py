"""Visual JEPA demo — prove the visual world model learns + the Monitor acts.

No torch, no hands, no overnight sleep. We synthesize action-conditioned visual
transitions with a learnable rule (the Submit-vs-Delete story made concrete),
train the visual forward model in seconds, then show:

  1. it predicts success/failure on HELD-OUT screens it never trained on, and
  2. the Predictor (learned Monitor) VETOES the action it predicts will fail.

The rule is *action-conditioned*: the same screen, clicked at x=10 ("Submit"),
succeeds; clicked at x=200 ("Delete Report") fails. So the model must read the
ACTION, not just the screen — exactly the JEPA property we care about. Real
screens would come from DINOv2; here the screen vectors are synthetic so the
demo runs anywhere with just numpy (or MLX).

Usage:
  python3 -m eval.visual_jepa_demo
  python3 -m eval.visual_jepa_demo --n 160 --epochs 150 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.config import Config  # noqa: E402
from brain.memory import Memory  # noqa: E402
from brain.workspace import Broadcast, Workspace  # noqa: E402
from brain.world_model import WorldModelStore, render_action  # noqa: E402


GOOD = ("screen_click", {"x": 10})     # the "Submit" button → succeeds
BAD = ("screen_click", {"x": 200})     # the "Delete Report" button → fails
VIS_DIM = 24


class _DenseBackend:
    """Deterministic dense embedding backend for action text — hash → fixed-dim
    unit vector. Lets the demo run with no LLM/embedding server."""
    name = "demo-dense"
    persistent = True
    dim = 16

    def encode_one(self, text):
        h = hashlib.sha256((text or "").encode()).digest()
        vals = [((h[i] / 255.0) * 2 - 1) for i in range(self.dim)]
        return struct.pack(f"<I{self.dim}f", self.dim, *vals)

    def from_bytes(self, blob):
        n = struct.unpack("<I", blob[:4])[0]
        return list(struct.unpack(f"<{n}f", blob[4:4 + 4 * n]))


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _screen(rng):
    """A synthetic 'screenshot embedding' — varied background, like DINOv2
    would give. The success rule does NOT depend on this (it depends on the
    action), so the model can't cheat by memorizing screens."""
    return _unit([rng.uniform(-1, 1) for _ in range(VIS_DIM)])


def _outcome(ok, rng):
    """Distinct success vs failure 'result screen' vectors (the regression
    target). Success screens cluster one way, failures the other."""
    sig = 0.5 if ok else -0.5
    return _unit([sig + rng.uniform(-0.15, 0.15) if i < VIS_DIM // 2
                  else rng.uniform(-0.3, 0.3) for i in range(VIS_DIM)])


def _cfg(tmp):
    return Config(raw={}, api_key="", base_url="http://x", require_auth=False,
                  extra_headers={}, models={"reflex": "x", "executive": "y"},
                  timeout_seconds=10, max_retries=0, sandbox_dir=tmp,
                  db_path=tmp / "m.sqlite", loop={}, memory={}, effectors={},
                  regions={})


def _ws():
    ws = Workspace(task="submit the expense report")
    ws.post(Broadcast(source="sensory_cortex", kind="percept", content="g",
                      salience=0.9, data={"goal": "submit the expense report",
                                          "entities": ["Concur", "form"]}))
    return ws


def main() -> int:
    ap = argparse.ArgumentParser(description="Visual JEPA demo (offline).")
    ap.add_argument("--n", type=int, default=160, help="training transitions")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("This demo needs numpy (or MLX). pip install numpy")
        return 1

    rng = random.Random(args.seed)
    tmp = Path(tempfile.mkdtemp())
    backend = _DenseBackend()
    wm = WorldModelStore(tmp / "wm.sqlite", backend=backend)
    mem = Memory(tmp / "m.sqlite", backend=backend)

    print("═" * 64)
    print("  VISUAL JEPA DEMO — action-conditioned screen world model")
    print("═" * 64)
    print(f"  Rule:  click x=10 ('Submit')  → SUCCESS")
    print(f"         click x=200 ('Delete') → FAILURE   (same screen!)")
    print(f"  Generating {args.n} action-conditioned visual transitions…")

    for _ in range(args.n // 2):
        s1 = _screen(rng)
        wm.observe("ui screen", render_action(*GOOD), "report submitted",
                   ok=True, state_vis=s1, outcome_vis=_outcome(True, rng))
        s2 = _screen(rng)
        wm.observe("ui screen", render_action(*BAD), "draft deleted",
                   ok=False, state_vis=s2, outcome_vis=_outcome(False, rng))
    print(f"  world model: {wm.count_visual()} visual triples\n")

    # ── train the visual forward model ──────────────────────────────────────
    from brain.sleep import VisualForwardModelTrainer
    from brain.regions.cerebellum import Cerebellum
    cfg = _cfg(tmp)
    cb = Cerebellum(cfg, llm=None, world_model=wm, embedding_backend=backend)
    print("  Training visual forward model (NREM pass)…")
    stats = VisualForwardModelTrainer(epochs=args.epochs, min_rows=20).run(
        mem, wm, cerebellum=cb, log=lambda m: print(f"    {m}"))
    if not stats.get("trained"):
        print(f"  ✗ training skipped: {stats.get('reason')}")
        return 1
    print(f"  ✓ trained: vis_dim={stats['vis_dim']} act_dim={stats['act_dim']} "
          f"val_loss={stats.get('val_loss')}\n")

    # ── 1) held-out prediction accuracy ─────────────────────────────────────
    print("  [1] HELD-OUT prediction (fresh screens never trained on):")
    correct = tot = 0
    for _ in range(30):
        s = _screen(rng)
        pg = cb.visual_predict(s, *GOOD)
        pb = cb.visual_predict(s, *BAD)
        if pg and pg.predicted_ok:
            correct += 1
        if pb and not pb.predicted_ok:
            correct += 1
        tot += 2
    acc = correct / tot if tot else 0.0
    sample_s = _screen(rng)
    pg = cb.visual_predict(sample_s, *GOOD)
    pb = cb.visual_predict(sample_s, *BAD)
    print(f"      click x=10  → predicted_ok={pg.predicted_ok}  "
          f"(conf {pg.confidence:.2f})   [want True]")
    print(f"      click x=200 → predicted_ok={pb.predicted_ok}  "
          f"(conf {pb.confidence:.2f})   [want False]")
    print(f"      accuracy over 30 held-out screens: {acc:.0%}\n")

    # ── 2) the learned Monitor in action ────────────────────────────────────
    print("  [2] LEARNED MONITOR (Predictor) vetoing the predicted-bad action:")
    from brain.regions import Predictor
    predictor = Predictor(cfg, llm=None, cerebellum=cb, world_model=wm,
                          veto_floor=0.30)
    ws = _ws()
    good_plan = predictor.evaluate(ws, [GOOD], state_vis=_screen(rng))
    bad_plan = predictor.evaluate(ws, [BAD], state_vis=_screen(rng))
    print(f"      Submit (x=10)  → vetoed={good_plan.vetoed}   [want False]")
    print(f"      Delete (x=200) → vetoed={bad_plan.vetoed}   [want True]")
    if bad_plan.reason:
        print(f"        reason: {bad_plan.reason}")

    ok = (acc >= 0.8 and not good_plan.vetoed and bad_plan.vetoed)
    print("\n" + "═" * 64)
    print(f"  RESULT: {'✓ PASS' if ok else '✗ check output'} — the visual world "
          f"model learned the\n  action-conditioned rule and the Monitor acts "
          f"on it (no hardware).")
    print("═" * 64)
    mem.close(); wm.close()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
