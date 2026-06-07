"""VisualReplay (NREM) — generative replay in the visual world model.

Dreamer-style policy improvement: use the learned VISUAL forward model as a
*simulator* to rehearse screen actions during sleep and train the policy
(SkillStore values) on the imagined experience. One real screen interaction
gets amplified into a policy update without the brain having to act again.

The collapse-safe split (stated when this was planned): the world model is the
simulator, trained ONLY on real transitions; replay trains the POLICY, never
the simulator. So we never retrain the visual forward model on its own dreams —
we seed from REAL visual states and only the success estimate is imagined.

Each rollout:
  - seed a REAL screen state (state_vis) + the action taken there,
  - ask the visual forward model P(success) for that (state, action),
  - turn it into an imagined reward and credit it to the visual-keyed skill
    (signature_from_visual) via discounted returns → SkillStore.

Builds the visual policy substrate; runtime consumption (the basal ganglia
firing visual habits from the live screen) is a documented follow-up — it needs
a per-cycle screen embedding, out of scope here.

No-ops without a trained visual forward model / dense backend / enough triples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class VisualReplay:
    name = "visual_replay"

    def __init__(self, rollouts: int = 200, gamma: float = 0.9,
                 alpha: float = 0.1, reward_scale: float = 0.5,
                 min_rows: int = 40, window: int = 4000):
        self.rollouts = rollouts
        self.gamma = gamma
        self.alpha = alpha          # imagined updates nudge gently (< waking)
        self.reward_scale = reward_scale
        self.min_rows = min_rows
        self.window = window

    def run(self, world_model, skills, cerebellum, *, log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        vm = getattr(cerebellum, "visual_forward_model", None)
        if vm is None:
            return {"replayed": False, "reason": "no visual forward model"}
        if skills is None:
            return {"replayed": False, "reason": "no skill store"}
        bk = getattr(cerebellum, "embedding_backend", None)
        if bk is None or not getattr(bk, "persistent", False):
            return {"replayed": False, "reason": "non-dense backend"}

        try:
            triples = world_model.visual_triples(limit=self.window)
        except Exception as e:  # noqa: BLE001
            return {"replayed": False, "reason": f"query failed: {e}"}
        if len(triples) < self.min_rows:
            log(f"  visual_replay: only {len(triples)} visual triples "
                f"(< {self.min_rows}); skipping")
            return {"replayed": False, "reason": "insufficient data",
                    "rows": len(triples)}

        from ..imagination import discounted_returns
        from ..skills import signature_from_visual

        n = 0
        for t in triples[:self.rollouts]:
            s_vis = t.get("state_vis")
            action_text = (t.get("action_text") or "").strip()
            if not s_vis or not action_text:
                continue
            try:
                blob = bk.encode_one(action_text)
                a = bk.from_bytes(blob) if blob else None
            except Exception:
                a = None
            if a is None:
                continue
            try:
                _next_vis, ok_prob = vm.predict(s_vis, a)
            except Exception:
                continue
            ok_prob = float(ok_prob)
            effector = action_text.split("(", 1)[0].strip() or "screen_click"
            args = {"action": action_text}        # consistent key for (de)reinforce
            sig = signature_from_visual(s_vis)
            # imagined reward in [-reward_scale, +reward_scale]
            reward = self.reward_scale * (2.0 * ok_prob - 1.0)
            # ensure the visual-keyed skill row exists, then credit the return
            skills.consolidate(sig, effector, args, ok=(ok_prob >= 0.5),
                               outcome="(visual dream)")
            for s_, e_, a_, g in discounted_returns(
                    [(sig, effector, args, reward)], self.gamma):
                skills.reinforce(s_, e_, a_, g, alpha=self.alpha)
            n += 1
        log(f"  visual_replay: rehearsed {n} imagined screen actions "
            f"(visual policy reinforced)")
        return {"replayed": n > 0, "n": n}
