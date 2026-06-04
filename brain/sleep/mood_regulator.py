"""MoodRegulator — NREM agent that aggressively drifts affect toward baseline.

While awake the brain's AffectState decays slowly each cognitive cycle
(`decay_toward_baseline`). Sleep is when valence regression to the mean
happens fast — a stressed day doesn't usually feel as bad the next morning.
This agent applies many decay steps in one NREM bout.

Notably:
  - Fatigue DROPS during NREM (you wake refreshed).
  - Hunger doesn't recover from sleep, so the regular `decay_toward_baseline`
    is bypassed for that drive — sleep doesn't feed you.
"""
from __future__ import annotations

from typing import Any

from ..affect import AffectState


class MoodRegulator:
    name = "mood_regulator"

    def __init__(self, recovery_steps: int = 24,
                 fatigue_recovery_per_step: float = 0.04):
        self.recovery_steps = recovery_steps
        self.fatigue_recovery_per_step = fatigue_recovery_per_step

    def run(self, affect: AffectState) -> dict[str, Any]:
        before = {
            "valence": affect.valence, "arousal": affect.arousal,
            "stress": affect.stress, "fatigue": affect.fatigue,
            "boredom": affect.boredom,
        }
        # Many small drift steps + fatigue actively recovers
        for _ in range(self.recovery_steps):
            # Save hunger; sleep doesn't feed you
            h_save = affect.hunger
            affect.decay_toward_baseline()
            affect.hunger = h_save
            # Active fatigue recovery
            affect.fatigue = max(0.0, affect.fatigue - self.fatigue_recovery_per_step)
        after = {
            "valence": affect.valence, "arousal": affect.arousal,
            "stress": affect.stress, "fatigue": affect.fatigue,
            "boredom": affect.boredom,
        }
        return {"before": before, "after": after}
