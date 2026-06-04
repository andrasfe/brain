"""Affective state — the persistent emotional context that colors cognition.

The architectural rule: regions never call each other; they read/write the
workspace. AffectState lives on Workspace so any region can read it (to thread
into its prompt) or update it (by calling `update` directly — affect-producing
regions like the amygdala/interoception/VTA/LC are the only ones expected to
mutate).

Model: a hybrid of the PAD (Pleasure-Arousal-Dominance) circumplex and a small
set of homeostatic drives. We deliberately keep dimensions cheap so they fit in
a prompt line.

NOTE: emotion has *inertia*. We never overwrite — we EMA. A single cycle's
appraisal nudges state; it does not replace it. This is what makes the brain
behave differently from a one-shot LLM call — affect persists across turns and
biases future choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


# Big-Five-lite trait anchors. Each is fixed per "person" (set at brain init).
# They bias the dynamics, e.g. high neuroticism = stronger negative valence
# response, high conscientiousness = stronger basal-ganglia inhibition, etc.
@dataclass
class Traits:
    neuroticism: float = 0.5         # 0..1; affect volatility, negative bias
    extraversion: float = 0.5        # 0..1; social_need recovery, reward sensitivity
    openness: float = 0.5            # 0..1; curiosity gain, DMN tangent rate
    conscientiousness: float = 0.5   # 0..1; gate strictness, distraction resistance
    agreeableness: float = 0.5       # 0..1; veto on harm, tone softening


@dataclass
class AffectState:
    """Persistent mood/arousal/drives — updated, not replaced, each cycle."""
    # PAD circumplex
    valence: float = 0.0      # -1 (unpleasant) .. +1 (pleasant)
    arousal: float = 0.4      # 0 (drowsy) .. 1 (wired)
    dominance: float = 0.0    # -1 (helpless) .. +1 (in control)

    # Homeostatic drives — all 0 (sated) .. 1 (urgent need)
    hunger: float = 0.15
    fatigue: float = 0.20
    boredom: float = 0.25
    social_need: float = 0.20
    curiosity: float = 0.45
    stress: float = 0.15

    # Reward tone (VTA): recent reward prediction error, decays
    reward_tone: float = 0.0  # -1 (disappointed) .. +1 (elated)

    # Personality (stable across the run)
    traits: Traits = field(default_factory=Traits)

    # Light audit trail (most recent first, capped) for inspection / eval
    changes: List[Tuple[int, str, Dict[str, float]]] = field(default_factory=list)

    # ── update mechanics ─────────────────────────────────────────────────────
    def update(self, source: str, cycle: int, deltas: Dict[str, float],
               smoothing: float = 0.6) -> None:
        """EMA-blend deltas into state. `smoothing` is the carryover weight on
        the existing value, so a single appraisal can't overwrite mood — only
        nudge it. Trait scaling is applied here.
        """
        applied: Dict[str, float] = {}
        for k, delta in deltas.items():
            if not hasattr(self, k):
                continue
            cur = float(getattr(self, k))
            scaled = self._trait_scale(k, float(delta))
            new = smoothing * cur + (1.0 - smoothing) * (cur + scaled)
            new = _clamp(k, new)
            setattr(self, k, new)
            applied[k] = round(new - cur, 4)
        if applied:
            self.changes.insert(0, (cycle, source, applied))
            del self.changes[24:]  # cap log

    def decay_toward_baseline(self) -> None:
        """One step of drift toward homeostatic baselines. Slow — emotions
        linger. Drives accumulate need over time (hunger/fatigue/boredom rise)."""
        # PAD: valence/arousal/dominance drift toward 0 (mood regression to mean)
        self.valence *= 0.97
        self.arousal = 0.95 * self.arousal + 0.05 * 0.35  # baseline arousal
        self.dominance *= 0.97
        self.reward_tone *= 0.85

        # Drives accumulate over time
        self.hunger = _clamp("hunger", self.hunger + 0.015)
        self.fatigue = _clamp("fatigue", self.fatigue + 0.010)
        self.boredom = _clamp("boredom", self.boredom + 0.020)
        self.social_need = _clamp("social_need", self.social_need + 0.008)
        # Curiosity decays toward trait baseline
        c_base = 0.3 + 0.4 * self.traits.openness
        self.curiosity = 0.9 * self.curiosity + 0.1 * c_base
        # Stress drifts down without a stressor
        self.stress = max(0.0, self.stress * 0.93)

    def _trait_scale(self, key: str, delta: float) -> float:
        t = self.traits
        # Neurotics amplify negative valence and stress
        if key == "valence" and delta < 0:
            return delta * (0.6 + 0.8 * t.neuroticism)
        if key == "stress":
            return delta * (0.6 + 0.8 * t.neuroticism)
        if key == "curiosity":
            return delta * (0.5 + t.openness)
        if key == "social_need":
            return delta * (0.6 + 0.8 * t.extraversion)
        if key == "boredom":
            # Open people get bored faster of dull work
            return delta * (0.7 + 0.6 * t.openness)
        if key == "reward_tone":
            return delta * (0.5 + t.extraversion)
        return delta

    # ── derived / display ────────────────────────────────────────────────────
    @property
    def mood_label(self) -> str:
        v, a, s, f = self.valence, self.arousal, self.stress, self.fatigue
        if s > 0.65:
            return "stressed"
        if f > 0.7:
            return "exhausted"
        if v > 0.4 and a > 0.55:
            return "excited"
        if v > 0.3 and a < 0.45:
            return "content"
        if v < -0.4 and a > 0.6:
            return "anxious"
        if v < -0.3 and a < 0.4:
            return "low"
        if self.boredom > 0.7:
            return "bored"
        if self.curiosity > 0.7:
            return "curious"
        return "neutral"

    @property
    def attention_width(self) -> float:
        """High arousal narrows attention (Yerkes–Dodson). 0=narrow, 1=wide."""
        a = self.arousal
        # inverted-U: narrowest at extremes, widest mid
        return max(0.15, 1.0 - abs(a - 0.5) * 1.4)

    @property
    def distractibility(self) -> float:
        """Probability the DMN can grab the spotlight this cycle."""
        # boredom + fatigue + low arousal raise it; conscientiousness lowers it
        base = 0.45 * self.boredom + 0.35 * self.fatigue + 0.25 * (1 - self.arousal)
        damp = 0.7 + 0.6 * self.traits.conscientiousness
        return _clip01(base / damp)

    def render(self) -> str:
        """Compact line to thread into region prompts."""
        return (
            f"AFFECT[mood={self.mood_label} "
            f"val={self.valence:+.2f} arou={self.arousal:.2f} dom={self.dominance:+.2f} "
            f"stress={self.stress:.2f} fatigue={self.fatigue:.2f} hunger={self.hunger:.2f} "
            f"bored={self.boredom:.2f} social={self.social_need:.2f} "
            f"curious={self.curiosity:.2f} reward={self.reward_tone:+.2f}]"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mood_label"] = self.mood_label
        d["attention_width"] = round(self.attention_width, 3)
        d["distractibility"] = round(self.distractibility, 3)
        return d


# ── helpers ────────────────────────────────────────────────────────────────
_BOUNDS = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "dominance": (-1.0, 1.0),
    "hunger": (0.0, 1.0),
    "fatigue": (0.0, 1.0),
    "boredom": (0.0, 1.0),
    "social_need": (0.0, 1.0),
    "curiosity": (0.0, 1.0),
    "stress": (0.0, 1.0),
    "reward_tone": (-1.0, 1.0),
}


def _clamp(key: str, v: float) -> float:
    lo, hi = _BOUNDS.get(key, (-1.0, 1.0))
    return max(lo, min(hi, v))


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))
