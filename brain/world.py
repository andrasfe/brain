"""World — the rich environment the humanized brain inhabits.

A real human is never in a sensory vacuum: there is light, temperature, time of
day, ambient noise, hunger building, a partner asking when you'll be done,
notifications, a deadline, a half-remembered task in the next room. The raw
LLM has none of this; that is half of why its outputs feel mechanical.

`World` is a deterministic-or-stochastic ticker that emits ambient stimuli each
cognitive cycle. The orchestrator pulls them in and posts them as Broadcasts
from `source="world"` so the sensory cortex / interoception can pick them up,
just as the eyes/ears/skin would in a body.

Boundary respected: the world DOES NOT call regions. It returns a list of
Stimuli per tick, which the orchestrator posts to the workspace. Regions read
the workspace as usual.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Stimulus:
    kind: str                       # "ambient" | "notification" | "social" | "body" | "deadline"
    content: str                    # natural-language description
    salience: float = 0.4           # 0..1; high stimuli grab the spotlight
    affect_delta: Dict[str, float] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)


# ── scenario presets ───────────────────────────────────────────────────────
# Each preset seeds initial body/affect and the ambient event schedule.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "calm_morning": {
        "start_hour": 9.0,
        "ambient": ["sunlight through window", "coffee on the desk", "quiet apartment"],
        "init_affect": {"valence": 0.25, "arousal": 0.45, "fatigue": 0.10, "stress": 0.05},
        "event_rate": 0.15,   # P(event per cycle)
        "weather": "clear",
    },
    "deadline_night": {
        "start_hour": 23.0,
        "ambient": ["dim desk lamp", "empty office", "third coffee"],
        "init_affect": {"valence": -0.10, "arousal": 0.75, "fatigue": 0.55,
                         "stress": 0.55, "hunger": 0.45},
        "event_rate": 0.55,
        "weather": "rain",
        "deadline_in_cycles": 8,
    },
    "boring_afternoon": {
        "start_hour": 14.5,
        "ambient": ["fluorescent office hum", "AC droning", "open-plan chatter"],
        "init_affect": {"valence": -0.05, "arousal": 0.30, "fatigue": 0.45,
                         "boredom": 0.55, "social_need": 0.40},
        "event_rate": 0.40,
        "weather": "overcast",
    },
    "social_evening": {
        "start_hour": 19.0,
        "ambient": ["friends in the kitchen", "music playing", "warm light"],
        "init_affect": {"valence": 0.35, "arousal": 0.55, "social_need": 0.10,
                         "fatigue": 0.30},
        "event_rate": 0.65,
        "weather": "clear",
    },
    "sick_day": {
        "start_hour": 11.0,
        "ambient": ["bed", "blanket", "humidifier"],
        "init_affect": {"valence": -0.30, "arousal": 0.20, "fatigue": 0.80,
                         "stress": 0.30, "hunger": 0.10},
        "event_rate": 0.10,
        "weather": "grey",
    },
    "neutral": {
        "start_hour": 12.0,
        "ambient": ["desk", "laptop", "background hum"],
        "init_affect": {},
        "event_rate": 0.25,
        "weather": "neutral",
    },
}


# Notifications, social pings, intrusive ambient stimuli — drawn at random
# according to the scenario's event_rate. Each carries a small affect_delta.
_EVENT_POOL: List[Stimulus] = [
    Stimulus("notification", "phone buzzes: Slack DM from a colleague",
             salience=0.55, affect_delta={"social_need": -0.05, "arousal": +0.04}),
    Stimulus("notification", "email alert: 'quick question'",
             salience=0.45, affect_delta={"stress": +0.04, "arousal": +0.03}),
    Stimulus("social", "partner texts: 'when are you done?'",
             salience=0.7, affect_delta={"stress": +0.06, "social_need": -0.08,
                                          "valence": -0.04}),
    Stimulus("social", "friend texts a meme",
             salience=0.5, affect_delta={"valence": +0.06, "social_need": -0.06,
                                          "boredom": -0.05}),
    Stimulus("ambient", "neighbor's dog barks",
             salience=0.35, affect_delta={"arousal": +0.04, "stress": +0.02}),
    Stimulus("ambient", "rain picks up against the window",
             salience=0.30, affect_delta={"valence": +0.02, "arousal": -0.02}),
    Stimulus("ambient", "a memory of last weekend surfaces uninvited",
             salience=0.40, affect_delta={"valence": +0.05}),
    Stimulus("ambient", "stomach growls",
             salience=0.45, affect_delta={"hunger": +0.05}),
    Stimulus("notification", "calendar reminder: meeting in 15 minutes",
             salience=0.75, affect_delta={"stress": +0.08, "arousal": +0.06}),
    Stimulus("ambient", "someone laughs loudly in the next room",
             salience=0.4, affect_delta={"social_need": +0.04, "arousal": +0.03}),
    Stimulus("ambient", "a song you haven't heard in years starts playing",
             salience=0.5, affect_delta={"valence": +0.07, "arousal": +0.04,
                                          "boredom": -0.07}),
]


class World:
    """Ambient environment around the brain. Ticks once per cognitive cycle."""

    def __init__(self, scenario: str = "neutral", seed: Optional[int] = None,
                 cycle_minutes: float = 12.0):
        if scenario not in SCENARIOS:
            scenario = "neutral"
        self.name = scenario
        self.spec = SCENARIOS[scenario]
        self.rng = random.Random(seed)
        self.cycle_minutes = cycle_minutes
        self.hour: float = float(self.spec.get("start_hour", 12.0))
        self.day = "Tuesday"
        self.weather = self.spec.get("weather", "neutral")
        self.cycle = 0
        self.deadline_in_cycles: Optional[int] = self.spec.get("deadline_in_cycles")

    # ── public ──────────────────────────────────────────────────────────────
    def initial_affect_deltas(self) -> Dict[str, float]:
        """Scenario seed: applied once to AffectState before cycle 1."""
        return dict(self.spec.get("init_affect", {}))

    def tick(self) -> List[Stimulus]:
        """Advance one cycle, return ambient stimuli for the orchestrator."""
        self.cycle += 1
        self.hour = (self.hour + self.cycle_minutes / 60.0) % 24.0

        stim: List[Stimulus] = []

        # Always: a quiet body/environment update so the brain knows the time
        # and ambient state (low salience — only surfaces if nothing else matters)
        stim.append(Stimulus(
            kind="body", content=self._body_line(), salience=0.22,
            affect_delta=self._time_pressure_affect(),
            data={"hour": round(self.hour, 2), "day": self.day, "weather": self.weather},
        ))

        # Scenario ambient flavor — once early on
        if self.cycle <= 2:
            for desc in self.spec.get("ambient", [])[:2]:
                stim.append(Stimulus("ambient", desc, salience=0.28))

        # Stochastic events
        rate = float(self.spec.get("event_rate", 0.25))
        if self.rng.random() < rate:
            ev = self.rng.choice(_EVENT_POOL)
            # copy so we don't mutate the pool
            stim.append(Stimulus(
                kind=ev.kind, content=ev.content,
                salience=ev.salience, affect_delta=dict(ev.affect_delta),
                data=dict(ev.data),
            ))

        # Deadline pressure
        if self.deadline_in_cycles is not None:
            remaining = self.deadline_in_cycles - self.cycle
            if 0 < remaining <= 4:
                stim.append(Stimulus(
                    kind="deadline",
                    content=f"deadline pressure: ~{remaining} cycles remaining",
                    salience=0.6 + 0.1 * (4 - remaining),
                    affect_delta={"stress": +0.06 + 0.03 * (4 - remaining),
                                   "arousal": +0.04},
                ))
            elif remaining <= 0:
                stim.append(Stimulus(
                    kind="deadline",
                    content="deadline passed — consequences mounting",
                    salience=0.85,
                    affect_delta={"stress": +0.12, "valence": -0.08,
                                   "dominance": -0.05},
                ))

        return stim

    # ── internals ───────────────────────────────────────────────────────────
    def _body_line(self) -> str:
        tod = _time_of_day(self.hour)
        return (f"{self.day} {self.hour:05.2f}h ({tod}); weather={self.weather}; "
                f"ambient={self._ambient_short()}")

    def _ambient_short(self) -> str:
        amb = self.spec.get("ambient", [])
        return amb[0] if amb else "neutral"

    def _time_pressure_affect(self) -> Dict[str, float]:
        """Circadian-ish nudges: late night raises fatigue, midday lowers it."""
        # cosine over the day: peak energy ~10am and ~6pm-ish
        circ = -0.5 * math.cos((self.hour - 10) * math.pi / 12)  # -0.5..+0.5
        return {
            "fatigue": +0.005 - 0.01 * circ,    # late night => fatigue rises
            "arousal": +0.002 + 0.01 * circ,    # mid-day => arousal modest+
        }


def _time_of_day(hour: float) -> str:
    if hour < 5: return "deep night"
    if hour < 9: return "morning"
    if hour < 12: return "late morning"
    if hour < 14: return "midday"
    if hour < 18: return "afternoon"
    if hour < 21: return "evening"
    if hour < 24: return "night"
    return "night"
