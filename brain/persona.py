"""Persona — loadable prior facts that *implicitly* define a person.

A Persona file (YAML or JSON) bundles biographical facts about the individual:
name, age, occupation, relationships, history, recent events, hobbies,
dispositions. Two things happen at brain init:

  1. Each fact is written into the Hippocampus as a special `prior` episode
     with HIGH salience. The existing retrieval mechanism will surface
     relevant facts as the brain reasons (e.g. a task mentioning "code"
     surfaces the user's occupation).

  2. Traits (Big-Five-lite) are derived from the facts by lightweight tag
     heuristics. Explicit trait_overrides in the persona file always win.

We don't need an LLM call to do this: tags on facts are enough for a v1, and
keeping it deterministic means the same persona always produces the same
trait anchors — which is essential for the research eval.

Persona file shape (YAML):

  identity:
    name: "Alex Mendez"
    age: 34
    occupation: "data scientist at a logistics startup"
    location: "Berlin"
  history:
    - "grew up in Lisbon; moved to Berlin at 27"
    - "broke an ankle bouldering last spring"
  relationships:
    - "lives with partner Sam, 6 years"
    - "older sister Maria, doctor, calls weekly"
  recent_events:                 # last few weeks; tints initial affect
    - text: "missed promotion last month"
      affect: {valence: -0.10, dominance: -0.05}
    - text: "finally finished marathon training plan"
      affect: {valence: +0.08, dominance: +0.04}
  hobbies: ["climbing", "vinyl records", "obscure podcasts"]
  dispositions:                  # free-form tags drive trait derivation
    - "introvert"               # → extraversion -
    - "perfectionist"           # → conscientiousness +, neuroticism +
    - "open to weird ideas"     # → openness +
  trait_overrides:              # optional — wins over derived defaults
    neuroticism: 0.62
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .affect import AffectState, Traits


# Disposition tag → trait nudges. Additive on top of 0.5 baseline; clipped 0..1.
# Designed to be obvious & inspectable. Add freely.
_DISPOSITION_TAGS: Dict[str, Dict[str, float]] = {
    # Extraversion axis
    "introvert":              {"extraversion": -0.25},
    "extrovert":              {"extraversion": +0.25},
    "shy":                    {"extraversion": -0.20, "neuroticism": +0.10},
    "social":                 {"extraversion": +0.20, "agreeableness": +0.10},
    # Conscientiousness
    "perfectionist":          {"conscientiousness": +0.20, "neuroticism": +0.15},
    "disciplined":            {"conscientiousness": +0.20},
    "procrastinator":         {"conscientiousness": -0.18},
    "messy":                  {"conscientiousness": -0.15},
    # Neuroticism
    "anxious":                {"neuroticism": +0.25},
    "calm":                   {"neuroticism": -0.20},
    "easygoing":              {"neuroticism": -0.15, "agreeableness": +0.10},
    "moody":                  {"neuroticism": +0.18},
    # Openness
    "curious":                {"openness": +0.20},
    "open to weird ideas":    {"openness": +0.25},
    "traditional":            {"openness": -0.15},
    "creative":               {"openness": +0.20},
    "skeptical":              {"openness": -0.05, "agreeableness": -0.10},
    # Agreeableness
    "warm":                   {"agreeableness": +0.20, "extraversion": +0.05},
    "blunt":                  {"agreeableness": -0.15},
    "competitive":            {"agreeableness": -0.10, "conscientiousness": +0.05},
    "empathetic":             {"agreeableness": +0.20},
    # Mixed / occupational hints
    "workaholic":             {"conscientiousness": +0.15, "neuroticism": +0.10},
    "burnt-out":              {"neuroticism": +0.20, "extraversion": -0.10},
    "ambitious":              {"conscientiousness": +0.10, "neuroticism": +0.05},
}


@dataclass
class Persona:
    name: str = "anonymous"
    age: Optional[int] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    history: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    dispositions: List[str] = field(default_factory=list)
    trait_overrides: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        ident = data.get("identity", {}) or {}
        return cls(
            name=str(ident.get("name", "anonymous")),
            age=ident.get("age"),
            occupation=ident.get("occupation"),
            location=ident.get("location"),
            history=list(data.get("history", []) or []),
            relationships=list(data.get("relationships", []) or []),
            recent_events=list(data.get("recent_events", []) or []),
            hobbies=list(data.get("hobbies", []) or []),
            dispositions=[str(d).lower().strip() for d in (data.get("dispositions") or [])],
            trait_overrides=dict(data.get("trait_overrides", {}) or {}),
            raw=data,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Persona":
        p = Path(path).expanduser()
        with open(p) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    # ── implicit trait derivation ───────────────────────────────────────────
    def derive_traits(self) -> Traits:
        """Tag-based heuristic: each disposition nudges trait values from 0.5."""
        acc: Dict[str, float] = {
            "neuroticism": 0.5, "extraversion": 0.5, "openness": 0.5,
            "conscientiousness": 0.5, "agreeableness": 0.5,
        }
        for tag in self.dispositions:
            key = tag.lower().strip()
            nudge = _DISPOSITION_TAGS.get(key)
            if nudge is None:
                # also try substring match for free-form tags
                for k, v in _DISPOSITION_TAGS.items():
                    if k in key:
                        nudge = v
                        break
            if not nudge:
                continue
            for trait_name, d in nudge.items():
                acc[trait_name] += d
        # Apply overrides
        for k, v in self.trait_overrides.items():
            if k in acc:
                acc[k] = float(v)
        # Clip to [0,1]
        for k in acc:
            acc[k] = max(0.05, min(0.95, acc[k]))
        return Traits(**acc)

    # ── affect seed from recent events ──────────────────────────────────────
    def initial_affect_deltas(self) -> Dict[str, float]:
        """Sum the affect tags on `recent_events` (already small, by design)."""
        deltas: Dict[str, float] = {}
        for ev in self.recent_events:
            if not isinstance(ev, dict):
                continue
            for k, v in (ev.get("affect") or {}).items():
                deltas[k] = deltas.get(k, 0.0) + float(v)
        return deltas

    # ── facts to seed into hippocampus ──────────────────────────────────────
    def memory_seeds(self) -> List[Dict[str, Any]]:
        """Each returned dict is one episode to insert into memory.

        Salience choices: identity > recent events > history/relationships >
        hobbies. Higher salience means the existing retrieve() ranker is more
        likely to surface them.
        """
        seeds: List[Dict[str, Any]] = []
        ident_str = self._identity_line()
        if ident_str:
            seeds.append({"kind": "prior:identity", "content": ident_str,
                          "salience": 0.95})
        for h in self.history:
            seeds.append({"kind": "prior:history", "content": h, "salience": 0.7})
        for r in self.relationships:
            seeds.append({"kind": "prior:relationship", "content": r,
                          "salience": 0.75})
        for ev in self.recent_events:
            text = ev["text"] if isinstance(ev, dict) else str(ev)
            seeds.append({"kind": "prior:recent", "content": text, "salience": 0.85})
        for h in self.hobbies:
            seeds.append({"kind": "prior:hobby", "content": f"hobby: {h}",
                          "salience": 0.55})
        for d in self.dispositions:
            seeds.append({"kind": "prior:disposition",
                          "content": f"self-perceived disposition: {d}",
                          "salience": 0.6})
        return seeds

    def _identity_line(self) -> str:
        parts = [f"I am {self.name}"]
        if self.age is not None:
            parts.append(f"age {self.age}")
        if self.occupation:
            parts.append(self.occupation)
        if self.location:
            parts.append(f"based in {self.location}")
        return ", ".join(parts) + "."

    # ── render for region prompts ───────────────────────────────────────────
    def render_identity_block(self) -> str:
        """A short identity blurb regions can prepend — keeps the 'self' present
        even before retrieve() surfaces facts."""
        bits = [self._identity_line()]
        if self.dispositions:
            bits.append("Disposition tags: " + ", ".join(self.dispositions))
        if self.recent_events:
            recent = [(e["text"] if isinstance(e, dict) else str(e))
                      for e in self.recent_events[:2]]
            bits.append("Recent: " + " | ".join(recent))
        return " ".join(bits)


def load_persona(path: Optional[str | Path]) -> Optional[Persona]:
    """Convenience: returns None if path is None or file missing."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return Persona.load(p)
