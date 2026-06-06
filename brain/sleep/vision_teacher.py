"""VisionTeacher (NREM) — the teacher curates the student's screen-reading skill.

Teacher-student: the *student* is the weak local vision model that describes
screenshots during the day; the *teacher* is the strong executive model
(qwen-35b). During sleep the teacher reviews a batch of the student's recent
descriptions and distills new "where to look on screen" rules, appending them
to the curated skill (`brain/knowledge.py`). Those rules are injected into the
student's prompt thereafter — so the student gets better at reading screens
without being retrained.

This complements the deterministic fix (the frontmost app is injected as
ground truth): the teacher catches the *softer* recurring mistakes (mislabeled
activity, confusing look-alike apps) and writes durable guidance for them.

Bounded: a few rules per bout, deduped + capped in the knowledge store.
No-ops below `min_obs` or without an executive model.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..knowledge import append_rule
from ..memory import OBSERVATION

_TEACHER_SYSTEM = (
    "You are a TEACHER improving a weak vision model that reads macOS screens. "
    "You output only concise, general 'where to look' rules — never commentary."
)

_MARKER = re.compile(r"^\s*(\d+[.)]|[-*+])\s+")


class VisionTeacher:
    name = "vision_teacher"

    def __init__(self, window: int = 40, sample: int = 25,
                 max_new_rules: int = 2, min_obs: int = 10):
        self.window = window
        self.sample = sample
        self.max_new_rules = max_new_rules
        self.min_obs = min_obs

    def run(self, memory, llm, model: Optional[str], *, db_path,
            log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        if not llm or not model:
            return {"taught": False, "reason": "no teacher model"}
        rows = memory.conn.execute(
            "SELECT content FROM episodes WHERE mem_type=? ORDER BY ts DESC LIMIT ?",
            (OBSERVATION, self.window),
        ).fetchall()
        if len(rows) < self.min_obs:
            return {"taught": False, "reason": "too few observations",
                    "n": len(rows)}
        samples = [r["content"] for r in rows][:self.sample]
        prompt = (
            "Recent one-line screen-activity descriptions produced by the weak "
            "vision model:\n"
            + "\n".join(f"- {s[:160]}" for s in samples)
            + f"\n\nPropose up to {self.max_new_rules} SHORT, GENERAL rules "
            "(<=16 words each, imperative) telling it WHERE TO LOOK to read such "
            "macOS screens more accurately — focused app, document title, "
            "activity cues, look-alike-app pitfalls. One rule per line, no "
            "numbering, no preamble. If the descriptions already look accurate "
            "and specific, output nothing."
        )
        try:
            text = llm.chat(model, _TEACHER_SYSTEM, prompt,
                            temperature=0.3, max_tokens=200)
        except Exception as e:  # noqa: BLE001 — teaching must never break sleep
            return {"taught": False, "reason": f"teacher error: {e}"}

        added = 0
        for line in (text or "").splitlines():
            line = _MARKER.sub("", line).strip().strip('"').strip("'")
            if len(line) < 8 or len(line) > 160:
                continue
            if append_rule(db_path, line):
                added += 1
                log(f"  📚 vision rule learned: {line[:80]}")
                if added >= self.max_new_rules:
                    break
        return {"taught": added > 0, "reviewed": len(samples), "rules_added": added}
