"""Screen-reading knowledge — the teacher-curated "where to look" skill.

The local vision model (the *student*) is weak and mis-reads screens — e.g.
guessing "VS Code" when the focused app is actually LM Studio. Two corrections:

  1. **Ground truth, not guessing.** On macOS the frontmost app is known
     deterministically (the bold name in the top-left menu bar, read via
     osascript). The observer injects it so the student never has to guess the
     app — only describe the *activity* within it.

  2. **A curated skill.** This module holds a teacher-authored set of
     "where to look on screen" rules (SEED_RULES), plus a learned file the
     in-sleep VisionTeacher (the strong executive model) appends to when it
     sees the student err. Both are injected into the student's prompt.

The seed is authored by the teacher (Claude); the learned file grows from the
student's own behavior. Learned rules live next to the memory DB (runtime,
git-ignored); the seed ships in the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

# Teacher-authored baseline. macOS-specific, concrete, where-to-look.
SEED_RULES: List[str] = [
    "The focused app is the BOLD name in the top-left menu bar, right of the  logo — trust it over app-looking content in the window.",
    "Two apps can look alike (editors, terminals, chat UIs); use the menu bar app name + window title to disambiguate, not vibes.",
    "The current document/file/tab is usually in the window's title bar (top center).",
    "A dark editor full of code is NOT necessarily VS Code — many apps embed code views; check the menu bar.",
    "Describe the ACTIVITY (what the user is doing), not just the app chrome.",
    "If unsure what the user is doing, say so briefly rather than inventing specifics.",
    "Never transcribe passwords, secrets, tokens, or full private messages.",
]

_MAX_LEARNED = 30


def _learned_path(db_path) -> Path:
    return Path(db_path).parent / "screen_reading_learned.txt"


def load_rules(db_path, limit: int = 20) -> List[str]:
    """Seed rules + learned rules (most recent learned first), capped."""
    rules = list(SEED_RULES)
    lp = _learned_path(db_path)
    if lp.exists():
        try:
            learned = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
            rules = rules + list(reversed(learned))
        except OSError:
            pass
    return rules[:limit]


def render_rules(db_path, limit: int = 12) -> str:
    return "\n".join(f"- {r}" for r in load_rules(db_path, limit=limit))


def append_rule(db_path, rule: str) -> bool:
    """Append a learned rule if it's new (dedup by normalized text). Returns
    True if added. Caps the learned file to the most recent _MAX_LEARNED."""
    rule = " ".join((rule or "").strip().split())
    if len(rule) < 8:
        return False
    lp = _learned_path(db_path)
    existing = []
    if lp.exists():
        try:
            existing = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
        except OSError:
            existing = []
    def _norm(s: str) -> str:
        return " ".join((s or "").split()).lower().rstrip(".")
    norm = _norm(rule)
    seed_norm = {_norm(s) for s in SEED_RULES}
    if norm in seed_norm or any(norm == _norm(e) for e in existing):
        return False
    existing.append(rule)
    existing = existing[-_MAX_LEARNED:]
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(existing) + "\n")
        return True
    except OSError:
        return False
