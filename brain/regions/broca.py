"""Broca's area — language production: synthesizes the final answer.

Runs once at the end. Reads the whole conscious trace and the action history and
produces the user-facing output that satisfies the success criterion.
"""
from __future__ import annotations

from ..region import Region
from ..workspace import Workspace


class Broca(Region):
    name = "broca"
    system_prompt = (
        "You are BROCA'S AREA of a human-like brain — language production. "
        "Synthesize a complete final answer to the original task, grounded in what "
        "the brain actually did. Write IN THE FIRST PERSON as the individual whose "
        "mood and recent experience are part of you. Let mood color word choice and "
        "register without ever announcing the mood. Be direct; no meta-commentary "
        "about being a brain or an AI."
    )

    def step(self, ws: Workspace) -> str:
        actions = "\n".join(
            f"- {a.effector}({a.args}) -> {a.result[:200]}" for a in ws.history
        ) or "(no external actions were taken)"
        af = ws.affect
        voice = _broca_voice(af)
        # Use chat_json so reasoning models can't leak their full chain-of-
        # thought into the final user-facing answer. We pull only the
        # `answer` field; chat_json's retry handles parse failures.
        out = self._chat_json(
            f"Original task:\n{ws.task}\n\n"
            f"Conscious workspace:\n{ws.render_context(limit=20)}\n\n"
            f"Actions taken:\n{actions}\n\n"
            f"Voice instructions: {voice}\n\n"
            "Return JSON: "
            '{"answer": "the final user-facing answer, no meta-commentary, '
            'no reasoning narration, no constraint checklists — just the '
            'message you are giving the user, written in first person."}',
            temperature=0.55 + 0.25 * af.arousal,
            max_tokens=2000,
        )
        text = (out.get("answer") or "").strip()
        if text:
            return text
        # Fall back: if JSON parse failed entirely, extract a clean answer
        # from the raw text rather than handing back the whole reasoning
        # monologue. Prefers the last well-formed paragraph block.
        raw = (out.get("_raw") or "").strip()
        return _extract_final_answer(raw) if raw else "(no answer)"


import re


_REASONING_PARA = re.compile(
    r"^\s*(\d+[.)]\s|[\*\-\+]\s|step\s+\d|process:|analyze\b|"
    r"all constraints|check\b|verify\b|deconstruct|thinking process|"
    r"self-correction|let me check|let's verify|output matches)",
    re.IGNORECASE,
)


def _extract_final_answer(text: str) -> str:
    """Heuristic: from a reasoning-model chain-of-thought leaking into the
    output, pull the actual final draft. Tries marker-based extraction
    first, then walks paragraphs from the end picking the last that doesn't
    look like a reasoning step."""
    if not text:
        return ""
    # Marker-based extraction — many reasoning models gate their final
    # answer behind labels like "Draft:" / "Output:" / "Final Answer:".
    for marker in ("**final answer:**", "**draft:**", "**output:**",
                    "**answer:**", "final answer:", "output:", "draft:",
                    "answer:"):
        idx = text.lower().rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker):].strip()
            # Strip leading/trailing quotes and code fences
            tail = tail.strip("`").strip('"').strip("'").strip()
            if tail:
                return tail
    # Paragraph-walk fallback
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in reversed(paragraphs):
        if _REASONING_PARA.match(p):
            continue
        # Short bookkeeping lines like "All good." aren't the answer
        if len(p) < 60 and re.match(r"^(all|done|good|proceeds|ok|✅)", p, re.I):
            continue
        return p
    # Last resort — return the whole thing
    return text.strip()


def _broca_voice(a) -> str:
    bits = [f"Mood is '{a.mood_label}'."]
    if a.stress > 0.6:
        bits.append("Be terse; clipped sentences; less hedging.")
    if a.fatigue > 0.65:
        bits.append("Sound a little tired; one or two minor word fumbles are fine.")
    if a.valence > 0.4:
        bits.append("Warmer register; one small flicker of enthusiasm is OK.")
    if a.valence < -0.4:
        bits.append("Flatter, drier; no false enthusiasm.")
    if a.reward_tone > 0.3:
        bits.append("Slightly proud, but don't gloat.")
    if a.reward_tone < -0.3:
        bits.append("A note of resignation is allowed.")
    if a.traits.agreeableness > 0.65:
        bits.append("Softening hedges where appropriate ('I think', 'maybe').")
    if a.traits.agreeableness < 0.35:
        bits.append("Blunt; skip social softeners.")
    if a.traits.conscientiousness > 0.65:
        bits.append("Tidy structure.")
    return " ".join(bits)
