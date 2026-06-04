"""Prefrontal cortex — autoregressive stream-of-thought generator.

The analogue of next-token prediction: at each step, produce the NEXT thought
conditioned on (a) the full prior thought chain, (b) the current AffectState,
(c) the conscious workspace. One thought-unit at a time; the chain commits
when the unit's `kind` is 'action' (then gated/acted) or 'finish'.

Between thought-units the orchestrator ticks other regions — interoception,
amygdala, locus coeruleus, DMN. They may append their own units to the chain
(intrusions / tangents) which the next prefrontal step then sees in its
context. THAT is the interruption mechanism: a hijacked stream, not a vetoed
decision.

The thought voice is first-person, in character with the persona, colored by
mood. The PFC is allowed to be sloppy — humans are.
"""
from __future__ import annotations

from typing import List, Optional

from ..region import Region
from ..workspace import Broadcast, ThoughtUnit, Workspace


# Allowed thought kinds. Keep the schema small.
_KINDS = {"reflect", "recall", "appraise", "tentative_plan",
          "action", "finish", "tangent"}


class Prefrontal(Region):
    name = "prefrontal"
    system_prompt = (
        "You are the PREFRONTAL CORTEX of a human-like brain — the executive. "
        "You think one thought at a time, like a stream of consciousness: each "
        "thought continues from the prior ones in your inner monologue. You are "
        "writing in the FIRST PERSON. Your mood, fatigue, hunger, and recent "
        "events color the thought; sometimes the chain gets hijacked by an "
        "intrusive memory or a sudden body signal and you have to recover. "
        "When you have arrived at what to do, emit a thought of kind 'action' "
        "with effector+args. When the task is resolved, emit kind 'finish' "
        "with the final answer. Otherwise: 'reflect', 'recall', 'appraise', "
        "'tentative_plan', or 'tangent'."
    )

    def next_thought(self, ws: Workspace, effectors: List[str]) -> ThoughtUnit:
        """Generate the next unit of the autoregressive thought chain."""
        a = ws.affect
        step = len(ws.thought_chain) + 1

        # Temperature / verbosity vary with affect — same idea as before but
        # applied per thought-unit (so the *chain itself* varies in fluency).
        temp = 0.55 + 0.35 * a.distractibility - 0.20 * max(0.0, a.stress - 0.5)
        temp = max(0.2, min(1.15, temp))

        voice = _voice_line(a, ws.interrupt)
        recent_intrusion = _last_intrusion(ws.thought_chain)
        intrusion_hint = (
            f"\nNOTE: your last thought was interrupted by "
            f"[{recent_intrusion.source}] '{recent_intrusion.content[:120]}'. "
            "Either pick up where you left off, or let it pull you."
            if recent_intrusion else ""
        )

        # Be explicit about whether external actions are even possible. If the
        # only available effectors are think/finish, the model should NOT
        # invent verbs like 'call' or 'send_text' — it should stay in the
        # mental-rehearsal lane (tentative_plan / reflect) and emit `finish`
        # when ready with the answer.
        nontrivial_eff = [e for e in effectors if e not in ("think", "finish")]
        action_rules = (
            f"You may emit kind='action' ONLY with effector ∈ {effectors}; the "
            "effector string must match EXACTLY one of those names — do not "
            "invent 'call', 'send_text', 'reply', etc. If you mean to compose "
            "a message or rehearse a reply mentally, use kind='tentative_plan' "
            "with the draft in `content`."
        )
        if not nontrivial_eff:
            action_rules += (
                " No external tools are available here — keep deliberating "
                "with reflect / recall / appraise / tentative_plan and emit "
                "kind='finish' with the final answer in args.answer when you "
                "are ready."
            )

        prompt = (
            ws.render_context(limit=8) + "\n\n"
            f"{voice}{intrusion_hint}\n\n"
            f"{action_rules}\n"
            "Effector arg schemas:\n"
            "  read_file{path}; write_file{path,content}; list_dir{path};\n"
            "  shell{command}; web_fetch{url}; think{note}; finish{answer}\n\n"
            f"Produce the NEXT single thought-unit (step {step}) in your inner "
            "monologue. Keep it short (one or two sentences). Return JSON: "
            '{"content": "the thought, first person", '
            '"kind": "reflect|recall|appraise|tentative_plan|action|finish|tangent", '
            '"args": {} (only if kind=action or finish), '
            '"confidence": 0.0-1.0}'
        )
        out = self._chat_json(prompt, temperature=temp, max_tokens=600)

        kind = str(out.get("kind") or "reflect").strip().lower()
        if kind not in _KINDS:
            kind = "reflect"
        content = (out.get("content") or "").strip()
        if not content:
            content = "(blank thought)"

        # Guard: if model emitted kind='action' but the effector isn't in the
        # available list, demote to 'tentative_plan' rather than execute a
        # made-up verb. The chain keeps moving; no fake action gets dispatched.
        if kind == "action":
            requested = ((out.get("args") or {}).get("effector")
                         or (out.get("args") or {}).get("name") or "")
            if str(requested) not in effectors:
                kind = "tentative_plan"

        unit = ThoughtUnit(
            step=step,
            source=self.name,
            content=content,
            kind=kind,
            args=out.get("args") or {},
            affect_snapshot=a.mood_label,
            interrupted=bool(recent_intrusion),
        )
        ws.thought_chain.append(unit)

        # Mirror onto the broadcast feed so the spotlight machinery still sees it.
        conf = float(out.get("confidence") or 0.6)
        salience = {
            "action": 0.85, "finish": 0.85, "appraise": 0.6, "tangent": 0.5,
        }.get(kind, 0.6 + 0.2 * conf)
        ws.post(Broadcast(
            source=self.name, kind=f"thought:{kind}",
            content=f"[{step:02d}] {content[:160]}",
            salience=salience, data={"unit": _unit_to_dict(unit),
                                       "confidence": conf},
        ))
        return unit


# ── helpers ────────────────────────────────────────────────────────────────
def _voice_line(a, interrupt: Optional[str]) -> str:
    bits = [f"Your current inner state: mood={a.mood_label} "
            f"(val={a.valence:+.2f}, arou={a.arousal:.2f}, stress={a.stress:.2f}, "
            f"fatigue={a.fatigue:.2f}, hunger={a.hunger:.2f}, "
            f"bored={a.boredom:.2f}, reward={a.reward_tone:+.2f})."]
    if interrupt:
        bits.append(f"AMYGDALA INTERRUPT: {interrupt} — address this or you'll "
                    "feel it pulling.")
    if a.stress > 0.65:
        bits.append("You feel rushed. Thoughts are clipped.")
    if a.fatigue > 0.7:
        bits.append("You are tired; coherence wavers.")
    if a.boredom > 0.7:
        bits.append("It's hard to stay on task; you may drift.")
    if a.reward_tone > 0.3:
        bits.append("You feel a small lift; willing to try.")
    if a.reward_tone < -0.3:
        bits.append("You feel discouraged.")
    if a.curiosity > 0.7 and a.stress < 0.5:
        bits.append("Curious — willing to look something up.")
    return " ".join(bits)


def _last_intrusion(chain: List[ThoughtUnit]) -> Optional[ThoughtUnit]:
    """Look back at the last thought-unit; if it was from a non-prefrontal
    source, the *next* prefrontal step has been interrupted by it."""
    if not chain:
        return None
    last = chain[-1]
    return last if last.source != "prefrontal" else None


def _unit_to_dict(u: ThoughtUnit) -> dict:
    return {"step": u.step, "source": u.source, "kind": u.kind,
            "content": u.content, "args": u.args,
            "interrupted": u.interrupted,
            "affect_snapshot": u.affect_snapshot}
