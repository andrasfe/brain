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
from ..world_model import render_state


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

    def next_thought(self, ws: Workspace, effectors: List[str],
                     world_model=None) -> ThoughtUnit:
        """Generate the next unit of the autoregressive thought chain.

        If a `world_model` (WorldModelStore) is provided, the PFC peeks at
        the learned forward model BEFORE composing this thought. When the
        spotlight indicates an action is likely (recent tentative_plan,
        nothing on the stack, etc.), the WM is queried with the current
        state and the most-likely action; the top matches are surfaced in
        the prompt as 'last time you saw this state and did X, the result
        was Y'. That makes `expected_result` a *learned* prediction, not a
        guess from text alone."""
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

        # ── world-model peek (the learned-prediction lane) ─────────────────
        wm_hint = ""
        if world_model is not None and effectors:
            # Query speculatively against the most-likely effector — the
            # last action verb we've considered, or the first non-internal
            # effector. The result is a *cue* to the prompt, not a commit.
            speculative_action = _speculative_action(ws, effectors)
            if speculative_action:
                state_text = render_state(ws)
                try:
                    hits = world_model.predict(state_text, speculative_action,
                                                 k=2, min_score=0.08)
                except Exception:
                    hits = []
                if hits:
                    lines = []
                    for h in hits[:2]:
                        outcome = (h.get("outcome_text") or "")[:100]
                        lines.append(
                            f"  - past similar state + {h['action_text'][:50]} "
                            f"→ {'ok' if h.get('ok') else 'ERR'}: {outcome} "
                            f"(sim={h['score']:.2f})")
                    wm_hint = ("\nLEARNED FORWARD MODEL (k-NN over past "
                               "experience):\n" + "\n".join(lines) +
                               "\nIf you emit kind='action', let these "
                               "ground your `expected_result`.")

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
            f"{voice}{intrusion_hint}{wm_hint}\n\n"
            f"{action_rules}\n"
            "Effector arg schemas:\n"
            "  read_file{path}; write_file{path,content}; list_dir{path};\n"
            "  shell{command}; web_fetch{url}; think{note};\n"
            "  remind_self{content, trigger, pattern}; finish{answer}\n\n"
            f"Produce the NEXT single thought-unit (step {step}) in your inner "
            "monologue. Keep `content` to 1-2 sentences; do not narrate your "
            "reasoning in `content`. Return JSON shaped EXACTLY like:\n"
            '{"content": "the thought, first person, 1-2 sentences",\n'
            ' "kind": "reflect|recall|appraise|tentative_plan|action|finish|tangent",\n'
            ' "effector": "<exact name from the available list — ONLY when kind=action>",\n'
            ' "args": {<args for the effector, or {} for finish>},\n'
            ' "expected_result": "~15 words predicting effector result; omit for non-action",\n'
            ' "confidence": 0.0-1.0}\n'
            "The `effector` field must be exactly one of the available names — "
            "DO NOT invent verbs (no 'go for a walk', no 'send_text', no "
            "'shut_laptop'). To rehearse an action mentally, use "
            "kind='tentative_plan' and put the draft in `content`."
        )
        # qwen-35b a3b and other reasoning models need more room: they
        # frequently exhaust 600 tokens on chain-of-thought before emitting
        # the JSON. 1500 gives reasoning headroom; chat_json's retry handles
        # the tail-of-the-distribution failures.
        out = self._chat_json(prompt, temperature=temp, max_tokens=1500)

        kind = str(out.get("kind") or "reflect").strip().lower()
        if kind not in _KINDS:
            kind = "reflect"
        content = (out.get("content") or "").strip()
        # Fall back: when content is empty but the model produced raw text
        # (e.g. reasoning-content overflow), use a short slice of that as
        # the thought rather than a "(blank thought)" placeholder.
        if not content:
            raw = (out.get("_raw") or "").strip()
            if raw:
                # Take the last non-trivial sentence as the thought
                import re as _re
                sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", raw)
                          if 12 < len(s.strip()) < 240]
                if sents:
                    content = sents[-1]
        if not content:
            content = "(blank thought)"

        # Effector resolution — accept multiple JSON shapes the model might
        # emit (top-level `effector`, nested `args.effector`, or `args.name`).
        # If kind=='action' but the resolved effector isn't in the available
        # list, demote to 'tentative_plan' so no bogus verb gets dispatched.
        if kind == "action":
            requested = _resolve_effector(out)
            if requested not in effectors:
                kind = "tentative_plan"
            else:
                # Normalize the unit args so the orchestrator's lookups work:
                # unit.args["effector"] holds the verb, unit.args["args"]
                # holds the verb's own args.
                raw_args = out.get("args")
                inner = raw_args.get("args") if isinstance(raw_args, dict) else {}
                # Pull common nested forms (effector args under the effector
                # name, e.g. args.shell.command). Best-effort.
                if isinstance(raw_args, dict) and not inner:
                    nested = raw_args.get(requested)
                    if isinstance(nested, dict):
                        inner = nested
                    elif isinstance(raw_args, dict):
                        # Default: take all non-effector keys as the args
                        inner = {k: v for k, v in raw_args.items()
                                 if k not in ("effector", "name")}
                out["args"] = {"effector": requested,
                                "args": inner if isinstance(inner, dict) else {}}

        unit = ThoughtUnit(
            step=step,
            source=self.name,
            content=content,
            kind=kind,
            args=out.get("args") or {},
            affect_snapshot=a.mood_label,
            interrupted=bool(recent_intrusion),
        )
        # Stash the predicted action result so the orchestrator can compute
        # surprise after execution (predictive coding signal).
        if kind == "action":
            expected = (out.get("expected_result") or "").strip()
            if expected:
                ws.last_prediction = expected
            else:
                ws.last_prediction = None
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


def _resolve_effector(out: dict) -> str:
    """Extract an effector verb from a thought-unit response, tolerating
    several shapes the model might emit. Returns "" if nothing parseable."""
    # Top-level field
    top = out.get("effector")
    if isinstance(top, str) and top.strip():
        return top.strip()
    args = out.get("args")
    if isinstance(args, dict):
        for key in ("effector", "name", "verb", "action"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # args as bare string (some models do this)
    if isinstance(args, str) and args.strip():
        # First token (verbs are short) — but only if it looks plausible
        first = args.strip().split()[0]
        if first.isidentifier() and len(first) <= 32:
            return first
    return ""


def _speculative_action(ws: Workspace, effectors: List[str]) -> str:
    """Best guess at the next action verb for world-model peek. Uses the
    last tentative_plan / action in the thought chain whose effector is in
    the available list; otherwise empty."""
    from ..world_model import render_action
    for t in reversed(ws.thought_chain):
        args = t.args or {}
        eff = args.get("effector") or t.kind
        inner = args.get("args") or {}
        if eff in effectors and eff not in ("think", "finish"):
            return render_action(eff, inner if isinstance(inner, dict) else {})
    return ""


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
