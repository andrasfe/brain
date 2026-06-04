"""Orchestrator — autoregressive stream-of-thought reasoning loop.

The prior version was a discrete pipeline (perceive → recall → appraise →
plan → gate → act) repeated each cycle. This version is **autoregressive over
thought-units**, analogous to next-token prediction:

  perceive (once)
  loop:
    world.tick   → ambient stimuli into workspace
    interoception → body→affect
    amygdala     → valence/stress writes, maybe interrupt
    locus_coeruleus → arousal
    hippocampus  → mood-congruent recall (every K steps)
    DMN          → maybe inject a tangent INTO THE CHAIN (the hijack)
    prefrontal.next_thought(chain, ws) → one ThoughtUnit appended to chain
    if unit.kind == "action":
      basal_ganglia.gate(unit) → if approved: act, then VTA reward signal
    elif unit.kind == "finish":
      stop
    affect.decay_toward_baseline (slow homeostasis)
  speak (broca) → final voice (colored by current mood)

The chain IS the reasoning. Other regions tick between thought-units and can
append their own ThoughtUnits (DMN tangents, amygdala intrusions), which the
prefrontal sees in its next-step context. That is what makes interruption
visible *inside* the thought stream, not just between turns.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .affect import AffectState, Traits
from .config import Config
from . import consolidator as _consolidator
from .effectors import Effectors
from .embeddings import make_backend
from .llm import LLM
from .memory import EPISODIC, Memory, SEMANTIC
from .persona import Persona, load_persona
from .skills import SkillStore, prediction_surprise, signature_from_percept
from .world_model import WorldModelStore, render_action, render_state
from .regions import (
    Amygdala,
    BasalGanglia,
    Broca,
    Cerebellum,
    DefaultMode,
    Hippocampus,
    Interoception,
    LocusCoeruleus,
    Prefrontal,
    SensoryCortex,
    VTA,
)
from .workspace import ActionRecord, Broadcast, ThoughtUnit, Workspace
from .world import World


# How often (in thought-units) to re-pull memory and re-tick the world.
# Memory recall is expensive (DB hit) and world events should be rarer than
# thoughts (a person doesn't get a new notification every internal sentence).
_RECALL_EVERY = 3
_WORLD_EVERY = 2


class Brain:
    def __init__(self, cfg: Config, confirm: Callable[[str], bool] | None = None,
                 log: Callable[[str], None] | None = None,
                 persona: Optional[Persona] = None,
                 scenario: Optional[str] = None,
                 humanize: bool = True,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.log = log or (lambda _m: None)
        self.llm = LLM(cfg)
        # Embedding backend (pluggable). Defaults to TF-IDF; switches to
        # sentence-transformers / OpenRouter if configured.
        backend = make_backend(cfg, llm=self.llm)
        self.memory = Memory(cfg.db_path, backend=backend)
        # The skill store shares the memory db on purpose: skills are a kind
        # of procedural memory and persist across runs alongside episodes.
        self.skills = SkillStore(cfg.db_path)
        # World model — k-NN over (state, action, outcome) in the configured
        # embedding space. The PFC consults it to ground its expected_result
        # predictions; the orchestrator writes a row after every action.
        self.world_model = WorldModelStore(cfg.db_path, backend=backend)
        # Effectors get a memory handle so `remind_self` can register
        # prospective items the hippocampus will surface later.
        self.effectors = Effectors(cfg, confirm=confirm, memory=self.memory)
        self.humanize = humanize
        self.seed = seed

        raw = cfg.raw
        if persona is None:
            persona = load_persona(raw.get("persona_path"))
        self.persona = persona
        self.scenario = scenario or raw.get("scenario") or "neutral"
        self.world = World(self.scenario, seed=seed) if humanize else None

        # Core regions
        self.sensory = SensoryCortex(cfg, self.llm)
        self.amygdala = Amygdala(cfg, self.llm)
        self.hippocampus = Hippocampus(cfg, self.llm, self.memory)
        self.prefrontal = Prefrontal(cfg, self.llm)
        self.basal_ganglia = BasalGanglia(cfg, self.llm)
        self.broca = Broca(cfg, self.llm)

        # Humanization regions
        if humanize:
            self.interoception = Interoception(cfg, self.llm)
            self.default_mode = DefaultMode(cfg, self.llm, seed=seed)
            self.locus_coeruleus = LocusCoeruleus(cfg, self.llm)
            self.vta = VTA(cfg, self.llm)
        # Cerebellum: deterministic, no LLM. Always built (cheap), always
        # reads the world model. Lives outside the humanize switch because
        # it's the fast path for habit-fire decisions either way.
        self.cerebellum = Cerebellum(cfg, self.llm,
                                       world_model=self.world_model)

        # Seed memory with persona priors
        if persona is not None:
            self._seed_persona_memory(persona)

    def close(self) -> None:
        self.llm.close()
        self.memory.close()
        self.skills.close()
        self.world_model.close()

    def _seed_persona_memory(self, persona: Persona) -> None:
        for seed in persona.memory_seeds():
            self.memory.store(
                task="persona", kind=seed["kind"],
                content=seed["content"], salience=float(seed["salience"]),
            )

    # ── main entry point ──────────────────────────────────────────────────────
    def run(self, task: str) -> str:
        cfg_loop = self.cfg.loop
        affect = self._build_initial_affect()
        ws = Workspace(
            task=task,
            decay=float(cfg_loop.get("attention_decay", 0.85)),
            affect=affect,
        )
        # `max_cycles` is now reinterpreted as max thought-units, the analogue
        # of max generated tokens in an autoregressive decoder. The orchestrator
        # ticks regions between each unit.
        max_units = int(cfg_loop.get("max_cycles", 12))

        self.log(f"⟶ perceiving task  [{affect.render()}]")
        self.sensory.step(ws)

        if self.persona is not None:
            ws.post(Broadcast(
                source="self_model", kind="identity",
                content=self.persona.render_identity_block(),
                salience=0.6,
            ))

        # ── stream of thought ──────────────────────────────────────────────
        commit: Optional[ThoughtUnit] = None
        step = 0
        # The orchestrator's `cycle` advances with each thought-unit so that
        # affect timestamps and consolidation continue to make sense.
        while step < max_units:
            step += 1
            ws.cycle = step
            self.log(f"\n── step {step} ──────────────────────────────")

            # ── world / body tick (rate-limited) ────────────────────────────
            if self.humanize and self.world is not None and step % _WORLD_EVERY == 1:
                stims = self.world.tick()
                for s in stims:
                    ws.post(Broadcast(
                        source="world", kind=s.kind, content=s.content,
                        salience=s.salience, data=s.data,
                    ))
                    if s.affect_delta:
                        ws.affect.update("world", step, s.affect_delta,
                                          smoothing=0.8)
                self.interoception.step(ws)

            # ── recall (rate-limited) ──────────────────────────────────────
            if step == 1 or step % _RECALL_EVERY == 0:
                self.hippocampus.step(ws)

            # ── appraise (every step — emotion is continuous) ──────────────
            self.amygdala.step(ws)
            if ws.interrupt:
                self.log(f"  ! amygdala interrupt: {ws.interrupt}")

            # ── arousal modulation ─────────────────────────────────────────
            if self.humanize:
                self.locus_coeruleus.step(ws)

            # ── DMN: maybe hijack the chain (appends a tangent unit) ───────
            if self.humanize:
                t = self.default_mode.step(ws)
                if t:
                    self.log(f"  ~ mind wanders: {t.content[:80]}")
                    # IMPORTANT: when DMN appended a unit, the chain has grown
                    # without the prefrontal speaking. The next prefrontal
                    # step will see the tangent in its context and either
                    # recover or drift — that's the hijack. We still want it
                    # to think this step, but step++ would skip ahead too far.
                    # So we leave step as-is and let the prefrontal speak next.

            # spotlight
            spotlight = ws.tick_attention()
            if spotlight:
                self.log(f"  ◎ spotlight: [{spotlight.source}/{spotlight.kind}] "
                         f"{spotlight.content[:80]}")

            # ── direct-path habit-fire (System-1, no LLM) ──────────────────
            # The basal ganglia consults the SkillStore for a cached action
            # for this percept signature. If a fireable skill exists AND the
            # psychological conditions allow (no interrupt, low surprise,
            # cognitive load, low curiosity), we skip the prefrontal entirely.
            ws.habit_fired = False
            habit_proposal = self.basal_ganglia.propose_habit(
                ws, self.skills, cerebellum=self.cerebellum)
            if habit_proposal is not None:
                self.log(f"  ⚡ habit fires (no PFC): {habit_proposal['effector']} "
                         f"{habit_proposal['reasoning']}")
                ws.habit_fired = True
                # Capture state-at-decision BEFORE we mutate workspace with
                # the synthetic action ThoughtUnit, so the world-model row
                # reflects the moment the action was selected.
                _wm_state = render_state(ws)
                _wm_action = render_action(habit_proposal["effector"],
                                            habit_proposal["args"])
                # Append a synthetic ThoughtUnit so the chain still reads as
                # a continuous stream (BG-sourced, kind=action).
                ws.thought_chain.append(ThoughtUnit(
                    step=len(ws.thought_chain) + 1,
                    source="basal_ganglia",
                    content=f"(reflex) {habit_proposal['reasoning']}",
                    kind="action",
                    args={"effector": habit_proposal["effector"],
                           "args": habit_proposal["args"]},
                    affect_snapshot=ws.affect.mood_label,
                    interrupted=False,
                ))
                eff = habit_proposal["effector"]
                args = habit_proposal["args"]
                ok, result = self.effectors.execute(eff, args)
                self.log(f"  ▶ {eff} -> {'ok' if ok else 'ERR'}: {result[:80]}")
                ws.history.append(ActionRecord(
                    cycle=step, effector=eff, args=args,
                    result=result, ok=ok))
                ws.post(Broadcast(
                    source="motor", kind="result",
                    content=f"{eff} {'ok' if ok else 'FAILED'}: {result[:200]}",
                    salience=0.75 if ok else 0.85,
                ))
                ws.thought_chain.append(ThoughtUnit(
                    step=len(ws.thought_chain) + 1, source="motor",
                    content=f"{eff} -> {'ok' if ok else 'ERR'}: {result[:160]}",
                    kind="result", affect_snapshot=ws.affect.mood_label,
                ))
                # Skill update on the habit-fire outcome itself
                sig = ws.last_habit_signature or "<unknown>"
                self.skills.consolidate(sig, eff, args, ok, outcome=result)
                # World-model observation: the habit-fire is real experience
                # too. Salience is mild because no surprise to learn from.
                try:
                    self.world_model.observe(
                        _wm_state, _wm_action, result, ok=ok,
                        source="observed", salience=0.5)
                except Exception as e:  # noqa: BLE001 — never fail a run
                    self.log(f"  ⚠ world_model.observe skipped: {e}")
                if not ok:
                    # The cached habit failed — decay its confidence so the
                    # next encounter falls back to System-2.
                    self.skills.punish(habit_proposal["_skill_id"])
                if self.humanize:
                    self.vta.step(ws)
                self.hippocampus.consolidate(
                    ws, kind="action",
                    content=f"(habit) {eff}({args}) -> "
                            f"{'ok' if ok else 'err'}: {result[:200]}",
                    salience=0.6 if ok else 0.8,
                )
                # Habit-fire had no prediction, so no surprise this cycle
                ws.last_prediction = None
                ws.last_surprise = 0.0
                ws.affect.decay_toward_baseline()
                continue

            # ── prefrontal next thought (the autoregressive step) ──────────
            unit = self.prefrontal.next_thought(
                ws, self.effectors.available(),
                world_model=self.world_model)
            self.log(f"  • thought[{unit.step:02d} {unit.kind}]"
                     f"{' ⟪after intrusion⟫' if unit.interrupted else ''}: "
                     f"{unit.content[:100]}")

            # ── commit gates ───────────────────────────────────────────────
            if unit.kind == "finish":
                ws.final_output = (unit.args or {}).get("answer") or unit.content
                ws.done = True
                commit = unit
                self.log("  ✓ finish")
                break

            if unit.kind == "action":
                # basal ganglia gates the action (may veto OR repair args)
                proposal = {"effector": (unit.args or {}).get("effector", "think"),
                             "args": (unit.args or {}).get("args") or unit.args,
                             "reasoning": unit.content}
                gate = self.basal_ganglia.step(ws, proposal)
                if gate.data.get("decision") != "go":
                    self.log(f"  ⊘ basal ganglia veto: "
                             f"{gate.data.get('reason', '')[:80]}")
                    ws.post(Broadcast(
                        source="basal_ganglia", kind="result",
                        content=f"action vetoed: {gate.data.get('reason', '')}",
                        salience=0.65,
                    ))
                    # Treat veto as a continuation: the chain keeps going.
                    ws.affect.decay_toward_baseline()
                    continue

                eff = gate.data.get("effector", proposal["effector"])
                args = gate.data.get("args") or proposal["args"]

                # Defense-in-depth: BG can "repair" effector and sometimes
                # repairs it INTO a bogus verb (e.g. "go for a walk" rather
                # than a real effector name). Validate before dispatch and
                # downgrade to 'think' if it isn't a registered effector.
                # We post a workspace broadcast so the trace shows it.
                if eff not in self.effectors.available():
                    self.log(f"  ⚠ invalid effector after BG: {eff!r} → think")
                    ws.post(Broadcast(
                        source="orchestrator", kind="invalid_effector",
                        content=f"invalid effector dropped: {eff!r} "
                                 "(downgraded to think)",
                        salience=0.5,
                    ))
                    args = {"note": f"intended: {eff} {args}"}
                    eff = "think"

                # Capture state-at-decision BEFORE acting (the WM row should
                # describe the situation that LED to this choice).
                _wm_state = render_state(ws)
                _wm_action = render_action(eff, args)

                ok, result = self.effectors.execute(eff, args)
                self.log(f"  ▶ {eff} -> {'ok' if ok else 'ERR'}: {result[:80]}")
                ws.history.append(ActionRecord(
                    cycle=step, effector=eff, args=args, result=result, ok=ok))
                ws.post(Broadcast(
                    source="motor", kind="result",
                    content=f"{eff} {'ok' if ok else 'FAILED'}: {result[:200]}",
                    salience=0.75 if ok else 0.85,
                ))
                # the motor result is also injected into the chain so the next
                # thought sees the outcome
                ws.thought_chain.append(ThoughtUnit(
                    step=len(ws.thought_chain) + 1,
                    source="motor",
                    content=f"{eff} -> {'ok' if ok else 'ERR'}: {result[:160]}",
                    kind="result",
                    interrupted=False,
                    affect_snapshot=ws.affect.mood_label,
                ))

                # ── predictive coding: compare expected_result to actual ──
                # The prefrontal stashed its prediction in ws.last_prediction
                # when it emitted this action. We compute trigram surprise
                # and use it to (a) modulate LC arousal, (b) post a
                # prediction_error broadcast that competes for the spotlight
                # next cycle, (c) break habit-fire on the following step.
                surprise = prediction_surprise(ws.last_prediction or "", result)
                ws.last_surprise = surprise
                if ws.last_prediction and surprise > 0.0:
                    self.log(f"  ◊ prediction surprise: {surprise:.2f} "
                             f"(expected: {ws.last_prediction[:60]})")
                if surprise > 0.55:
                    ws.post(Broadcast(
                        source="prediction_error", kind="surprise",
                        content=f"surprise={surprise:.2f}: expected "
                                f"\"{(ws.last_prediction or '')[:80]}\" but got "
                                f"\"{result[:80]}\"",
                        salience=0.7 + 0.2 * surprise,
                        data={"surprise": surprise,
                               "expected": ws.last_prediction, "actual": result},
                    ))
                    # Surprise spikes arousal directly (LC-style)
                    ws.affect.update("prediction_error", step,
                                      {"arousal": +0.06,
                                        "stress": +0.03 * (surprise - 0.55)},
                                      smoothing=0.7)
                ws.last_prediction = None  # consumed

                if self.humanize:
                    self.vta.step(ws)

                # ── skill compilation (System-2 → System-1) ──
                # Every action contributes to the skill cache: successes raise
                # confidence (EMA), failures lower it. After enough successful
                # repetitions of the same signature the BG can fire this skill
                # next time without invoking the prefrontal at all.
                percept = ws.latest(kind="percept")
                sig = signature_from_percept(
                    (percept.data if percept else {}) or {}, ws.interrupt)
                self.skills.consolidate(sig, eff, args, ok, outcome=result)

                # ── world-model observation (LeCun-style passive learning) ──
                # Salience scaled by surprise so high-prediction-error
                # observations are weighted more in future predict() ranking.
                try:
                    wm_sal = 0.45 + 0.5 * max(0.0, surprise)
                    self.world_model.observe(
                        _wm_state, _wm_action, result, ok=ok,
                        source="observed", salience=min(0.95, wm_sal))
                except Exception as e:  # noqa: BLE001
                    self.log(f"  ⚠ world_model.observe skipped: {e}")

                self.hippocampus.consolidate(
                    ws, kind="action",
                    content=f"{eff}({args}) -> {'ok' if ok else 'err'}: {result[:200]}",
                    salience=0.6 if ok else 0.8,
                )

            # slow homeostasis between thoughts
            ws.affect.decay_toward_baseline()

        # ── speak ──
        # Broca always runs. The PFC's `finish` with args.answer is a
        # commitment hint, NOT the user-facing answer — Broca produces
        # mood-colored prose over the whole thought chain. (The hint is
        # already visible to Broca via render_context's thought-chain
        # lines, so we don't need to thread it explicitly.)
        self.log(f"\n⟶ synthesizing final answer (broca)  [{ws.affect.render()}]")
        answer = self.broca.step(ws)
        self.hippocampus.consolidate(ws, kind="outcome",
                                      content=answer[:500], salience=0.7)

        # End-of-task consolidation pass: extract recurring patterns from
        # recent episodic rows into durable semantic facts. Small and cheap
        # here (≤ 2 facts); `python -m brain.consolidator` does deeper sleep.
        try:
            stats = _consolidator.consolidate(
                self.memory, self.llm,
                model=self.cfg.models.get("reflex"),
                n_episodes=int(self.cfg.memory.get("consolidate_window", 40)),
                min_cluster_size=int(self.cfg.memory.get("consolidate_min_cluster", 3)),
                max_new_facts=int(self.cfg.memory.get("consolidate_max_facts", 2)),
                log=self.log,
            )
            if stats.get("facts_written"):
                self.log(f"  ⤷ consolidator wrote {stats['facts_written']} "
                         f"semantic fact(s) from {stats['rows_tagged']} episode(s)")
        except Exception as e:  # noqa: BLE001 — never let consolidation fail a run
            self.log(f"  ⚠ consolidator skipped: {type(e).__name__}: {e}")
        return answer

    # ── helpers ──────────────────────────────────────────────────────────────
    def _build_initial_affect(self) -> AffectState:
        traits = self.persona.derive_traits() if self.persona else Traits()
        affect = AffectState(traits=traits)
        affect.stress = max(affect.stress,
                            0.10 + 0.25 * (traits.neuroticism - 0.5))
        affect.curiosity = max(0.2, 0.30 + 0.50 * traits.openness)
        affect.social_need = max(affect.social_need,
                                 0.15 + 0.25 * (traits.extraversion - 0.5))
        if self.persona is not None:
            ev = self.persona.initial_affect_deltas()
            if ev:
                affect.update("persona:init", 0, ev, smoothing=0.5)
        if self.world is not None:
            sc = self.world.initial_affect_deltas()
            if sc:
                affect.update("world:init", 0, sc, smoothing=0.5)
        return affect
