"""BrainDaemon — long-lived event loop with a sleep/wake state machine.

States
  WAKE     processes input via the existing `Brain.run(task)` stream-of-thought.
           When idle, may emit a low-cost spontaneous thought, gated by
           AffectState.curiosity / boredom and an idle-rate config knob.
  DROWSY   transition state: only the scheduler ticks; affect drifts heavily.
  NREM     consolidator + forgetter + skill_pruner + mood_regulator run one
           bout (each ~once per NREM cycle).
  REM      dreamer runs N dreams per bout.

Real biology: 90-min sleep cycles alternating NREM/REM, REM lengthens through
the night. Simplified here as a bout sequence (NREM, NREM, REM, NREM, REM,
REM, …) where REM share grows late.

Time
  The daemon ticks on wall-clock seconds (or simulated time via
  `--tick-seconds`). Each tick advances the `World` by `world_cycle_minutes`
  simulated minutes. Sleep is triggered by AffectState.fatigue + World.hour;
  wake by accumulated sleep budget OR external input arrival.

Input channels
  - stdin lines while running interactively (`enqueue` if non-empty)
  - `BrainDaemon.enqueue(task)` programmatic API
  - time-triggered prospective items (scheduler agent) — surface on next wake

CLI
  python -m brain.daemon
  python -m brain.daemon --tick-seconds 1.0 --idle-rate 0
  python -m brain.daemon --max-ticks 200      (for tests)

This is a single-process loop — no threads, no asyncio. The brain doesn't
need either; sleep is exactly when expensive work is acceptable.
"""
from __future__ import annotations

import argparse
import queue
import select
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain import consolidator  # noqa: E402
from brain.affect import AffectState  # noqa: E402
from brain.config import load_config  # noqa: E402
from brain.inputs import (  # noqa: E402
    InputAdapter, SalienceClassifier, StdinAdapter, StreamItem,
)
from brain.inputs.classifier import ClassifierRules  # noqa: E402
from brain.orchestrator import Brain  # noqa: E402
from brain.observer import ScreenObserver  # noqa: E402
from brain.presence import idle_seconds  # noqa: E402
from brain.sleep import (  # noqa: E402
    Dreamer, Forgetter, ForwardModelTrainer, MoodRegulator, ScreenPurger,
    Scheduler, ScreenSequenceTrainer, SkillPruner, VisionTeacher,
)
from brain.status import write_status  # noqa: E402
from brain.workspace import Broadcast  # noqa: E402


# ── states ─────────────────────────────────────────────────────────────────
WAKE = "wake"
DROWSY = "drowsy"
NREM = "nrem"
REM = "rem"


@dataclass
class DaemonStats:
    state_changes: list[tuple[float, str, str]] = field(default_factory=list)
    spontaneous_thoughts: int = 0
    tasks_processed: int = 0
    sleep_bouts: dict[str, int] = field(default_factory=lambda: {"nrem": 0, "rem": 0})
    dreams_written: int = 0
    episodes_pruned: int = 0
    skills_decayed: int = 0
    skills_deleted: int = 0
    facts_consolidated: int = 0
    prospective_fired: int = 0
    forward_model_trains: int = 0
    screen_model_trains: int = 0
    vision_rules_learned: int = 0
    observations_pruned: int = 0


class BrainDaemon:
    """Long-lived brain — WAKE/DROWSY/NREM/REM state machine."""

    def __init__(self, cfg, brain: Brain, *,
                 tick_seconds: float = 1.0,
                 world_cycle_minutes: float = 6.0,
                 idle_rate: float = 0.10,
                 drowsy_fatigue: float = 0.78,
                 sleep_fatigue: float = 0.88,
                 wake_fatigue: float = 0.25,
                 nrem_bout_ticks: int = 6,
                 rem_bout_ticks: int = 4,
                 adapters: Optional[list[InputAdapter]] = None,
                 classifier: Optional[SalienceClassifier] = None,
                 coalesce_window_seconds: float = 60.0,
                 coalesce_max_items: int = 8,
                 coalesce_force_salience: float = 0.85,
                 log: Callable[[str], None] = print,
                 seed: Optional[int] = None):
        self.cfg = cfg
        self.brain = brain
        self.tick_seconds = tick_seconds
        self.world_cycle_minutes = world_cycle_minutes
        self.idle_rate = idle_rate
        self.drowsy_fatigue = drowsy_fatigue
        self.sleep_fatigue = sleep_fatigue
        self.wake_fatigue = wake_fatigue
        self.nrem_bout_ticks = nrem_bout_ticks
        self.rem_bout_ticks = rem_bout_ticks
        self.log = log
        # Persistent affect — separate from the per-task affect Brain.run
        # builds. This is the daemon's CONTINUOUS affective state that
        # survives across tasks and sleep bouts.
        self.affect: AffectState = brain._build_initial_affect()
        # World — same instance lives across the daemon's life
        self.world = brain.world
        self.state: str = WAKE
        self.tick_n: int = 0
        self.input_q: "queue.Queue[str]" = queue.Queue()
        self.stats = DaemonStats()
        # Sleep bookkeeping
        self._sleep_bouts_remaining = 0  # ticks left in current bout
        self._sleep_bout_kind: Optional[str] = None  # 'nrem' | 'rem'
        self._sleep_cycle_index = 0       # counts NREM/REM alternations
        # Sleep agents
        self.forgetter = Forgetter()
        self.skill_pruner = SkillPruner()
        self.mood_regulator = MoodRegulator()
        self.dreamer = Dreamer(seed=seed)
        self.scheduler = Scheduler()
        fm_cfg = (cfg.raw.get("forward_model") if isinstance(cfg.raw, dict) else {}) or {}
        self.forward_model_trainer = ForwardModelTrainer(
            backend=str(fm_cfg.get("backend", "auto")),
            hidden=int(fm_cfg.get("hidden", 256)),
            depth=int(fm_cfg.get("depth", 2)),
            epochs=int(fm_cfg.get("epochs", 200)),
            lr=float(fm_cfg.get("lr", 1e-3)),
            min_rows=int(fm_cfg.get("min_rows", 40)),
        )
        self.screen_sequence_trainer = ScreenSequenceTrainer(
            backend=str(fm_cfg.get("backend", "auto")),
            hidden=int(fm_cfg.get("hidden", 256)),
            depth=int(fm_cfg.get("depth", 2)),
            epochs=int(fm_cfg.get("epochs", 200)),
            min_pairs=int((cfg.capture or {}).get("min_train_pairs", 40)),
        )
        # Teacher-student: the executive model curates the student's
        # screen-reading skill during sleep. Only when capture is on.
        self.vision_teacher = (
            VisionTeacher(max_new_rules=int((cfg.capture or {}).get("vision_teacher_rules", 2)))
            if (cfg.capture or {}).get("enabled") and (cfg.capture or {}).get("vision_teacher", True)
            else None
        )

        # Screen observation (privacy-first) — only when capture is enabled in
        # config AND the brain is embodied (needs eyes). The observer enforces
        # local-only and drops pixels; the purger enforces retention.
        cap_cfg = cfg.capture or {}
        frames_dir = str(Path(cfg.sandbox_dir) / "frames")
        # Presence: when the user is away (idle past this threshold — screensaver
        # / lock / stepped-away), the brain sleeps and capture pauses. None until
        # first sampled. presence_sleep makes presence the dominant wake/sleep
        # trigger (over the affect/clock model) while capture is enabled.
        self.away_threshold_s = float(cap_cfg.get("away_threshold_seconds", 300))
        # Presence governs sleep only when capture/observation is enabled;
        # otherwise the affect/clock model drives wake/sleep as before.
        self.presence_sleep = bool(cap_cfg.get("enabled")) and \
            bool(cap_cfg.get("presence_sleep", True))
        self._user_present: Optional[bool] = None
        self.observer = None
        if cap_cfg.get("enabled") and getattr(brain, "embodiment", None) is not None:
            self.observer = ScreenObserver(
                cfg, brain.llm, brain.memory, brain.embodiment,
                interval_seconds=float(cap_cfg.get("interval_seconds", 60)),
                min_interval_seconds=float(cap_cfg.get("min_interval_seconds", 8)),
                activity_window_seconds=float(cap_cfg.get("activity_window_seconds", 8)),
                exclude_apps=cap_cfg.get("exclude_apps"),
                vision_model=str(cap_cfg.get("vision_model", "")),
                vision_model_strong=str(cap_cfg.get("vision_model_strong", "")),
                change_detect=bool(cap_cfg.get("change_detect", True)),
                visual_embedder=getattr(brain, "visual_embedder", None),
            )
            if not self.observer.ok:
                self.log(f"  ⚠ screen capture refused: {self.observer.reason}")
        self.screen_purger = ScreenPurger(
            max_age_days=float(cap_cfg.get("max_age_days", 30)),
            max_rows=int(cap_cfg.get("max_rows", 20000)),
            capture_dir=frames_dir,
            max_dir_mb=float(cap_cfg.get("max_dir_mb", 200)),
        )

        # Input adapters + classifier (load-shedding for continuous streams)
        self.adapters: list[InputAdapter] = list(adapters or [])
        if classifier is None:
            classifier = SalienceClassifier(
                rules=ClassifierRules(),
                affect=self.affect,
                memory=brain.memory,
                embedding_backend=brain.memory.backend,
            )
        else:
            # Wire the persistent affect handle so classifier sees mood
            classifier.affect = self.affect
            if classifier.memory is None:
                classifier.memory = brain.memory
        self.classifier = classifier

        # Coalesced wake batching
        self.coalesce_window_seconds = coalesce_window_seconds
        self.coalesce_max_items = coalesce_max_items
        self.coalesce_force_salience = coalesce_force_salience
        self._direct_buffer: list[StreamItem] = []
        self._ambient_buffer: list[StreamItem] = []
        self._buffer_opened_at: Optional[float] = None
        # Bring adapters online
        for ad in self.adapters:
            try:
                ad.start()
            except Exception as e:
                self.log(f"[daemon] adapter {ad.name} failed to start: {e}")

    # ── public API ──────────────────────────────────────────────────────────
    def enqueue(self, task: str) -> None:
        """Add a task to the input queue. Bypasses the classifier and goes
        straight into the direct buffer at maximum salience. Use for
        programmatic injection (tests, glue scripts)."""
        if task and task.strip():
            self._direct_buffer.append(StreamItem(
                source="enqueue", kind="task", content=task.strip(),
                channel="direct", sender="programmatic",
                salience=1.0,
            ))

    def add_adapter(self, adapter: InputAdapter) -> None:
        try:
            adapter.start()
        except Exception as e:
            self.log(f"[daemon] adapter {adapter.name} failed to start: {e}")
            return
        self.adapters.append(adapter)

    def run_forever(self, max_ticks: Optional[int] = None) -> DaemonStats:
        """Main loop. `max_ticks` is for tests / bounded runs."""
        try:
            while True:
                if max_ticks is not None and self.tick_n >= max_ticks:
                    self.log(f"[daemon] max_ticks={max_ticks} reached; exiting")
                    return self.stats
                self.tick()
                if self.tick_seconds > 0:
                    time.sleep(self.tick_seconds)
        except KeyboardInterrupt:
            self.log("\n[daemon] interrupted; saving and exiting")
            return self.stats
        finally:
            for ad in self.adapters:
                try:
                    ad.close()
                except Exception:
                    pass

    # ── one tick ────────────────────────────────────────────────────────────
    def tick(self) -> None:
        self.tick_n += 1

        # ── presence + activity: one idle read drives both ─────────────────
        idle = None
        if self.observer is not None or self.presence_sleep:
            idle = idle_seconds()
            if idle is not None:
                self._user_present = idle < self.away_threshold_s

        # ── screen observation — only while the user is present (when away
        # there's nothing but a lock screen to see, and that's sleep time).
        # `idle` feeds the activity-settle trigger.
        if self.observer is not None and self._user_present is not False:
            try:
                res = self.observer.maybe_capture(idle=idle)
                if res and res.get("captured"):
                    self.log(
                        f"  👁 capture [{res.get('trigger')}] "
                        f"{res.get('model_tier')} model "
                        f"({res.get('model') or 'none'}) → {res.get('content')}")
            except Exception as e:
                self.log(f"[daemon] observer error: {e}")

        # ── publish live status for the UI ─────────────────────────────────
        self._write_status()

        # ── poll input adapters → classify → route ─────────────────────────
        # This is the streaming load-shedder. Every item goes through the
        # deterministic classifier; below-threshold items are dropped (or
        # stored as low-salience episodic for morning recall) so they never
        # cost the LLM. Above-threshold items become broadcasts or batch
        # into the direct buffer for coalesced wake.
        new_items: list[StreamItem] = []
        for ad in self.adapters:
            try:
                new_items.extend(ad.poll())
            except Exception as e:
                self.log(f"[daemon] adapter {ad.name} poll error: {e}")
        if new_items:
            self.classifier.classify_batch(new_items)
            for it in new_items:
                bucket = self.classifier.route(it)
                if bucket == "drop":
                    # Store as a low-salience episodic so morning recall sees it
                    try:
                        self.brain.memory.store(
                            task="ambient_stream", kind=it.kind,
                            content=it.short(160),
                            salience=max(0.05, it.salience),
                            mem_type="episodic",
                            tags=[f"src:{it.source}", "below_threshold"],
                        )
                    except Exception:
                        pass
                    continue
                if bucket == "direct":
                    self._direct_buffer.append(it)
                    if self._buffer_opened_at is None:
                        self._buffer_opened_at = time.time()
                else:
                    self._ambient_buffer.append(it)

        # Advance the world (ambient stimuli, time-of-day, drives drift).
        # We do this in any state, but the effects on affect are smaller
        # during sleep (most stimuli are filtered out).
        stims = []
        if self.world is not None:
            stims = self.world.tick()
            for s in stims:
                if self.state == WAKE:
                    if s.affect_delta:
                        self.affect.update("world", self.tick_n,
                                            s.affect_delta, smoothing=0.85)
                elif self.state == DROWSY:
                    # half-magnitude effects
                    if s.affect_delta:
                        half = {k: v * 0.4 for k, v in s.affect_delta.items()}
                        self.affect.update("world", self.tick_n, half, smoothing=0.9)
                # NREM/REM: world ticks but doesn't update affect — that's
                # what sleep IS, sensory gating

        # Time-triggered prospective items always run
        fired = self.scheduler.fire_due(self.brain.memory)
        if fired:
            self.stats.prospective_fired += len(fired)
            for f in fired:
                self.log(f"  ⏰ prospective due: {f['content'][:80]}")
            # A due intention rouses you from sleep
            if self.state in (NREM, REM, DROWSY):
                self.log("  ↑ scheduler rousing brain (time trigger)")
                self._enter_state(WAKE)
                # Inject into input queue as a task to think about
                for f in fired:
                    self.input_q.put(f"REMINDER: {f['content']}")

        # External input always wakes the brain
        if (self._direct_buffer or not self.input_q.empty()) and self.state != WAKE:
            self.log(f"  ↑ external input arrived → WAKE")
            self._enter_state(WAKE)

        # Drain any high-salience ambient items into broadcasts on the
        # workspace at the start of any state — but only the workspace of
        # the brain's current task (or future task). For now we just
        # accumulate; they get folded into the next task summary by
        # _maybe_process_coalesced_batch.

        # Dispatch by state
        if self.state == WAKE:
            self._tick_wake()
        elif self.state == DROWSY:
            self._tick_drowsy()
        elif self.state == NREM:
            self._tick_nrem()
        elif self.state == REM:
            self._tick_rem()

        # Slow drift between ticks (hunger/fatigue accumulate when awake,
        # only hunger when asleep — handled in MoodRegulator)
        if self.state == WAKE or self.state == DROWSY:
            self.affect.decay_toward_baseline()

        # State transition based on affect + clock
        self._maybe_transition()

    # ── status publishing for the UI ─────────────────────────────────────────
    def _write_status(self) -> None:
        try:
            a = self.affect
            payload = {
                "ts": time.time(),
                "state": self.state,
                "tick": self.tick_n,
                "sleep_bout": self._sleep_bout_kind,
                "affect": {
                    "mood_label": a.mood_label,
                    "valence": round(a.valence, 3),
                    "arousal": round(a.arousal, 3),
                    "stress": round(a.stress, 3),
                    "fatigue": round(a.fatigue, 3),
                    "boredom": round(a.boredom, 3),
                },
                "stats": self.stats.__dict__,
                "capture": self.observer.stats() if self.observer else None,
                "user_present": self._user_present,
            }
            write_status(self.cfg, payload)
        except Exception:
            pass

    # ── per-state ticks ─────────────────────────────────────────────────────
    def _tick_wake(self) -> None:
        # 1. Legacy input queue (programmatic enqueue from before)
        if not self.input_q.empty():
            task = self.input_q.get()
            self.log(f"\n[wake] processing: {task[:80]}")
            self._run_task_with_persistent_affect(task)
            self.stats.tasks_processed += 1
            return

        # 2. Coalesced-wake: drain the direct/ambient buffers as ONE task
        if self._should_flush_buffers():
            self._process_coalesced_batch()
            return

        # Idle: maybe think a spontaneous thought (rate-limited).
        # Curiosity + low stress favor it; boredom resists it (paradoxically:
        # bored brains drift in DMN rather than 'deliberately think').
        if self.idle_rate <= 0:
            return
        # Effective rate scales with curiosity, dampened by stress
        eff_rate = self.idle_rate * (0.4 + 0.8 * self.affect.curiosity) \
                    * (1.0 - 0.6 * self.affect.stress)
        import random
        if random.random() < eff_rate:
            self._spontaneous_thought()

    def _tick_drowsy(self) -> None:
        # very few thoughts; affect drift dominates
        pass

    def _tick_nrem(self) -> None:
        if self._sleep_bouts_remaining <= 0:
            return
        # On the first tick of an NREM bout, run the agents
        ticks_done = self.nrem_bout_ticks - self._sleep_bouts_remaining
        if ticks_done == 0:
            self.log(f"\n[nrem] bout starts (cycle {self._sleep_cycle_index + 1})")
            # mood regulator: fatigue recovery, mood drift
            self.mood_regulator.run(self.affect)
            # forgetter: prune low-salience old episodes
            stats = self.forgetter.run(self.brain.memory)
            self.stats.episodes_pruned += stats.get("pruned", 0)
            if stats.get("pruned"):
                self.log(f"  ✂ forgetter pruned {stats['pruned']} "
                         f"of {stats['considered']} considered")
            # skill pruner: decay unused, delete lost causes
            sp = self.skill_pruner.run(self.brain.skills)
            self.stats.skills_decayed += sp.get("decayed", 0)
            self.stats.skills_deleted += sp.get("deleted", 0)
            if sp.get("decayed") or sp.get("deleted"):
                self.log(f"  ◇ skills decayed={sp['decayed']} deleted={sp['deleted']}")
            # consolidator: distill semantic facts from clusters
            try:
                cs = consolidator.consolidate(
                    self.brain.memory, self.brain.llm,
                    model=self.cfg.models.get("reflex"),
                    n_episodes=int(self.cfg.memory.get("sleep_consolidate_window", 80)),
                    min_cluster_size=int(self.cfg.memory.get("consolidate_min_cluster", 3)),
                    max_new_facts=int(self.cfg.memory.get("sleep_consolidate_max_facts", 4)),
                    log=lambda m: self.log(f"  {m}"),
                )
                self.stats.facts_consolidated += cs.get("facts_written", 0)
            except Exception as e:
                self.log(f"  ⚠ consolidator failed: {type(e).__name__}: {e}")
            # forward-model trainer: learn the world model (gradients) on the
            # accumulated (state, action, outcome) triples; refresh the live
            # cerebellum's model. No-ops on non-dense backends / numpy absent.
            try:
                fm = self.forward_model_trainer.run(
                    self.brain.memory, self.brain.world_model,
                    cerebellum=getattr(self.brain, "cerebellum", None),
                    log=lambda m: self.log(f"  {m}"),
                )
                if fm.get("trained"):
                    self.stats.forward_model_trains += 1
            except Exception as e:
                self.log(f"  ⚠ forward_model trainer failed: {type(e).__name__}: {e}")
            # screen-sequence trainer: learn the dynamics of the user's day
            # (next-screen prediction) from the observation stream; refresh the
            # live occipital model.
            try:
                sm = self.screen_sequence_trainer.run(
                    self.brain.memory,
                    occipital=getattr(self.brain, "occipital", None),
                    log=lambda m: self.log(f"  {m}"))
                if sm.get("trained"):
                    self.stats.screen_model_trains += 1
            except Exception as e:
                self.log(f"  ⚠ screen_sequence trainer failed: {type(e).__name__}: {e}")
            # vision teacher: strong model curates the student's screen-reading
            # skill from its recent descriptions.
            if self.vision_teacher is not None:
                try:
                    vt = self.vision_teacher.run(
                        self.brain.memory, self.brain.llm,
                        self.cfg.models.get("executive"),
                        db_path=self.cfg.db_path,
                        log=lambda m: self.log(f"  {m}"))
                    self.stats.vision_rules_learned += vt.get("rules_added", 0)
                except Exception as e:
                    self.log(f"  ⚠ vision_teacher failed: {type(e).__name__}: {e}")
            # screen purger: retention on the observation stream + orphan/disk cleanup
            try:
                pr = self.screen_purger.run(self.brain.memory,
                                            log=lambda m: self.log(f"  {m}"))
                self.stats.observations_pruned += (
                    pr.get("pruned_age", 0) + pr.get("pruned_cap", 0))
            except Exception as e:
                self.log(f"  ⚠ screen_purger failed: {type(e).__name__}: {e}")
            self.stats.sleep_bouts["nrem"] += 1
        self._sleep_bouts_remaining -= 1

    def _tick_rem(self) -> None:
        if self._sleep_bouts_remaining <= 0:
            return
        ticks_done = self.rem_bout_ticks - self._sleep_bouts_remaining
        if ticks_done == 0:
            self.log(f"\n[rem] bout starts (cycle {self._sleep_cycle_index + 1})")
            try:
                stats = self.dreamer.run(
                    self.brain.memory, self.brain.llm,
                    model=self.cfg.models.get("reflex"),
                    world_model=self.brain.world_model,
                    log=lambda m: self.log(f"  {m}"),
                )
                self.stats.dreams_written += stats.get("written", 0)
                self.stats.sleep_bouts["rem"] += 1
            except Exception as e:
                self.log(f"  ⚠ dreamer failed: {type(e).__name__}: {e}")
        self._sleep_bouts_remaining -= 1

    # ── transitions ─────────────────────────────────────────────────────────
    def _maybe_transition(self) -> None:
        a = self.affect
        # Presence-driven sleep (dominant when enabled): the brain sleeps when
        # the user steps away and wakes when they return — grounding wake/sleep
        # in reality instead of the synthetic fatigue clock.
        if self.presence_sleep and self._user_present is not None:
            if not self._user_present:
                if self.state == WAKE:
                    self._enter_state(DROWSY)
                    return
                if self.state == DROWSY:
                    self._start_sleep_cycle()
                    return
                # already NREM/REM → let bouts alternate below
            else:  # user present → ensure awake
                if self.state in (DROWSY, NREM, REM):
                    self._enter_state(WAKE)
                    self._sleep_bouts_remaining = 0
                    return
                # WAKE: don't force fatigue-sleep while the user is here
                if self.state in (NREM, REM) and self._sleep_bouts_remaining <= 0:
                    self._advance_sleep_cycle()
                return
        # WAKE → DROWSY
        if self.state == WAKE and a.fatigue >= self.drowsy_fatigue:
            self._enter_state(DROWSY)
            return
        # DROWSY → NREM
        if self.state == DROWSY and a.fatigue >= self.sleep_fatigue:
            self._start_sleep_cycle()
            return
        # NREM/REM: when bout ends, alternate
        if self.state in (NREM, REM) and self._sleep_bouts_remaining <= 0:
            self._advance_sleep_cycle()
            return

    def _enter_state(self, new_state: str) -> None:
        if new_state == self.state:
            return
        self.stats.state_changes.append((time.time(), self.state, new_state))
        self.log(f"\n[state] {self.state} → {new_state} "
                 f"(fatigue={self.affect.fatigue:.2f} "
                 f"hour={self.world.hour if self.world else '-'})")
        self.state = new_state

    def _start_sleep_cycle(self) -> None:
        self._sleep_cycle_index = 0
        self._enter_state(NREM)
        self._sleep_bouts_remaining = self.nrem_bout_ticks

    def _advance_sleep_cycle(self) -> None:
        """Move to the next sleep bout. Real biology: NREM,NREM,REM,NREM,
        REM,REM... with REM growing late. Simplified: 2 NREM bouts then 1 REM
        per cycle for the first 2 cycles, then 1 NREM 2 REM."""
        self._sleep_cycle_index += 1
        # Wake up if fatigue is sufficiently recovered
        if self.affect.fatigue <= self.wake_fatigue:
            self._enter_state(WAKE)
            self._sleep_bouts_remaining = 0
            self._sleep_bout_kind = None
            return
        # Bias REM later in the night
        late_night = self._sleep_cycle_index >= 2
        # Simple pattern: alternate NREM/REM; longer REM late
        if self.state == NREM:
            self._enter_state(REM)
            self._sleep_bouts_remaining = (self.rem_bout_ticks
                                           + (2 if late_night else 0))
        else:
            self._enter_state(NREM)
            self._sleep_bouts_remaining = (self.nrem_bout_ticks
                                           - (2 if late_night else 0))

    # ── coalesced-wake batching ─────────────────────────────────────────────
    def _should_flush_buffers(self) -> bool:
        """Trigger a coalesced wake when: buffer is non-empty AND
        (max items reached, OR window elapsed, OR any item has very high
        salience that warrants immediate attention)."""
        if not self._direct_buffer and not self._ambient_buffer:
            return False
        # Force on any very-high-salience item
        if any(i.salience >= self.coalesce_force_salience
                for i in self._direct_buffer):
            return True
        # Max items reached
        if len(self._direct_buffer) >= self.coalesce_max_items:
            return True
        # Window elapsed
        if self._buffer_opened_at is None:
            return False
        if time.time() - self._buffer_opened_at >= self.coalesce_window_seconds:
            return True
        return False

    def _process_coalesced_batch(self) -> None:
        """Compose a single task text from the buffered items, run Brain.run
        once, then post any remaining ambient items as low-salience
        episodic for later recall."""
        direct = sorted(self._direct_buffer,
                         key=lambda i: i.salience, reverse=True)
        ambient = sorted(self._ambient_buffer,
                          key=lambda i: i.salience, reverse=True)
        if not direct and not ambient:
            return
        self._direct_buffer = []
        self._ambient_buffer = []
        self._buffer_opened_at = None

        # Build the task text. Direct items get full content; ambient items
        # get short glosses. The brain treats this as one cognitive cycle.
        bits: list[str] = []
        if direct:
            bits.append(f"You have {len(direct)} new direct items:")
            for it in direct[: self.coalesce_max_items]:
                tag = f"[{it.source}/{it.kind} s={it.salience:.2f}]"
                sender = f" {it.sender}:" if it.sender else ""
                bits.append(f"  {tag}{sender} {it.short(220)}")
        if ambient:
            top = ambient[:5]
            bits.append(f"\nAmbient (top {len(top)} of {len(ambient)}):")
            for it in top:
                bits.append(f"  [{it.source} s={it.salience:.2f}] {it.short(140)}")
        task_text = "\n".join(bits)

        self.log(f"\n[wake] coalesced batch: {len(direct)} direct + "
                 f"{len(ambient)} ambient items")
        self._run_task_with_persistent_affect(task_text)
        self.stats.tasks_processed += 1

        # Bury the rest of ambient (beyond the top 5) as low-sal episodic so
        # they're still recallable later but don't keep crowding the prompt.
        for it in ambient[5:]:
            try:
                self.brain.memory.store(
                    task="ambient_stream", kind=it.kind,
                    content=it.short(160),
                    salience=max(0.05, it.salience * 0.6),
                    mem_type="episodic",
                    tags=[f"src:{it.source}", "ambient_buried"],
                )
            except Exception:
                pass

    # ── spontaneous wake-thought (idle wandering) ──────────────────────────
    def _spontaneous_thought(self) -> None:
        """One-shot small cycle: bored mind notices something. Not a full
        Brain.run — just a single prefrontal next_thought call with a
        synthetic 'idle' task. Cheap and rare."""
        self.stats.spontaneous_thoughts += 1
        # Use the brain's affect-aware machinery via Brain.run with a tiny
        # cap so it terminates in 1-2 thoughts. The persistent affect is
        # passed through.
        self._run_task_with_persistent_affect(
            "Idle: just notice what's on your mind right now. Brief.",
            max_units_override=2,
            quiet=True,
        )

    # ── task processing with persistent affect ─────────────────────────────
    def _run_task_with_persistent_affect(self, task: str,
                                          max_units_override: Optional[int] = None,
                                          quiet: bool = False) -> None:
        """Run one Brain.run while preserving the daemon's persistent
        AffectState across tasks. The brain ordinarily builds a fresh
        AffectState per run; we temporarily monkey-patch _build_initial_affect
        to return our persistent one (mutated in place by the run)."""
        original = self.brain._build_initial_affect
        original_loop = dict(self.brain.cfg.loop)
        if max_units_override is not None:
            self.brain.cfg.loop = {**original_loop, "max_cycles": max_units_override}
        original_log = self.brain.log
        if quiet:
            self.brain.log = lambda _m: None
        # Pass the live affect by reference
        affect_ref = self.affect
        self.brain._build_initial_affect = lambda: affect_ref  # type: ignore
        try:
            answer = self.brain.run(task)
            if not quiet:
                self.log("\n" + "═" * 40 + "\nBRAIN:\n" + "═" * 40)
                self.log(answer)
        finally:
            self.brain._build_initial_affect = original
            self.brain.cfg.loop = original_loop
            self.brain.log = original_log


# ── CLI ─────────────────────────────────────────────────────────────────────
def _maybe_read_stdin_line(timeout: float = 0.05) -> Optional[str]:
    """Non-blocking stdin read. Returns a stripped line if one's ready, else None."""
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip() or None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the brain as a long-lived sleep/wake daemon.")
    ap.add_argument("--tick-seconds", type=float, default=1.0,
                    help="wall-clock seconds per tick (default 1.0)")
    ap.add_argument("--world-cycle-minutes", type=float, default=6.0,
                    help="simulated minutes the world advances per tick")
    ap.add_argument("--idle-rate", type=float, default=0.10,
                    help="P(spontaneous thought) per idle WAKE tick (0=off)")
    ap.add_argument("--max-ticks", type=int, default=None,
                    help="exit after N ticks (for tests / bounded runs)")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--initial-task", default=None,
                    help="enqueue a task before the loop starts")
    ap.add_argument("--no-stdin", action="store_true",
                    help="don't poll stdin for direct task input")
    ap.add_argument("--webhook-port", type=int, default=None,
                    help="enable HTTP webhook adapter on this port")
    ap.add_argument("--tail-file", action="append", default=[],
                    help="tail this file as an ambient stream (repeatable)")
    ap.add_argument("--coalesce-window", type=float, default=60.0,
                    help="seconds to accumulate items before processing")
    ap.add_argument("--coalesce-max", type=int, default=8,
                    help="max items to batch before forcing a wake cycle")
    args = ap.parse_args()

    cfg = load_config()
    if args.persona:
        cfg.raw["persona_path"] = args.persona
    if args.scenario:
        cfg.raw["scenario"] = args.scenario

    brain = Brain(cfg, confirm=lambda _m: False,
                  log=lambda m: print(m, flush=True),
                  humanize=True, seed=args.seed)

    # Build input adapters per CLI flags. stdin is on by default.
    adapters: list = []
    if not args.no_stdin:
        adapters.append(StdinAdapter())
    if args.webhook_port:
        from brain.inputs import WebhookAdapter
        adapters.append(WebhookAdapter(port=args.webhook_port))
    if args.tail_file:
        from brain.inputs import FileTailAdapter
        adapters.append(FileTailAdapter(paths=args.tail_file))

    daemon = BrainDaemon(
        cfg, brain,
        tick_seconds=args.tick_seconds,
        world_cycle_minutes=args.world_cycle_minutes,
        idle_rate=args.idle_rate,
        adapters=adapters,
        coalesce_window_seconds=args.coalesce_window,
        coalesce_max_items=args.coalesce_max,
        log=lambda m: print(m, flush=True),
        seed=args.seed,
    )
    if args.initial_task:
        daemon.enqueue(args.initial_task)

    adapter_names = ", ".join(a.name for a in adapters) or "none"
    print(f"[daemon] starting in state={daemon.state} "
          f"(tick={args.tick_seconds}s, idle_rate={args.idle_rate}, "
          f"adapters={adapter_names}, "
          f"coalesce={args.coalesce_window:.0f}s/{args.coalesce_max} items)\n"
          "Type a task and hit Enter to enqueue it (if stdin adapter is on). "
          "Ctrl-C to exit.\n",
          flush=True)

    try:
        stats = daemon.run_forever(max_ticks=args.max_ticks)
    finally:
        brain.close()
    print(f"\n[daemon] final stats: {stats.__dict__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
