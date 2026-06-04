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
from brain.orchestrator import Brain  # noqa: E402
from brain.sleep import (  # noqa: E402
    Dreamer, Forgetter, MoodRegulator, Scheduler, SkillPruner,
)
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

    # ── public API ──────────────────────────────────────────────────────────
    def enqueue(self, task: str) -> None:
        """Add a task to the input queue. Will be processed next time the
        daemon enters WAKE (and finishes current work)."""
        if task and task.strip():
            self.input_q.put(task.strip())

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

    # ── one tick ────────────────────────────────────────────────────────────
    def tick(self) -> None:
        self.tick_n += 1
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
        if not self.input_q.empty() and self.state != WAKE:
            self.log(f"  ↑ external input arrived → WAKE")
            self._enter_state(WAKE)

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

    # ── per-state ticks ─────────────────────────────────────────────────────
    def _tick_wake(self) -> None:
        if not self.input_q.empty():
            task = self.input_q.get()
            self.log(f"\n[wake] processing: {task[:80]}")
            # Hand current affect state in via the brain by patching its
            # _build_initial_affect to return the daemon's persistent affect.
            self._run_task_with_persistent_affect(task)
            self.stats.tasks_processed += 1
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
    args = ap.parse_args()

    cfg = load_config()
    if args.persona:
        cfg.raw["persona_path"] = args.persona
    if args.scenario:
        cfg.raw["scenario"] = args.scenario

    brain = Brain(cfg, confirm=lambda _m: False,
                  log=lambda m: print(m, flush=True),
                  humanize=True, seed=args.seed)
    daemon = BrainDaemon(
        cfg, brain,
        tick_seconds=args.tick_seconds,
        world_cycle_minutes=args.world_cycle_minutes,
        idle_rate=args.idle_rate,
        log=lambda m: print(m, flush=True),
        seed=args.seed,
    )
    if args.initial_task:
        daemon.enqueue(args.initial_task)

    print(f"[daemon] starting in state={daemon.state} "
          f"(tick={args.tick_seconds}s, idle_rate={args.idle_rate})\n"
          "Type a task and hit Enter to enqueue it. Ctrl-C to exit.\n",
          flush=True)

    # Wrap tick with a tiny stdin poll so the daemon stays interactive.
    original_tick = daemon.tick
    def tick_with_stdin():
        line = _maybe_read_stdin_line(timeout=min(0.05, daemon.tick_seconds))
        if line:
            daemon.enqueue(line)
        original_tick()
    daemon.tick = tick_with_stdin  # type: ignore

    try:
        stats = daemon.run_forever(max_ticks=args.max_ticks)
    finally:
        brain.close()
    print(f"\n[daemon] final stats: {stats.__dict__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
