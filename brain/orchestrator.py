"""Orchestrator — the reactive cognitive cycle that ties the regions together.

One run() per task. Pipeline per Global Workspace Theory:

  perceive (sensory cortex)
  ┌─ loop until prefrontal 'finish' or max_cycles ─────────────────────────┐
  │  recall (hippocampus)  → workspace                                      │
  │  appraise (amygdala)   → workspace, maybe interrupt                     │
  │  attention spotlight   → broadcast most salient item                    │
  │  plan (prefrontal)     → propose ONE action                            │
  │  gate (basal ganglia)  → go / no_go                                    │
  │  act (effectors)       → result → workspace                           │
  │  consolidate (hippocampus stores the episode)                          │
  └────────────────────────────────────────────────────────────────────────┘
  speak (broca) → final answer
"""
from __future__ import annotations

from typing import Callable, Optional

from .config import Config
from .effectors import Effectors
from .llm import LLM
from .memory import Memory
from .regions import (
    Amygdala,
    BasalGanglia,
    Broca,
    Hippocampus,
    Prefrontal,
    SensoryCortex,
)
from .workspace import ActionRecord, Broadcast, Workspace


class Brain:
    def __init__(self, cfg: Config, confirm: Callable[[str], bool] | None = None,
                 log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.log = log or (lambda _m: None)
        self.llm = LLM(cfg)
        self.memory = Memory(cfg.db_path)
        self.effectors = Effectors(cfg, confirm=confirm)

        self.sensory = SensoryCortex(cfg, self.llm)
        self.amygdala = Amygdala(cfg, self.llm)
        self.hippocampus = Hippocampus(cfg, self.llm, self.memory)
        self.prefrontal = Prefrontal(cfg, self.llm)
        self.basal_ganglia = BasalGanglia(cfg, self.llm)
        self.broca = Broca(cfg, self.llm)

    def close(self) -> None:
        self.llm.close()
        self.memory.close()

    # ── main entry point ──────────────────────────────────────────────────────
    def run(self, task: str) -> str:
        cfg_loop = self.cfg.loop
        ws = Workspace(task=task, decay=float(cfg_loop.get("attention_decay", 0.85)))
        max_cycles = int(cfg_loop.get("max_cycles", 12))

        self.log(f"⟶ perceiving task")
        self.sensory.step(ws)

        for i in range(1, max_cycles + 1):
            ws.cycle = i
            self.log(f"\n── cycle {i} ──────────────────────────────")

            # recall + appraise feed the spotlight
            self.hippocampus.step(ws)
            self.amygdala.step(ws)
            if ws.interrupt:
                self.log(f"  ! amygdala interrupt: {ws.interrupt}")

            spotlight = ws.tick_attention()
            if spotlight:
                self.log(f"  ◎ spotlight: [{spotlight.source}/{spotlight.kind}] "
                         f"{spotlight.content[:80]}")

            # plan
            plan = self.prefrontal.step(ws, self.effectors.available())
            proposal = plan.data
            eff = proposal.get("effector", "think")
            self.log(f"  ⊕ prefrontal proposes: {eff}")

            if eff == "finish":
                ws.final_output = proposal.get("args", {}).get("answer")
                ws.done = True
                self.log("  ✓ prefrontal signals: finish")
                break

            # gate
            gate = self.basal_ganglia.step(ws, proposal)
            if gate.data.get("decision") != "go":
                self.log(f"  ⊘ basal ganglia veto: {gate.data.get('reason', '')[:80]}")
                ws.post(Broadcast(
                    source="basal_ganglia", kind="result",
                    content=f"action vetoed: {gate.data.get('reason', '')}",
                    salience=0.65,
                ))
                continue

            # the gate may repair effector/args
            eff = gate.data.get("effector", eff)
            args = gate.data.get("args") or proposal.get("args", {})

            # act
            ok, result = self.effectors.execute(eff, args)
            self.log(f"  ▶ {eff} -> {'ok' if ok else 'ERR'}: {result[:80]}")
            ws.history.append(ActionRecord(
                cycle=i, effector=eff, args=args, result=result, ok=ok))
            ws.post(Broadcast(
                source="motor", kind="result",
                content=f"{eff} {'ok' if ok else 'FAILED'}: {result[:200]}",
                salience=0.75 if ok else 0.85,
            ))

            # consolidate to long-term memory
            self.hippocampus.consolidate(
                ws, kind="action",
                content=f"{eff}({args}) -> {'ok' if ok else 'err'}: {result[:200]}",
                salience=0.6 if ok else 0.8,
            )

        # speak
        self.log("\n⟶ synthesizing final answer (broca)")
        if ws.final_output:
            answer = ws.final_output
        else:
            answer = self.broca.step(ws)
        self.hippocampus.consolidate(ws, kind="outcome", content=answer[:500], salience=0.7)
        return answer
