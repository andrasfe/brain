"""The two conditions being compared.

A) brain   — the full multi-agent Global-Workspace brain, qwen on ALL modules,
              effectors disabled so it must PLAN by reasoning (no code execution).
B) qwen0   — the raw qwen model, single zero-shot call (the paper's baseline).

Both return (raw_text, n_llm_calls) so we can compare accuracy AND cost.
"""
from __future__ import annotations

from dataclasses import dataclass

from brain.config import load_config, Config
from brain.llm import LLM
from brain.orchestrator import Brain


QWEN = "qwen/qwen3.7-plus"


def _eval_config() -> Config:
    """Config override: qwen everywhere, no effectors, short planning loop."""
    cfg = load_config()
    cfg.models = {"reflex": QWEN, "executive": QWEN}
    cfg.regions = {r: "reflex" for r in
                   ["sensory_cortex", "amygdala", "basal_ganglia",
                    "hippocampus", "prefrontal", "broca"]}
    cfg.effectors = {
        "filesystem": {"enabled": False},
        "shell": {"enabled": False},
        "web": {"enabled": False},
    }
    cfg.loop = {"max_cycles": 3, "attention_decay": 0.85}
    return cfg


class _CountingLLM:
    """Wraps an LLM to count calls."""
    def __init__(self, llm: LLM):
        self._llm = llm
        self.calls = 0

    def chat(self, *a, **k):
        self.calls += 1
        return self._llm.chat(*a, **k)

    def chat_json(self, *a, **k):
        self.calls += 1
        return self._llm.chat_json(*a, **k)

    def close(self):
        self._llm.close()


# ── Condition A: the multi-agent brain ────────────────────────────────────────
def solve_with_brain(task_text: str) -> tuple[str, int]:
    cfg = _eval_config()
    brain = Brain(cfg, confirm=lambda _m: False, log=lambda _m: None)
    counter = _CountingLLM(brain.llm)
    brain.llm = counter
    # rebuild region LLM refs to point at the counter
    for region in (brain.sensory, brain.amygdala, brain.hippocampus,
                   brain.prefrontal, brain.basal_ganglia, brain.broca):
        region.llm = counter
    try:
        answer = brain.run(task_text)
    except Exception as e:  # noqa: BLE001
        answer = f"(brain error: {type(e).__name__}: {e})"
    finally:
        n = counter.calls
        brain.close()
    return answer, n


# ── Condition B: qwen zero-shot ───────────────────────────────────────────────
_BASELINE_SYSTEM = (
    "You are an expert puzzle solver. Follow the rules exactly and answer in the "
    "requested format."
)


def solve_with_qwen(prompt: str) -> tuple[str, int]:
    cfg = load_config()
    llm = LLM(cfg)
    try:
        text = llm.chat(QWEN, _BASELINE_SYSTEM, prompt,
                        temperature=0.0, max_tokens=1024)
    except Exception as e:  # noqa: BLE001
        text = f"(qwen error: {type(e).__name__}: {e})"
    finally:
        llm.close()
    return text, 1
