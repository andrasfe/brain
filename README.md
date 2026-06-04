# brain

A multi-agent model of cognition. Each brain region is an AI agent; they
coordinate through a shared **Global Workspace** (Global Workspace Theory): many
specialists process in parallel, the most salient contribution wins the
attention "spotlight" each cycle, and the brain acts until the goal is met.

It runs as an **autonomous task agent**: give it a goal, the regions plan and
act (filesystem, shell, web) until the prefrontal cortex declares it done.

## Regions

| Region | Role | Tier |
|---|---|---|
| Sensory cortex | parses the raw task into a structured percept | reflex |
| Hippocampus | episodic memory: recall + storage (SQLite) | reflex |
| Amygdala | salience / threat / urgency; can interrupt | reflex |
| Prefrontal cortex | executive: plans, proposes the next action | executive |
| Basal ganglia | action gating: go / no-go on each proposal | reflex |
| Broca's area | synthesizes the final answer | executive |

"Reflex" and "executive" map to two OpenRouter models set in `config.yaml`.

## The cognitive cycle (reactive)

```
perceive → [ recall → appraise → spotlight → plan → gate → act → consolidate ]* → speak
```

The loop repeats until the prefrontal cortex chooses `finish` or `max_cycles`
is reached, then Broca writes the answer.

## Setup

```bash
cd ~/brain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Credentials are read from `~/specter/.env` (`OPENROUTER_API_KEY`). Adjust the
path or the model tiers in `config.yaml`.

## Run

```bash
python run.py "summarize the files in the workspace and write a report.md"
python run.py --yes "create a fibonacci.py and run it"   # auto-approve shell
python run.py --quiet "your task"                        # hide the trace
```

## Safety

Every effector is confined to `~/brain/workspace/`. Shell commands prompt for
confirmation by default (`require_confirmation` in `config.yaml`); `--yes`
auto-approves. The amygdala flags risky/destructive actions and the basal
ganglia can veto them.

## Layout

```
brain/
  workspace.py     # the global blackboard + attention spotlight
  llm.py           # OpenRouter client, tier routing
  memory.py        # SQLite hippocampus store
  region.py        # base Region agent
  regions/         # one agent per region
  effectors.py     # fs / shell / web tools (sandboxed)
  orchestrator.py  # the reactive cognitive-cycle loop
run.py             # CLI
config.yaml        # models, tiers, limits, sandbox
```

## Extending

- Add a region: subclass `Region`, give it a `name`, `system_prompt`, and a
  `step(ws)` that posts a `Broadcast`; wire it into `orchestrator.py` and add a
  tier in `config.yaml`.
- Swap memory for vectors: reimplement `Memory.retrieve/store` (e.g. chromadb);
  nothing else changes.
- Add an effector: add a method in `effectors.py` and list it in `available()`.
