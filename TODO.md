# TODO — next steps

Ordered roughly by impact. Items 1–2 are the ones most likely to reproduce the
MAP paper's effect (modular multi-agent > raw LLM).

## 1. Add a Predictor + Monitor (close the gap to true MAP) — highest impact

The brain currently has no move-level state simulation or validity gate, so it
emits invalid Hanoi moves just like the raw model. The paper's 0%-invalid result
comes specifically from these two modules.

- **Predictor region** (`brain/regions/predictor.py`): given the current state and
  a proposed action, predict (or, where rules are known, *compute*) the resulting
  state. For Hanoi/graph the transition is deterministic, so the Predictor can be
  backed by a small deterministic simulator rather than an LLM call — faster and
  exact.
- **Monitor region** (`brain/regions/monitor.py`): gate each proposed action
  against task rules; reject rule-violating actions before they are committed and
  feed the reason back to the prefrontal/actor for a retry. This is the piece that
  drives invalid-move rate toward zero.
- Wire both into the cognitive cycle between `plan` (prefrontal) and `act`:
  `plan → predict → monitor(go/no-go) → act`. The basal ganglia stays as the
  higher-level go/no-go; the Monitor is the rule-level filter.
- Expectation: on Hanoi this should sharply cut % invalid moves and raise %
  solved for the brain condition specifically — the paper's core finding.

## 2. Run a real sample size (get statistical signal)

n=1 proved the pipeline; it says nothing about the hypothesis. Reproduce the
paper's comparison with enough instances.

- Bump `eval/run_eval.py` to `--n 10` (or more) per task; consider 3-disk **and**
  4-disk (OOD) Hanoi as the paper does.
- Because `qwen/qwen3.7-plus` is ~4–5 min/call, either:
  - run it **overnight/background** via `run_eval.command`, or
  - switch the eval to a **faster non-reasoning qwen** (e.g. a qwen-2.5-instruct
    variant) for both conditions — still qwen-vs-qwen-in-brain, minutes not hours.
- Report mean ± spread, and the paper's reference numbers alongside (already in
  `results.md`).

## 3. Faithful graph task

Our graph is a faithful *reconstruction* of community structure, not the paper's
exact adjacency.

- Pull the real CogEval graph (arXiv:2309.15129) / the MAP repo
  (`github.com/MAPLLM/MAPICLR2025sub`) and replace `eval/graph.py`'s adjacency.
- Add the other three graph conditions from the paper: **valuepath**, **detour**,
  and **reward revaluation** (not just steppath shortest-path).

## 4. Add the remaining MAP modules

For a closer architectural match to MAP beyond Predictor/Monitor:

- **TaskDecomposer**: emit a single intermediate subgoal (the paper's goal-
  recursion strategy for Hanoi notably lifts even the zero-shot baseline).
- **Evaluator**: estimate steps-to-goal (an admissible heuristic) to rank
  candidate actions — enables the paper's B=2 / depth=2 search.
- **Orchestrator**: explicit subgoal-achieved check that decides when to emit the
  plan. (Our prefrontal `finish` currently plays this role implicitly.)

## 5. Stronger output contracts / parsing robustness

- Tighten the planning-task answer format so parsing never silently drops a move
  (e.g., require a fenced block; reject and re-ask on parse failure).
- Add a "best-of-k" mode to mirror the paper's average-vs-best reporting.

## 6. Cost & latency instrumentation

- Record wall-clock per call and per region, plus token counts, into the results
  JSON. Makes the "modular is many small calls" trade-off explicit (accuracy
  gain vs. call-count/latency cost).
- Add a per-call timeout/early-abort so one slow call can't stall a whole run.

## 7. Productization niceties (lower priority)

- "Default mode network" idle loop (the always-on tick model we deferred) so the
  brain can deliberate without input.
- Set the reflex tier to qwen too if you want a single-model brain (currently
  reflex = gemini-flash-lite, executive = qwen).
- Persist and inspect the cognitive trace per run (structured, not just stdout).
- Tests: promote the offline eval checks into a `tests/` suite run by pytest.

---

### Immediate recommendation

Do **#1 (Predictor + Monitor)** then **#2 with a faster qwen**. That combination
is the smallest amount of work that should actually demonstrate the paper's
claim — the brain beating the raw model on Tower of Hanoi — rather than the
inconclusive n=1 we have now.
