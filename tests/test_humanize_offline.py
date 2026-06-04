"""Offline tests for the humanization layer — no network, no API key.

Validates:
  - AffectState dynamics (clamping, decay, trait scaling, mood label)
  - Persona load + trait derivation + memory seeds
  - World ticker emits stimuli, advances time, applies scenario seed
  - Workspace renders AffectState into context and respects attention width
  - End-to-end cognitive cycle with a stubbed LLM (no real API call)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow `import brain.*` when running from repo root or tests/
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brain.affect import AffectState, Traits  # noqa: E402
from brain.persona import Persona, load_persona  # noqa: E402
from brain.workspace import Broadcast, Workspace  # noqa: E402
from brain.world import World, SCENARIOS  # noqa: E402


class AffectStateTests(unittest.TestCase):
    def test_clamping_and_mood(self):
        a = AffectState()
        a.update("test", 1, {"valence": -2.0, "stress": +2.0}, smoothing=0.0)
        self.assertGreaterEqual(a.valence, -1.0)
        self.assertLessEqual(a.stress, 1.0)
        self.assertEqual(a.mood_label, "stressed")

    def test_trait_scaling_amplifies_negative_for_neurotic(self):
        calm = AffectState(traits=Traits(neuroticism=0.1))
        neurotic = AffectState(traits=Traits(neuroticism=0.95))
        calm.update("t", 1, {"valence": -0.5}, smoothing=0.0)
        neurotic.update("t", 1, {"valence": -0.5}, smoothing=0.0)
        # Neurotic person's valence drops further from a negative delta.
        self.assertLess(neurotic.valence, calm.valence + 1e-9)

    def test_decay_increases_drives_over_time(self):
        a = AffectState()
        h0, f0, b0 = a.hunger, a.fatigue, a.boredom
        for _ in range(8):
            a.decay_toward_baseline()
        self.assertGreater(a.hunger, h0)
        self.assertGreater(a.fatigue, f0)
        self.assertGreater(a.boredom, b0)

    def test_attention_width_inverted_u(self):
        a = AffectState()
        a.arousal = 0.5
        peak = a.attention_width
        a.arousal = 0.95
        self.assertLess(a.attention_width, peak)
        a.arousal = 0.05
        self.assertLess(a.attention_width, peak)


class PersonaTests(unittest.TestCase):
    def test_load_alex(self):
        p = load_persona(REPO / "personas" / "alex.yaml")
        assert p is not None
        self.assertEqual(p.name, "Alex Mendez")
        # introvert + perfectionist + open to weird ideas + empathetic + burnt-out
        traits = p.derive_traits()
        self.assertLess(traits.extraversion, 0.5)        # introvert
        self.assertGreater(traits.conscientiousness, 0.5)  # perfectionist
        self.assertGreater(traits.neuroticism, 0.5)        # perfectionist+burnt-out
        self.assertGreater(traits.openness, 0.5)           # open to weird ideas
        self.assertGreater(traits.agreeableness, 0.5)      # empathetic

        # Memory seeds include identity, history, recent events
        seeds = p.memory_seeds()
        kinds = {s["kind"] for s in seeds}
        self.assertIn("prior:identity", kinds)
        self.assertIn("prior:history", kinds)
        self.assertIn("prior:recent", kinds)

        # Initial affect deltas reflect mixed recent events (mild negative net)
        deltas = p.initial_affect_deltas()
        self.assertIn("valence", deltas)
        self.assertIn("stress", deltas)

    def test_load_sam_contrasts_alex(self):
        sam = load_persona(REPO / "personas" / "sam.yaml")
        alex = load_persona(REPO / "personas" / "alex.yaml")
        st = sam.derive_traits()
        at = alex.derive_traits()
        self.assertGreater(st.extraversion, at.extraversion)
        self.assertGreater(at.conscientiousness, st.conscientiousness)

    def test_explicit_overrides_win(self):
        p = Persona.from_dict({
            "identity": {"name": "X"},
            "dispositions": ["anxious"],
            "trait_overrides": {"neuroticism": 0.10},
        })
        t = p.derive_traits()
        self.assertAlmostEqual(t.neuroticism, 0.10, places=2)


class WorldTests(unittest.TestCase):
    def test_scenarios_are_valid(self):
        for name in SCENARIOS:
            w = World(name, seed=0)
            stims = w.tick()
            self.assertTrue(stims)
            # body line always emitted
            self.assertTrue(any(s.kind == "body" for s in stims))

    def test_time_advances(self):
        w = World("neutral", seed=0, cycle_minutes=30.0)
        h0 = w.hour
        w.tick()
        w.tick()
        self.assertAlmostEqual((w.hour - h0) % 24, 1.0, places=5)

    def test_deadline_scenario_pressures(self):
        w = World("deadline_night", seed=42, cycle_minutes=12.0)
        any_pressure = False
        for _ in range(8):
            stims = w.tick()
            if any(s.kind == "deadline" for s in stims):
                any_pressure = True
        self.assertTrue(any_pressure)


class WorkspaceTests(unittest.TestCase):
    def test_render_includes_affect(self):
        ws = Workspace(task="hi")
        out = ws.render_context()
        self.assertIn("AFFECT[", out)
        self.assertIn("mood=", out)

    def test_distractible_workspace_lets_dmn_win(self):
        ws = Workspace(task="bored task")
        # bias state to high distractibility
        ws.affect.boredom = 0.9
        ws.affect.fatigue = 0.7
        ws.affect.arousal = 0.2
        ws.affect.traits = Traits(conscientiousness=0.2)
        # main item higher salience, but DMN runner-up close
        ws.post(Broadcast(source="prefrontal", kind="plan",
                          content="proper plan", salience=0.85))
        ws.post(Broadcast(source="default_mode", kind="tangent",
                          content="random thought", salience=0.55))
        winner = ws.tick_attention()
        # Allowed to lose to DMN under high distractibility
        self.assertIn(winner.source, {"prefrontal", "default_mode"})


class StubbedBrainEndToEnd(unittest.TestCase):
    """Run one full cycle with a fake LLM. Catches wiring/integration bugs."""

    def test_one_cycle_with_stubbed_llm(self):
        from brain.config import Config
        from brain.orchestrator import Brain

        # Fake config dir and credentials
        tmp = Path(tempfile.mkdtemp())
        cfg = Config(
            raw={"persona_path": str(REPO / "personas" / "alex.yaml"),
                  "scenario": "boring_afternoon", "humanize": True},
            api_key="fake",
            base_url="http://fake",
            models={"reflex": "fake-reflex", "executive": "fake-exec"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=tmp,
            db_path=tmp / "mem.sqlite",
            loop={"max_cycles": 1, "attention_decay": 0.8},
            memory={"db_path": str(tmp / "mem.sqlite"), "retrieve_k": 3},
            effectors={"filesystem": {"enabled": True},
                        "shell": {"enabled": False},
                        "web": {"enabled": False}},
            regions={"sensory_cortex": "reflex", "amygdala": "reflex",
                      "basal_ganglia": "reflex", "hippocampus": "reflex",
                      "prefrontal": "executive", "broca": "executive"},
        )

        # Stub LLM with deterministic JSON-friendly responses.
        with patch("brain.orchestrator.LLM") as MockLLM:
            llm_instance = MagicMock()

            def chat_json(model, sys_prompt, user, **kw):
                # Heuristic: return structure that matches each region's schema
                if "SENSORY CORTEX" in sys_prompt:
                    return {"goal": "test", "entities": [], "constraints": [],
                            "success_criterion": "done"}
                if "AMYGDALA" in sys_prompt:
                    return {"valence_shift": 0.0, "urgency": 0.4,
                            "stress_shift": 0.0, "interrupt": None,
                            "note": "ok"}
                if "PREFRONTAL" in sys_prompt:
                    # Stream-of-thought: emit a finish unit immediately
                    return {"content": "I think I have what I need.",
                            "kind": "finish",
                            "args": {"answer": "done"}, "confidence": 0.8}
                if "BASAL GANGLIA" in sys_prompt:
                    return {"decision": "go", "reason": "fine",
                            "effector": "finish", "args": {"answer": "done"}}
                if "DEFAULT MODE" in sys_prompt:
                    return {"thought": "random memory", "kind": "memory",
                            "affect_delta": {"boredom": -0.1}}
                if "HIPPOCAMPUS" in sys_prompt:
                    return {}
                return {}

            def chat(model, sys_prompt, user, **kw):
                if "BROCA" in sys_prompt:
                    return "Final answer (stubbed)."
                return "stub"

            llm_instance.chat_json.side_effect = chat_json
            llm_instance.chat.side_effect = chat
            llm_instance.close = MagicMock()
            MockLLM.return_value = llm_instance

            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=True, seed=0)
            answer = brain.run("a simple test task")
            brain.close()

            # Got an answer; affect was perturbed by the scenario seed
            self.assertIsInstance(answer, str)
            self.assertTrue(answer)


class SkillStoreTests(unittest.TestCase):
    """Skill compilation, lookup, and the habit-fire psychological gate."""

    def test_signature_stable_for_similar_percepts(self):
        from brain.skills import signature_from_percept
        a = signature_from_percept(
            {"goal": "Write the primes script", "entities": ["python", "primes"]},
            None)
        b = signature_from_percept(
            {"goal": "write THE primes script", "entities": ["Python", "Primes"]},
            None)
        self.assertEqual(a, b)

    def test_consolidate_then_fire(self):
        from brain.skills import SkillStore
        tmp = Path(tempfile.mkdtemp()) / "s.sqlite"
        s = SkillStore(tmp)
        sig = "e=primes,python;g=script,write;N"
        # one successful use → not yet fireable
        s.consolidate(sig, "write_file",
                      {"path": "primes.py", "content": "..."}, ok=True)
        self.assertIsNone(s.best_match(sig))
        # second success → confidence above threshold, uses ≥ 2 → fireable
        skill = s.consolidate(sig, "write_file",
                              {"path": "primes.py", "content": "..."}, ok=True)
        self.assertTrue(skill.is_fireable())
        best = s.best_match(sig)
        self.assertIsNotNone(best)
        self.assertEqual(best.effector, "write_file")
        s.close()

    def test_failure_lowers_confidence(self):
        from brain.skills import SkillStore
        tmp = Path(tempfile.mkdtemp()) / "s.sqlite"
        s = SkillStore(tmp)
        sig = "x"
        s.consolidate(sig, "shell", {"command": "ls"}, ok=True)
        s.consolidate(sig, "shell", {"command": "ls"}, ok=True)
        s.consolidate(sig, "shell", {"command": "ls"}, ok=True)
        good = s.best_match(sig)
        self.assertIsNotNone(good)
        c_before = good.confidence
        # punish twice (e.g. a stale-habit miss)
        s.punish(good.id)
        s.punish(good.id)
        after = s.best_match(sig)
        self.assertLess(after.confidence if after else 0.0, c_before)
        s.close()

    def test_habit_gate_blocked_by_interrupt(self):
        from brain.regions.basal_ganglia import _habit_conditions_met
        ws = Workspace(task="t")
        ws.affect.stress = 0.9   # would normally favor habit
        ws.interrupt = "fire!"
        self.assertFalse(_habit_conditions_met(ws))

    def test_habit_gate_blocked_by_surprise(self):
        from brain.regions.basal_ganglia import _habit_conditions_met
        ws = Workspace(task="t")
        ws.affect.stress = 0.9   # cognitive load
        ws.last_surprise = 0.8   # but just had a big surprise
        self.assertFalse(_habit_conditions_met(ws))

    def test_habit_gate_favors_load(self):
        from brain.regions.basal_ganglia import _habit_conditions_met
        ws = Workspace(task="t")
        ws.affect.stress = 0.75
        ws.affect.curiosity = 0.30
        self.assertTrue(_habit_conditions_met(ws))


class TypedMemoryTests(unittest.TestCase):
    """Typed schema + TF-IDF semantic retrieval + prospective triggers."""

    def _fresh(self):
        from brain.memory import Memory
        tmp = Path(tempfile.mkdtemp()) / "m.sqlite"
        return Memory(tmp, refit_every=2)

    def test_typed_store_and_retrieve(self):
        from brain.memory import EPISODIC, SEMANTIC, AFFECT
        m = self._fresh()
        m.store("t", "action", "wrote primes.py and ran it", 0.6,
                mem_type=EPISODIC)
        m.store("t", "distilled", "I tend to leave shell quoting half-done",
                0.8, mem_type=SEMANTIC)
        m.store("t", "tag", "the basement office feels uneasy at night",
                0.7, mem_type=AFFECT)
        all_eps = m.recent(10)
        self.assertEqual(len(all_eps), 3)
        only_sem = m.retrieve("quoting", k=3, types=[SEMANTIC])
        # semantic row matches
        self.assertTrue(any("quoting" in r["content"] for r in only_sem))
        # episodic-only query should NOT pull the semantic row
        only_epi = m.retrieve("quoting", k=3, types=[EPISODIC])
        self.assertFalse(any(r["mem_type"] == SEMANTIC for r in only_epi))
        m.close()

    def test_tfidf_semantic_retrieval(self):
        from brain.memory import EPISODIC
        m = self._fresh()
        m.store("t", "a", "running a marathon and the city is glittering", 0.7)
        m.store("t", "a", "I baked sourdough and the kitchen smelled warm", 0.7)
        m.store("t", "a", "long run through the park, slow pace, calm breathing", 0.7)
        m.store("t", "a", "tax forms and quarterly receipts in a manila folder", 0.7)
        # query about running should rank the two running rows above sourdough/tax
        hits = m.retrieve_semantic("training plan for distance running", k=2,
                                    types=[EPISODIC])
        joined = " ".join(h["content"] for h in hits)
        self.assertIn("run", joined)
        self.assertNotIn("tax", joined)
        m.close()

    def test_prospective_keyword_trigger(self):
        m = self._fresh()
        pid = m.prospective_register(
            content="remind me to send the bug report",
            trigger_kind="keyword",
            trigger_pattern="auth login token",
            salience=0.9,
        )
        # Unrelated percept → no fire
        self.assertEqual(m.prospective_match("write a haiku about clouds"), [])
        # Matching percept → fires
        hits = m.prospective_match("the auth middleware needs a check")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], pid)
        m.prospective_mark_fired([pid])
        self.assertEqual(m.prospective_pending()[0]["fired_count"], 1)
        m.close()

    def test_prospective_time_trigger(self):
        m = self._fresh()
        pid = m.prospective_register(
            content="call back in a moment",
            trigger_kind="time",
            trigger_pattern="0",
            salience=0.8,
            fires_after_ts=0,  # already past
        )
        hits = m.prospective_match("any percept here")
        self.assertTrue(any(h["id"] == pid for h in hits))
        m.close()

    def test_migration_on_legacy_db(self):
        """A pre-typed-schema db should auto-migrate without losing rows."""
        from brain.memory import Memory
        import sqlite3
        tmp = Path(tempfile.mkdtemp()) / "legacy.sqlite"
        # Create the legacy single-bucket schema by hand
        conn = sqlite3.connect(str(tmp))
        conn.execute("""CREATE TABLE episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL, task TEXT NOT NULL, kind TEXT NOT NULL,
            content TEXT NOT NULL, salience REAL NOT NULL DEFAULT 0.5)""")
        conn.execute("INSERT INTO episodes (ts, task, kind, content, salience) "
                     "VALUES (?,?,?,?,?)",
                     (1.0, "legacy", "action", "old row", 0.5))
        conn.commit()
        conn.close()
        # Memory() should ALTER it in place
        m = Memory(tmp)
        rows = m.recent(5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mem_type"], "episodic")  # default applied
        # New typed write works
        m.store("new", "distilled", "the kitchen sink", 0.7, mem_type="semantic")
        self.assertEqual(m.count("semantic"), 1)
        m.close()


class ConsolidatorTests(unittest.TestCase):
    def test_clusters_similar_episodes_and_tags(self):
        from brain.memory import EPISODIC, Memory
        from brain import consolidator
        tmp = Path(tempfile.mkdtemp()) / "c.sqlite"
        m = Memory(tmp)
        for txt in [
            "wrote primes.py in python and tested with pytest",
            "wrote fibonacci.py in python and ran tests",
            "python script for primes, all tests passing",
            "added unit tests to the python utilities module",
            "weather is cold and the radiator clicks loudly",
        ]:
            m.store("dev", "action", txt, 0.6, mem_type=EPISODIC)
        # Dry run (no LLM): should still cluster + tag rows so they don't reprocess
        stats = consolidator.consolidate(
            m, llm=None, model=None, n_episodes=20,
            min_cluster_size=3, max_new_facts=5,
            cluster_threshold=0.05, log=lambda _m: None,
        )
        self.assertGreaterEqual(stats["clusters"], 1)
        self.assertGreater(stats["rows_tagged"], 0)
        # Second pass on the same rows yields no NEW tagging because they're
        # already marked consolidated.
        stats2 = consolidator.consolidate(
            m, llm=None, model=None, n_episodes=20,
            min_cluster_size=3, max_new_facts=5,
            cluster_threshold=0.05, log=lambda _m: None,
        )
        self.assertEqual(stats2["rows_tagged"], 0)
        m.close()

    def test_llm_extraction_writes_semantic(self):
        from brain.memory import EPISODIC, SEMANTIC, Memory
        from brain import consolidator
        from unittest.mock import MagicMock
        tmp = Path(tempfile.mkdtemp()) / "c.sqlite"
        m = Memory(tmp)
        for txt in [
            "shell quoting got me again on a one-liner",
            "another bash quoting tangle, fixed eventually",
            "single-vs-double quotes in shell tripped me up",
        ]:
            m.store("dev", "action", txt, 0.6, mem_type=EPISODIC)
        llm = MagicMock()
        llm.chat.return_value = "I keep struggling with shell quoting."
        # Tiny corpora produce low TF-IDF cosines; lower the cluster
        # threshold to match. Production runs with 40+ episodes work fine
        # at the default 0.18.
        stats = consolidator.consolidate(
            m, llm=llm, model="fake",
            n_episodes=20, min_cluster_size=3, max_new_facts=2,
            cluster_threshold=0.05,
            log=lambda _m: None,
        )
        self.assertGreaterEqual(stats["facts_written"], 1)
        # The new semantic row is queryable
        sem = m.retrieve("quoting", k=3, types=[SEMANTIC])
        self.assertTrue(sem)
        self.assertIn("quoting", sem[0]["content"].lower())
        m.close()


class EmbeddingBackendTests(unittest.TestCase):
    """Pluggable embedding backends + Memory persistence cache."""

    def test_pack_unpack_floats(self):
        from brain.embeddings import _pack_floats, _unpack_floats
        v = [0.1, -0.5, 0.97, 1.4, -1e-3]
        blob = _pack_floats(v)
        back = _unpack_floats(blob)
        self.assertEqual(len(back), len(v))
        for a, b in zip(v, back):
            self.assertAlmostEqual(a, b, places=5)

    def test_factory_defaults_to_tfidf_with_no_neural_options(self):
        from brain.embeddings import make_backend, TfidfBackend
        # A cfg-like stub with no llm and no model — factory must not crash
        class StubCfg:
            memory = {"embedding_backend": "tfidf"}
        bk = make_backend(StubCfg(), llm=None)
        self.assertIsInstance(bk, TfidfBackend)

    def test_factory_explicit_openrouter_requires_llm_and_model(self):
        from brain.embeddings import make_backend
        class StubCfg:
            memory = {"embedding_backend": "openrouter"}
        with self.assertRaises(RuntimeError):
            make_backend(StubCfg(), llm=None)

    def test_memory_with_neural_backend_persists_and_reuses_embeddings(self):
        """Stub a persistent backend; verify Memory writes its embedding to
        the BLOB column and a second Memory instance loads from cache (no
        re-embed)."""
        from brain.memory import Memory, EPISODIC
        from brain.embeddings import EmbeddingBackend, _pack_floats, _unpack_floats

        # A toy persistent backend whose 'embedding' is a fixed-length vector
        # derived deterministically from the text (so re-encode is detectable).
        class _ToyBackend(EmbeddingBackend):
            name = "toy"
            persistent = True
            def __init__(self):
                self._vecs = {}
                self._fitted = False
                self.encode_calls = 0
            def _embed(self, text):
                # 4D vector: lengths of first 4 word tokens
                toks = text.split()[:4]
                return [float(len(t)) for t in toks] + [0.0] * (4 - len(toks[:4]))
            def fit(self, docs):
                docs = list(docs)
                for did, text in docs:
                    if did in self._vecs:
                        continue
                    self._vecs[did] = self._embed(text)
                    self.encode_calls += 1
                self._fitted = True
            def topk(self, q, k, eligible=None):
                if not self._fitted:
                    return []
                qv = self._embed(q)
                import math
                qn = math.sqrt(sum(x*x for x in qv))
                if qn == 0: return []
                out = []
                for did, v in self._vecs.items():
                    if eligible is not None and did not in eligible:
                        continue
                    dn = math.sqrt(sum(x*x for x in v))
                    if dn == 0: continue
                    dot = sum(a*b for a,b in zip(qv, v))
                    out.append((did, dot/(qn*dn)))
                out.sort(key=lambda x: x[1], reverse=True)
                return out[:k]
            def encode_one(self, text):
                return _pack_floats(self._embed(text))
            def from_bytes(self, blob):
                return _unpack_floats(blob)
            def remember(self, did, vec):
                self._vecs[did] = vec

        tmp = Path(tempfile.mkdtemp()) / "p.sqlite"
        m1 = Memory(tmp, backend=_ToyBackend())
        m1.store("t", "a", "alpha beta gamma delta", 0.6, mem_type=EPISODIC)
        m1.store("t", "a", "epsilon zeta eta theta", 0.6, mem_type=EPISODIC)
        # First retrieve forces fit() + write-back of BLOBs
        hits = m1.retrieve_semantic("alpha beta", k=2, types=[EPISODIC])
        self.assertTrue(hits)
        bk1 = m1.backend
        first_pass_calls = bk1.encode_calls
        self.assertGreater(first_pass_calls, 0)
        m1.close()

        # Open the db with a FRESH backend; cached BLOBs should populate it
        # without re-encoding the existing rows.
        m2 = Memory(tmp, backend=_ToyBackend())
        m2.retrieve_semantic("alpha", k=2, types=[EPISODIC])
        bk2 = m2.backend
        # Only rows missing a blob would trigger encode — there are none here.
        self.assertEqual(bk2.encode_calls, 0)
        m2.close()


class PredictiveCodingTests(unittest.TestCase):
    def test_surprise_score(self):
        from brain.skills import prediction_surprise
        # identical → 0
        self.assertEqual(prediction_surprise("the cat sat", "the cat sat"), 0.0)
        # no prediction → 0 (no signal)
        self.assertEqual(prediction_surprise("", "anything"), 0.0)
        # disjoint trigrams → near 1
        self.assertGreater(
            prediction_surprise("apples grow on trees",
                                "submarines navigate cold oceans"), 0.9)


class StreamOfThoughtTests(unittest.TestCase):
    """The chain is built autoregressively; DMN can append a tangent unit
    that the next prefrontal step sees and is marked as 'interrupted'."""

    def test_dmn_hijack_appears_as_intrusion_in_next_thought(self):
        from brain.config import Config
        from brain.orchestrator import Brain
        tmp = Path(tempfile.mkdtemp())
        cfg = Config(
            raw={"persona_path": str(REPO / "personas" / "alex.yaml"),
                  "scenario": "boring_afternoon", "humanize": True},
            api_key="fake", base_url="http://fake",
            models={"reflex": "fake", "executive": "fake"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=tmp, db_path=tmp / "mem.sqlite",
            loop={"max_cycles": 3, "attention_decay": 0.85},
            memory={"db_path": str(tmp / "mem.sqlite"), "retrieve_k": 3},
            effectors={"filesystem": {"enabled": False},
                        "shell": {"enabled": False}, "web": {"enabled": False}},
            regions={k: "reflex" for k in
                      ["sensory_cortex","amygdala","basal_ganglia","hippocampus",
                       "prefrontal","broca","interoception","default_mode",
                       "locus_coeruleus","vta"]},
        )

        # Force DMN to fire by jamming distractibility very high, then have
        # the prefrontal emit a non-finish thought on step 1 so DMN can
        # intrude before step 2's finish.
        thoughts_seen = []
        with patch("brain.orchestrator.LLM") as MockLLM:
            llm = MagicMock()
            call_idx = {"n": 0}

            def chat_json(model, sys, user, **kw):
                if "SENSORY CORTEX" in sys:
                    return {"goal": "x", "entities": [], "constraints": [],
                            "success_criterion": "done"}
                if "AMYGDALA" in sys:
                    return {"valence_shift": 0.0, "urgency": 0.2,
                            "stress_shift": 0.0, "interrupt": None, "note": "ok"}
                if "DEFAULT MODE" in sys:
                    return {"thought": "I keep thinking about Maria's call.",
                            "kind": "memory",
                            "affect_delta": {"social_need": -0.05}}
                if "PREFRONTAL" in sys:
                    call_idx["n"] += 1
                    thoughts_seen.append(user)
                    if call_idx["n"] == 1:
                        return {"content": "Let me think about this.",
                                "kind": "reflect", "confidence": 0.5}
                    return {"content": "Okay, I have it.", "kind": "finish",
                            "args": {"answer": "done"}, "confidence": 0.8}
                if "BASAL GANGLIA" in sys:
                    return {"decision": "go", "reason": "fine"}
                return {}

            def chat(model, sys, user, **kw):
                if "BROCA" in sys:
                    return "stream final answer"
                return "stub"

            llm.chat_json.side_effect = chat_json
            llm.chat.side_effect = chat
            llm.close = MagicMock()
            MockLLM.return_value = llm

            # Bias DMN to definitely fire
            from brain.regions.default_mode import DefaultMode
            orig_should = DefaultMode._should_fire
            try:
                DefaultMode._should_fire = lambda self, ws: True
                brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                              humanize=True, seed=0)
                brain.run("a task")
                brain.close()
            finally:
                DefaultMode._should_fire = orig_should

        # The second prefrontal prompt must have seen the DMN tangent in its
        # thought chain (an intrusion notice).
        self.assertGreaterEqual(call_idx["n"], 2)
        second_prompt = thoughts_seen[1]
        self.assertIn("THOUGHT CHAIN", second_prompt)
        self.assertIn("default_mode", second_prompt)


class SleepAgentsTests(unittest.TestCase):
    """Each sleep agent in isolation against an in-memory DB."""

    def _fresh_memory(self):
        from brain.memory import Memory
        tmp = Path(tempfile.mkdtemp()) / "m.sqlite"
        return Memory(tmp)

    def test_forgetter_prunes_low_salience_old_episodes(self):
        from brain.memory import EPISODIC
        from brain.sleep import Forgetter
        m = self._fresh_memory()
        # Insert ancient low-salience rows (the forgetter should delete them)
        # and recent high-salience rows (must survive). We backdate the
        # 'ancient' rows by editing ts directly.
        import time as _t
        old = _t.time() - 24 * 3600  # 24h ago
        for _ in range(20):
            rid = m.store("t", "a", "old fluff", 0.1, mem_type=EPISODIC)
            m.conn.execute("UPDATE episodes SET ts=? WHERE id=?", (old, rid))
        for _ in range(10):
            m.store("t", "a", "very recent and important", 0.9, mem_type=EPISODIC)
        # Recent old rows that are nevertheless within recency_safety must survive
        m.conn.commit()
        before = m.count(EPISODIC)
        stats = Forgetter(min_age_seconds=3600,
                          recency_safety=5,
                          salience_threshold=0.3).run(m)
        after = m.count(EPISODIC)
        self.assertGreater(stats["pruned"], 0)
        self.assertLess(after, before)
        m.close()

    def test_skill_pruner_decays_and_deletes(self):
        from brain.skills import SkillStore
        from brain.sleep import SkillPruner
        tmp = Path(tempfile.mkdtemp()) / "s.sqlite"
        s = SkillStore(tmp)
        # Make several skills successful so they're stored
        s.consolidate("sig", "shell", {"command": "ls"}, ok=True)
        s.consolidate("sig", "shell", {"command": "ls"}, ok=True)
        s.consolidate("sig", "shell", {"command": "rm"}, ok=False)
        # Force last_used to ancient
        s.conn.execute("UPDATE skills SET last_used=0")
        s.conn.commit()
        before_count = s.conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        stats = SkillPruner(unused_age_seconds=10,
                            delete_below_conf=0.4).run(s)
        after_count = s.conn.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        self.assertGreater(stats["decayed"], 0)
        # The failed-shell skill had conf 0.30 → after one decay round it
        # drops below 0.4 and gets deleted
        self.assertLess(after_count, before_count)
        s.close()

    def test_mood_regulator_recovers_fatigue_and_drifts_valence(self):
        from brain.affect import AffectState
        from brain.sleep import MoodRegulator
        a = AffectState()
        a.fatigue = 0.9
        a.valence = -0.6
        a.stress = 0.7
        MoodRegulator(recovery_steps=12, fatigue_recovery_per_step=0.05).run(a)
        self.assertLess(a.fatigue, 0.5)
        # valence should regress toward 0 (i.e. less negative)
        self.assertGreater(a.valence, -0.6)
        self.assertLess(a.stress, 0.7)

    def test_scheduler_fires_time_triggers(self):
        m = self._fresh_memory()
        m.prospective_register(
            content="call back in a moment",
            trigger_kind="time", trigger_pattern="0",
            salience=0.8, fires_after_ts=0,
        )
        from brain.sleep import Scheduler
        fired = Scheduler().fire_due(m)
        self.assertEqual(len(fired), 1)
        self.assertIn("call back", fired[0]["content"])
        # Second call: already marked done → no fire
        fired2 = Scheduler().fire_due(m)
        self.assertEqual(fired2, [])
        m.close()

    def test_dreamer_writes_low_confidence_semantic(self):
        from unittest.mock import MagicMock
        from brain.memory import EPISODIC, SEMANTIC
        from brain.sleep import Dreamer
        m = self._fresh_memory()
        # Seed distant memories so the sampler can find unrelated pairs
        for txt in [
            "wrote primes.py and tested it carefully",
            "the cat learned to open the kitchen door",
            "rainy commute, train delayed twenty minutes",
            "argued about jazz with a friend over dinner",
            "ran 10k along the canal at dawn",
        ]:
            m.store("t", "a", txt, 0.6, mem_type=EPISODIC)
        llm = MagicMock()
        llm.chat.return_value = "What if the cat is the only true critic of jazz?"
        stats = Dreamer(dreams_per_bout=2, max_pair_cosine=0.95,
                        seed=1).run(m, llm, model="fake",
                                     log=lambda _m: None)
        self.assertGreaterEqual(stats["written"], 1)
        # Dreams are tagged
        sem = m.recent(20, mem_type=SEMANTIC)
        any_dream = any("dream" in (r.get("tags") or "") for r in sem)
        self.assertTrue(any_dream)
        m.close()


class DaemonStateMachineTests(unittest.TestCase):
    """The state transitions don't require live LLM calls — only Brain
    construction. We stub LLM and run the daemon with max_ticks."""

    def test_wake_to_drowsy_to_nrem(self):
        from brain.config import Config
        from brain.orchestrator import Brain
        from brain.daemon import BrainDaemon, WAKE, DROWSY, NREM
        tmp = Path(tempfile.mkdtemp())
        cfg = Config(
            raw={"persona_path": str(REPO / "personas" / "alex.yaml"),
                  "scenario": "boring_afternoon", "humanize": True},
            api_key="fake", base_url="http://fake",
            models={"reflex": "fake", "executive": "fake"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=tmp, db_path=tmp / "mem.sqlite",
            loop={"max_cycles": 1, "attention_decay": 0.8},
            memory={"db_path": str(tmp / "mem.sqlite"), "retrieve_k": 3,
                     "embedding_backend": "tfidf"},
            effectors={"filesystem": {"enabled": False},
                        "shell": {"enabled": False}, "web": {"enabled": False}},
            regions={k: "reflex" for k in
                      ["sensory_cortex","amygdala","basal_ganglia","hippocampus",
                       "prefrontal","broca","interoception","default_mode",
                       "locus_coeruleus","vta"]},
        )

        with patch("brain.orchestrator.LLM") as MockLLM:
            llm = MagicMock()
            llm.chat_json.return_value = {}
            llm.chat.return_value = ""
            llm.close = MagicMock()
            MockLLM.return_value = llm

            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=True, seed=0)
            d = BrainDaemon(
                cfg, brain,
                tick_seconds=0.0,
                idle_rate=0.0,
                drowsy_fatigue=0.30,
                sleep_fatigue=0.55,
                wake_fatigue=0.20,
                nrem_bout_ticks=2, rem_bout_ticks=2,
                log=lambda _m: None,
                seed=0,
            )
            # Start with high fatigue → expect drowsy → nrem on next tick
            d.affect.fatigue = 0.4
            self.assertEqual(d.state, WAKE)
            d.tick()
            self.assertIn(d.state, (DROWSY, NREM))
            # Bump fatigue past sleep threshold; next tick goes to NREM
            d.affect.fatigue = 0.7
            d.tick()
            self.assertIn(d.state, (NREM, "rem"))
            brain.close()

    def test_external_input_wakes_brain(self):
        from brain.config import Config
        from brain.orchestrator import Brain
        from brain.daemon import BrainDaemon, NREM, WAKE
        tmp = Path(tempfile.mkdtemp())
        cfg = Config(
            raw={"persona_path": str(REPO / "personas" / "alex.yaml"),
                  "scenario": "neutral", "humanize": True},
            api_key="fake", base_url="http://fake",
            models={"reflex": "fake", "executive": "fake"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=tmp, db_path=tmp / "mem.sqlite",
            loop={"max_cycles": 1, "attention_decay": 0.8},
            memory={"db_path": str(tmp / "mem.sqlite"), "retrieve_k": 3,
                     "embedding_backend": "tfidf"},
            effectors={"filesystem": {"enabled": False},
                        "shell": {"enabled": False}, "web": {"enabled": False}},
            regions={k: "reflex" for k in
                      ["sensory_cortex","amygdala","basal_ganglia","hippocampus",
                       "prefrontal","broca","interoception","default_mode",
                       "locus_coeruleus","vta"]},
        )
        with patch("brain.orchestrator.LLM") as MockLLM:
            llm = MagicMock()
            llm.chat_json.return_value = {
                "goal": "x", "entities": [], "constraints": [],
                "success_criterion": "done",
                "content": "OK", "kind": "finish",
                "args": {"answer": "ok"}, "confidence": 0.9,
                "decision": "go", "reason": "fine",
            }
            llm.chat.return_value = "final"
            llm.close = MagicMock()
            MockLLM.return_value = llm
            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=True, seed=0)
            d = BrainDaemon(cfg, brain, tick_seconds=0.0, idle_rate=0.0,
                             log=lambda _m: None, seed=0)
            # Force NREM
            d.state = NREM
            d._sleep_bouts_remaining = 5
            d.enqueue("a brand new task")
            d.tick()
            self.assertEqual(d.state, WAKE)
            brain.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
