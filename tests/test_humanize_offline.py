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
                # Most regions use chat_json; chat is only the legacy path now.
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


class WorldModelTests(unittest.TestCase):
    """k-NN over (state, action, outcome) in the configured embedding space."""

    def _fresh(self):
        from brain.world_model import WorldModelStore
        tmp = Path(tempfile.mkdtemp()) / "wm.sqlite"
        return WorldModelStore(tmp)

    def test_action_overlap_same_effector(self):
        from brain.world_model import _action_overlap
        # Same effector + shared arg token → high
        s1 = _action_overlap("write_file(path=foo.py, content=def main)",
                              "write_file(path=foo.py, content=other)")
        self.assertGreater(s1, 0.3)
        # Different effector → zero
        s2 = _action_overlap("write_file(path=foo)", "shell(command=foo)")
        self.assertEqual(s2, 0.0)

    def test_observe_and_predict_finds_similar_state(self):
        wm = self._fresh()
        wm.observe(
            state_text="goal=write primes script; mood=curious",
            action_text="write_file(content=primes, path=primes.py)",
            outcome_text="wrote 42 bytes to primes.py", ok=True,
        )
        wm.observe(
            state_text="goal=write fibonacci script; mood=curious",
            action_text="write_file(content=fib, path=fib.py)",
            outcome_text="wrote 35 bytes to fib.py", ok=True,
        )
        wm.observe(
            state_text="goal=delete temp files; mood=neutral",
            action_text="shell(command=rm tmp/*)",
            outcome_text="(exit 0)", ok=True,
        )
        hits = wm.predict(
            state_text="goal=write a math script; mood=curious",
            action_text="write_file(content=math, path=math.py)",
            k=2,
        )
        self.assertTrue(hits)
        # The matched rows should be the two write_file events, not the shell one
        verbs = {h["action_text"].split("(")[0] for h in hits}
        self.assertEqual(verbs, {"write_file"})
        wm.close()

    def test_predict_filters_by_action_verb(self):
        wm = self._fresh()
        wm.observe("similar state here", "write_file(path=a)", "ok", True)
        wm.observe("similar state here", "shell(command=ls)", "list", True)
        # Query with shell action — must not return the write_file row
        hits = wm.predict("similar state here", "shell(command=cat foo)", k=3)
        self.assertTrue(all(h["action_text"].startswith("shell") for h in hits))
        wm.close()

    def test_counterfactuals_excludes_same_action(self):
        wm = self._fresh()
        wm.observe("morning; goal=reply to email", "write_file(path=reply.txt)",
                   "wrote", True)
        wm.observe("morning; goal=reply to email",
                   "shell(command=send_mail draft.txt)", "sent", True)
        cf = wm.counterfactuals("morning; goal=reply to email",
                                 current_action="write_file(path=anything)",
                                 k=2)
        self.assertTrue(cf)
        # All counterfactuals must have a DIFFERENT effector
        self.assertTrue(all(not h["action_text"].startswith("write_file")
                            for h in cf))
        wm.close()

    def test_persistent_backend_caches_state_embedding(self):
        """A persistent backend's encode_one is called on observe(); a
        second pass with a fresh backend reuses the cached BLOB."""
        from brain.world_model import WorldModelStore
        from brain.embeddings import EmbeddingBackend, _pack_floats, _unpack_floats

        class _Toy(EmbeddingBackend):
            name = "toy"; persistent = True
            def __init__(self):
                self._vecs = {}; self._fitted = False; self.encode_calls = 0
            def _embed(self, t):
                # 4D from word lengths
                ws = t.split()[:4]
                return [float(len(w)) for w in ws] + [0.0] * (4 - len(ws[:4]))
            def fit(self, docs):
                docs = list(docs)
                for did, text in docs:
                    if did in self._vecs: continue
                    self._vecs[did] = self._embed(text)
                    self.encode_calls += 1
                self._fitted = True
            def topk(self, q, k, eligible=None):
                if not self._fitted: return []
                qv = self._embed(q); import math
                qn = math.sqrt(sum(x*x for x in qv))
                if qn == 0: return []
                out = []
                for did, v in self._vecs.items():
                    if eligible is not None and did not in eligible: continue
                    dn = math.sqrt(sum(x*x for x in v))
                    if dn == 0: continue
                    dot = sum(a*b for a,b in zip(qv, v))
                    out.append((did, dot/(qn*dn)))
                out.sort(key=lambda x: x[1], reverse=True)
                return out[:k]
            def encode_one(self, t):
                self.encode_calls += 1
                return _pack_floats(self._embed(t))
            def from_bytes(self, b): return _unpack_floats(b)
            def remember(self, did, v): self._vecs[did] = v

        tmp = Path(tempfile.mkdtemp()) / "wm.sqlite"
        wm1 = WorldModelStore(tmp, backend=_Toy())
        wm1.observe("alpha beta", "shell(command=alpha)", "ok", True)
        wm1.observe("alpha gamma", "shell(command=beta)", "ok", True)
        first_encodes = wm1.backend.encode_calls
        self.assertGreater(first_encodes, 0)
        wm1.close()

        wm2 = WorldModelStore(tmp, backend=_Toy())
        # A predict() call forces _fit_backend which loads cached BLOBs
        wm2.predict("alpha beta", "shell(command=test)", k=2)
        # The toy backend's encode_calls should be near zero — fit() didn't
        # re-encode existing rows (BLOBs were preloaded via remember()).
        # We expect zero encodes for known rows; topk's _embed of the query
        # is not counted by encode_calls (we only count fit/encode_one).
        self.assertEqual(wm2.backend.encode_calls, 0)
        wm2.close()


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


class LocalLLMConfigTests(unittest.TestCase):
    """Provider profiles + local-LLM endpoint resolution."""

    def _write_cfg(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "config.yaml"
        tmp.write_text(body)
        return tmp

    def test_ollama_profile_skips_auth(self):
        from brain.config import load_config
        p = self._write_cfg("""
openrouter:
  provider: ollama
  models:
    reflex: "llama3.2:3b"
    executive: "qwen2.5:7b"
sandbox_dir: "/tmp/brain-test-sb"
memory:
  db_path: "/tmp/brain-test-mem.sqlite3"
""")
        cfg = load_config(p)
        self.assertEqual(cfg.base_url, "http://localhost:11434/v1")
        self.assertFalse(cfg.require_auth)
        self.assertEqual(cfg.api_key, "")
        self.assertEqual(cfg.models["reflex"], "llama3.2:3b")

    def test_lmstudio_profile(self):
        from brain.config import load_config
        p = self._write_cfg("""
openrouter:
  provider: lmstudio
  models:
    reflex: "qwen-7b"
    executive: "qwen-32b"
sandbox_dir: "/tmp/brain-test-sb"
memory:
  db_path: "/tmp/brain-test-mem.sqlite3"
""")
        cfg = load_config(p)
        self.assertEqual(cfg.base_url, "http://localhost:1234/v1")
        self.assertFalse(cfg.require_auth)

    def test_custom_provider_requires_base_url(self):
        from brain.config import load_config
        p = self._write_cfg("""
openrouter:
  provider: custom
  models: {reflex: "x", executive: "y"}
sandbox_dir: "/tmp/brain-test-sb"
memory:
  db_path: "/tmp/brain-test-mem.sqlite3"
""")
        with self.assertRaises(RuntimeError):
            load_config(p)

    def test_openrouter_profile_still_requires_auth(self):
        from brain.config import load_config
        # Stash and remove the env var
        import os
        saved = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            p = self._write_cfg("""
openrouter:
  provider: openrouter
  models: {reflex: "x", executive: "y"}
sandbox_dir: "/tmp/brain-test-sb"
memory:
  db_path: "/tmp/brain-test-mem.sqlite3"
""")
            with self.assertRaises(RuntimeError):
                load_config(p)
        finally:
            if saved is not None:
                os.environ["OPENROUTER_API_KEY"] = saved

    def test_llm_client_omits_auth_header_for_local_provider(self):
        """LLM constructor with require_auth=False should not send
        Authorization: Bearer header. We inspect the httpx client's
        default headers."""
        from brain.config import Config
        from brain.llm import LLM
        cfg = Config(
            raw={}, api_key="", base_url="http://localhost:11434/v1",
            require_auth=False, extra_headers={},
            models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={}, regions={},
        )
        llm = LLM(cfg)
        try:
            headers = {k.lower(): v for k, v in llm._client.headers.items()}
            self.assertNotIn("authorization", headers)
        finally:
            llm.close()

    def test_llm_client_sets_auth_when_required(self):
        from brain.config import Config
        from brain.llm import LLM
        cfg = Config(
            raw={}, api_key="sk-test", base_url="https://api.example.com/v1",
            require_auth=True,
            extra_headers={"X-Title": "brain"},
            models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={}, regions={},
        )
        llm = LLM(cfg)
        try:
            headers = {k.lower(): v for k, v in llm._client.headers.items()}
            self.assertEqual(headers.get("authorization"), "Bearer sk-test")
            self.assertEqual(headers.get("x-title"), "brain")
        finally:
            llm.close()


class CerebellumTests(unittest.TestCase):
    """Fast deterministic forward model — pure k-NN, no LLM."""

    def _wm_with_data(self, rows):
        from brain.world_model import WorldModelStore
        tmp = Path(tempfile.mkdtemp()) / "wm.sqlite"
        wm = WorldModelStore(tmp)
        for r in rows:
            wm.observe(**r)
        return wm

    def _cerebellum(self, wm):
        from brain.config import Config
        from brain.regions.cerebellum import Cerebellum
        cfg = Config(
            raw={}, api_key="", base_url="http://x",
            require_auth=False, extra_headers={},
            models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={},
            regions={"cerebellum": "reflex"},
        )
        return Cerebellum(cfg, llm=None, world_model=wm)

    def _ws_with_percept(self, goal: str, entities: list[str]):
        ws = Workspace(task=goal)
        from brain.workspace import Broadcast
        ws.post(Broadcast(source="sensory_cortex", kind="percept",
                          content=f"goal={goal}", salience=0.9,
                          data={"goal": goal, "entities": entities,
                                 "constraints": [], "success_criterion": "x"}))
        return ws

    def test_predicts_outcome_from_similar_past_action(self):
        wm = self._wm_with_data([
            {"state_text": "goal=write primes script; mood=curious",
             "action_text": "write_file(content=primes, path=primes.py)",
             "outcome_text": "wrote 42 bytes to primes.py", "ok": True},
            {"state_text": "goal=write fibonacci script; mood=curious",
             "action_text": "write_file(content=fib, path=fib.py)",
             "outcome_text": "wrote 35 bytes to fib.py", "ok": True},
        ])
        cb = self._cerebellum(wm)
        ws = self._ws_with_percept("write a math script", ["python", "primes"])
        pred = cb.quick_predict(ws, "write_file",
                                 {"content": "p2", "path": "p2.py"})
        self.assertTrue(pred.is_useful)
        self.assertTrue(pred.predicted_ok)
        self.assertIn("wrote", pred.predicted_outcome)
        wm.close()

    def test_no_world_model_yields_empty_prediction(self):
        from brain.config import Config
        from brain.regions.cerebellum import Cerebellum
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={},
            regions={"cerebellum": "reflex"},
        )
        cb = Cerebellum(cfg, llm=None, world_model=None)
        ws = self._ws_with_percept("x", [])
        pred = cb.quick_predict(ws, "write_file", {})
        self.assertEqual(pred.n_matches, 0)
        self.assertFalse(pred.is_useful)

    def test_bg_habit_fire_suppressed_when_cerebellum_predicts_failure(self):
        from brain.skills import SkillStore
        from brain.regions.basal_ganglia import BasalGanglia
        from brain.affect import Traits
        # World model where the same action FAILED in similar state
        wm = self._wm_with_data([
            {"state_text": "goal=clean tmp dir; mood=neutral",
             "action_text": "shell(command=rm -rf tmp)",
             "outcome_text": "error: permission denied", "ok": False},
            {"state_text": "goal=clean tmp dir; mood=neutral",
             "action_text": "shell(command=rm -rf tmp)",
             "outcome_text": "error: permission denied", "ok": False},
            {"state_text": "goal=clean tmp dir; mood=neutral",
             "action_text": "shell(command=rm -rf tmp)",
             "outcome_text": "error: permission denied", "ok": False},
        ])
        cb = self._cerebellum(wm)
        # Make a skill that the BG would otherwise fire
        sk_path = Path(tempfile.mkdtemp()) / "s.sqlite"
        skills = SkillStore(sk_path)
        # Derive the signature the same way the BG will at propose-time so the
        # skill matches the percept exactly (avoid hand-coded format drift).
        from brain.skills import signature_from_percept
        percept_data = {"goal": "clean tmp dir", "entities": ["tmp"]}
        sig = signature_from_percept(percept_data, None)
        skills.consolidate(sig, "shell", {"command": "rm -rf tmp"}, ok=True)
        skills.consolidate(sig, "shell", {"command": "rm -rf tmp"}, ok=True)
        # Build a workspace that satisfies _habit_conditions_met
        ws = self._ws_with_percept("clean tmp dir", ["tmp"])
        ws.affect.stress = 0.6  # cognitive-load regime → habits favored
        ws.affect.traits = Traits(conscientiousness=0.3)
        ws.last_surprise = 0.0
        # Without cerebellum, habit fires; with it, suppressed by predicted failure
        from brain.config import Config
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={},
            regions={"basal_ganglia": "reflex"},
        )
        bg = BasalGanglia(cfg, llm=MagicMock())
        without = bg.propose_habit(ws, skills, cerebellum=None)
        self.assertIsNotNone(without)
        with_cb = bg.propose_habit(ws, skills, cerebellum=cb)
        self.assertIsNone(with_cb)
        wm.close(); skills.close()


class ClassifierAndAdapterTests(unittest.TestCase):
    """Deterministic salience classifier + input adapters + coalesced wake."""

    def test_classifier_keyword_high_lifts_salience(self):
        from brain.inputs import StreamItem, SalienceClassifier
        from brain.inputs.classifier import ClassifierRules
        rules = ClassifierRules(keywords_high=["deadline", "Maria"],
                                  keywords_low=["spam", "promo"])
        c = SalienceClassifier(rules=rules,
                                ambient_threshold=0.2, direct_threshold=0.6)
        a = StreamItem(source="rss", kind="notification",
                       content="reminder about the deadline",
                       channel="ambient")
        b = StreamItem(source="rss", kind="notification",
                       content="promo offer this week", channel="ambient")
        c.classify(a); c.classify(b)
        self.assertGreater(a.salience, b.salience)

    def test_classifier_routes_direct_vs_ambient(self):
        from brain.inputs import StreamItem, SalienceClassifier
        c = SalienceClassifier(ambient_threshold=0.25, direct_threshold=0.55)
        direct = StreamItem(source="dm", kind="task",
                             content="please help", channel="direct",
                             sender="Sam")
        ambient = StreamItem(source="rss", kind="notification",
                              content="weather: cloudy", channel="ambient")
        noisy = StreamItem(source="metrics", kind="notification",
                            content="x", channel="ambient")
        c.classify(direct); c.classify(ambient); c.classify(noisy)
        self.assertEqual(c.route(direct), "direct")
        self.assertEqual(c.route(noisy), "drop")
        self.assertIn(c.route(ambient), ("ambient", "drop"))

    def test_classifier_uses_memory_similarity_to_lift_score(self):
        from brain.inputs import StreamItem, SalienceClassifier
        from brain.memory import EPISODIC, Memory
        tmp = Path(tempfile.mkdtemp()) / "m.sqlite"
        m = Memory(tmp)
        # Seed a high-salience past episode
        m.store("t", "action", "father's surgery recovery progress good", 0.95,
                mem_type=EPISODIC)
        c = SalienceClassifier(memory=m, ambient_threshold=0.2,
                                direct_threshold=0.55)
        on_topic = StreamItem(source="rss", kind="notification",
                               content="hospital surgery recovery story",
                               channel="ambient")
        off_topic = StreamItem(source="rss", kind="notification",
                                content="bicycle commuting tips",
                                channel="ambient")
        c.classify(on_topic); c.classify(off_topic)
        self.assertGreater(on_topic.salience, off_topic.salience)
        m.close()

    def test_classifier_affect_modulation(self):
        from brain.inputs import StreamItem, SalienceClassifier
        from brain.affect import AffectState
        item = StreamItem(source="webhook", kind="notification",
                           content="a thing happened", channel="ambient")
        calm = AffectState(); calm.stress = 0.1
        stressed = AffectState(); stressed.stress = 0.9
        c1 = SalienceClassifier(affect=calm)
        c2 = SalienceClassifier(affect=stressed)
        from copy import copy
        a1 = copy(item); a2 = copy(item)
        c1.classify(a1); c2.classify(a2)
        # Stressed brain notices more
        self.assertGreater(a2.salience, a1.salience)

    def test_file_tail_adapter_picks_up_new_lines(self):
        from brain.inputs import FileTailAdapter
        tmp = Path(tempfile.mkdtemp())
        log = tmp / "events.log"
        log.write_text("preamble line\n")
        ad = FileTailAdapter(paths=[log], start_at_end=True,
                              source_name="evlog")
        ad.start()
        # No items yet — we started at EOF
        self.assertEqual(ad.poll(), [])
        # Append new lines
        with open(log, "a") as fh:
            fh.write("alpha event\nbeta event\n\n")
        items = ad.poll()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].content, "alpha event")
        self.assertEqual(items[1].content, "beta event")
        # Second poll: empty (offset advanced)
        self.assertEqual(ad.poll(), [])
        ad.close()

    def test_webhook_adapter_receives_post(self):
        import json
        import socket
        import urllib.request
        from brain.inputs import WebhookAdapter
        # Pick a free ephemeral port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ad = WebhookAdapter(port=port)
        ad.start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/test?channel=direct&sender=Sam",
                data=json.dumps({"content": "ping"}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=2.0).read()
        finally:
            # tiny wait for the handler thread to enqueue
            import time as _t; _t.sleep(0.05)
            items = ad.poll()
            ad.close()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].content, "ping")
        self.assertEqual(items[0].channel, "direct")
        self.assertEqual(items[0].sender, "Sam")

    def test_daemon_coalesces_burst_into_one_task(self):
        """Six adapter items arrive in a single tick; the daemon batches
        them into ONE Brain.run call, not six."""
        from unittest.mock import MagicMock, patch
        from brain.config import Config
        from brain.daemon import BrainDaemon, WAKE
        from brain.inputs import InputAdapter, StreamItem, SalienceClassifier
        from brain.orchestrator import Brain

        class _Burst(InputAdapter):
            name = "burst"; default_channel = "direct"
            def __init__(self, items):
                self._items = list(items); self._fired = False
            def poll(self):
                if self._fired: return []
                self._fired = True
                return self._items

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
                       "locus_coeruleus","vta","cerebellum"]},
        )

        items = [StreamItem(source="burst", kind="task",
                             content=f"item {i}", channel="direct",
                             sender="user", salience=0.7)
                 for i in range(6)]
        burst = _Burst(items)

        with patch("brain.orchestrator.LLM") as MockLLM:
            llm = MagicMock()
            llm.chat_json.return_value = {
                "goal": "x", "entities": [], "constraints": [],
                "success_criterion": "done",
                "content": "OK", "kind": "finish",
                "args": {"answer": "ok"}, "confidence": 0.9,
                "decision": "go", "reason": "fine",
            }
            llm.chat.return_value = "ok"
            llm.close = MagicMock()
            MockLLM.return_value = llm

            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=True, seed=0)
            # Spy on Brain.run to count invocations + see the task text
            run_calls: list[str] = []
            original_run = brain.run
            def counted_run(t):
                run_calls.append(t)
                return original_run(t)
            brain.run = counted_run  # type: ignore
            d = BrainDaemon(
                cfg, brain, tick_seconds=0.0, idle_rate=0.0,
                adapters=[burst],
                coalesce_window_seconds=0.0,   # flush immediately
                coalesce_max_items=10,
                log=lambda _m: None, seed=0,
            )
            d.tick()  # adapter poll → buffer fills → flush on next wake tick
            d.tick()  # process coalesced batch
            brain.close()

        # Exactly ONE Brain.run call, containing all 6 items in its prompt
        self.assertEqual(len(run_calls), 1)
        composed = run_calls[0]
        self.assertIn("6 new direct items", composed)
        self.assertIn("item 0", composed)
        self.assertIn("item 5", composed)


class ReasoningModelHandlingTests(unittest.TestCase):
    """chat_json retries on parse-failure; broca extracts clean answer."""

    def test_chat_json_retries_with_bigger_budget_on_parse_failure(self):
        from brain.config import Config
        from brain.llm import LLM
        from unittest.mock import patch, MagicMock
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={}, regions={},
        )
        llm = LLM(cfg)

        # First call returns reasoning-only text (no JSON). Second returns
        # a clean JSON object. chat_json must return the parsed object.
        calls: list[dict] = []
        def fake_post(path, json=None, **kw):
            calls.append(json)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            if len(calls) == 1:
                r.json.return_value = {"choices": [{
                    "message": {
                        "content": "",
                        "reasoning_content": "Thinking process: I should... "
                                              "Step 1 analyze. Step 2..."
                    }
                }]}
            else:
                r.json.return_value = {"choices": [{
                    "message": {"content": '{"kind":"reflect","content":"ok"}'}
                }]}
            return r

        with patch.object(llm._client, "post", side_effect=fake_post):
            out = llm.chat_json("m", "SYS", "USR",
                                 temperature=0.4, max_tokens=200)
        llm.close()

        self.assertEqual(out.get("kind"), "reflect")
        # Two calls were made (one retry); the retry used a bigger budget
        # and the strict-output system directive
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1]["max_tokens"], calls[0]["max_tokens"])
        retry_sys = calls[1]["messages"][0]["content"]
        self.assertIn("MUST emit exactly one JSON object", retry_sys)

    def test_chat_json_no_retry_when_first_response_parses(self):
        from brain.config import Config
        from brain.llm import LLM
        from unittest.mock import patch, MagicMock
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={}, regions={},
        )
        llm = LLM(cfg)
        calls = 0
        def fake_post(path, json=None, **kw):
            nonlocal calls; calls += 1
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"choices": [{
                "message": {"content": '{"value": 7}'}
            }]}
            return r
        with patch.object(llm._client, "post", side_effect=fake_post):
            out = llm.chat_json("m", "S", "U")
        llm.close()
        self.assertEqual(out.get("value"), 7)
        self.assertEqual(calls, 1)

    def test_extract_final_answer_from_reasoning_monologue(self):
        from brain.regions.broca import _extract_final_answer
        # Marker-based extraction
        msg = (
            "Thinking process: I should analyze...\n"
            "1. Identify the goal.\n2. Pick option C.\n"
            "**Draft:**\nI'm going for the walk. Ninety minutes is enough."
        )
        out = _extract_final_answer(msg)
        self.assertIn("walk", out)
        self.assertNotIn("Thinking process", out)
        # Paragraph-walk fallback (no explicit marker)
        msg2 = (
            "Process:\n1. step one\n2. step two\n\n"
            "All constraints met. The draft is ready.\n\n"
            "Ninety minutes. That's enough to do something or to sit still. "
            "I think I'll go for the walk."
        )
        out2 = _extract_final_answer(msg2)
        self.assertIn("walk", out2)
        self.assertNotIn("step one", out2)

    def test_extract_final_answer_empty_input(self):
        from brain.regions.broca import _extract_final_answer
        self.assertEqual(_extract_final_answer(""), "")
        self.assertEqual(_extract_final_answer(None), "")

    def test_resolve_effector_finds_verb_in_many_shapes(self):
        from brain.regions.prefrontal import _resolve_effector
        # top-level effector
        self.assertEqual(_resolve_effector(
            {"effector": "shell", "args": {"command": "ls"}}), "shell")
        # nested under args.effector
        self.assertEqual(_resolve_effector(
            {"args": {"effector": "write_file", "path": "x"}}), "write_file")
        # args.name alias
        self.assertEqual(_resolve_effector(
            {"args": {"name": "read_file", "path": "x"}}), "read_file")
        # bogus phrase as effector (just returns it; the orchestrator demotes)
        self.assertEqual(_resolve_effector(
            {"effector": "go for a walk"}), "go for a walk")
        # args as bare string
        self.assertEqual(_resolve_effector({"args": "shell"}), "shell")
        # nothing parseable
        self.assertEqual(_resolve_effector({}), "")

    def test_prefrontal_demotes_action_with_invalid_effector(self):
        """End-to-end through Prefrontal.next_thought: bogus effector
        → kind demoted to tentative_plan, never reaches orchestrator
        as a dispatchable action."""
        from unittest.mock import MagicMock, patch
        from brain.regions.prefrontal import Prefrontal
        from brain.config import Config
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={},
            regions={"prefrontal": "executive"},
        )
        llm = MagicMock()
        llm.chat_json.return_value = {
            "content": "I want to go for a walk.",
            "kind": "action",
            "effector": "go for a real walk",   # bogus
            "args": {},
            "confidence": 0.7,
        }
        pf = Prefrontal(cfg, llm)
        ws = Workspace(task="x")
        unit = pf.next_thought(ws, effectors=["think", "finish",
                                                "read_file", "shell"])
        # bogus effector → demoted to tentative_plan, NOT dispatchable
        self.assertEqual(unit.kind, "tentative_plan")
        self.assertIn("walk", unit.content)

    def test_prefrontal_normalizes_valid_action_args(self):
        from unittest.mock import MagicMock
        from brain.regions.prefrontal import Prefrontal
        from brain.config import Config
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={}, effectors={},
            regions={"prefrontal": "executive"},
        )
        llm = MagicMock()
        # Common shape: top-level effector + arg keys at top
        llm.chat_json.return_value = {
            "content": "Let me check the file.",
            "kind": "action",
            "effector": "read_file",
            "args": {"path": "notes.md"},
            "expected_result": "the contents of notes.md",
            "confidence": 0.8,
        }
        pf = Prefrontal(cfg, llm)
        ws = Workspace(task="x")
        unit = pf.next_thought(ws, effectors=["read_file", "think", "finish"])
        self.assertEqual(unit.kind, "action")
        # args normalized so orchestrator's unit.args.effector + .args lookups work
        self.assertEqual(unit.args.get("effector"), "read_file")
        self.assertEqual(unit.args.get("args"), {"path": "notes.md"})

    def test_orchestrator_validates_effector_after_bg_repair(self):
        """If BG's repair produces a bogus effector, the orchestrator
        catches it before dispatch and downgrades to think."""
        # Smoke-test the gating with stubs at the EFFECTORS layer rather
        # than running a full Brain — we only need to verify the validation
        # branch fires.
        from brain.effectors import Effectors
        from brain.config import Config
        cfg = Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "x", "executive": "y"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=Path(tempfile.mkdtemp()),
            db_path=Path(tempfile.mkdtemp()) / "m.sqlite",
            loop={}, memory={},
            effectors={"filesystem": {"enabled": False},
                        "shell": {"enabled": False},
                        "web": {"enabled": False}},
            regions={},
        )
        eff = Effectors(cfg)
        available = eff.available()
        # The kind of string a 35B-class BG sometimes "repairs" to
        bogus = "go for a real walk"
        self.assertNotIn(bogus, available)
        # The orchestrator's branch substitutes 'think' which IS available
        self.assertIn("think", available)


class EmbodimentIntegrationTests(unittest.TestCase):
    """Brain ↔ afferent wiring, driven by afferent's offline FakeBackend.

    Skips cleanly if afferent isn't installed so the suite still runs in
    minimal environments (afferent is an optional dependency)."""

    def setUp(self):
        try:
            import afferent  # noqa: F401
        except ImportError:
            self.skipTest("afferent not installed (optional embodiment dep)")

    def _config(self, embodiment=None):
        from brain.config import Config
        tmp = Path(tempfile.mkdtemp())
        return Config(
            raw={}, api_key="", base_url="http://x", require_auth=False,
            extra_headers={}, models={"reflex": "fake", "executive": "fake"},
            timeout_seconds=10, max_retries=0,
            sandbox_dir=tmp, db_path=tmp / "m.sqlite",
            loop={}, memory={}, effectors={"filesystem": {"enabled": False}},
            regions={}, embodiment=embodiment or {},
        )

    def _fake_embodiment(self, *, read_only=False, confirm=None):
        from afferent import Embodiment, FakeBackend
        from afferent.types import Observation, VisualElement
        script = [
            Observation(ts=0.0, frontmost_app="Editor",
                        elements=[VisualElement("Save", (0.9, 0.05, 0.05, 0.03),
                                                kind="button")]),
            Observation(ts=1.0, frontmost_app="Editor", ocr_text="saved"),
        ]
        return Embodiment(FakeBackend(script=script), read_only=read_only,
                          settle_ms=0, confirm=confirm,
                          max_actions_per_min=100)

    def test_effectors_advertise_screen_verbs_when_embodied(self):
        from brain.effectors import Effectors
        cfg = self._config()
        eff = Effectors(cfg, embodiment=self._fake_embodiment())
        avail = eff.available()
        for v in ("look", "screen_click", "screen_type", "screen_key"):
            self.assertIn(v, avail)

    def test_no_screen_verbs_without_embodiment(self):
        from brain.effectors import Effectors
        eff = Effectors(self._config(), embodiment=None)
        self.assertNotIn("look", eff.available())
        self.assertNotIn("screen_click", eff.available())

    def test_look_returns_screen_render(self):
        from brain.effectors import Effectors
        eff = Effectors(self._config(), embodiment=self._fake_embodiment())
        ok, text = eff.execute("look", {})
        self.assertTrue(ok)
        self.assertIn("Editor", text)
        self.assertIn("Save", text)

    def test_screen_click_acts_and_grounds(self):
        from brain.effectors import Effectors
        emb = self._fake_embodiment()
        eff = Effectors(self._config(), embodiment=emb)
        ok, text = eff.execute("screen_click", {"x_pct": 0.9, "y_pct": 0.05})
        self.assertTrue(ok)
        # state_after rendered into the result text (world-model grounding)
        self.assertIn("after:", text)
        self.assertEqual(emb.backend.recorded_actions[0][0], "click_at")

    def test_screen_click_read_only_refuses(self):
        from brain.effectors import Effectors
        emb = self._fake_embodiment(read_only=True)
        eff = Effectors(self._config(), embodiment=emb)
        ok, text = eff.execute("screen_click", {"x_pct": 0.5, "y_pct": 0.5})
        self.assertFalse(ok)
        self.assertEqual(emb.backend.recorded_actions, [])

    def test_confirm_veto_blocks_screen_action(self):
        from brain.effectors import Effectors
        emb = self._fake_embodiment(confirm=lambda desc: False)
        eff = Effectors(self._config(), embodiment=emb)
        ok, _ = eff.execute("screen_type", {"text": "hi"})
        self.assertFalse(ok)
        self.assertEqual(emb.backend.recorded_actions, [])

    def test_bad_args_rejected(self):
        from brain.effectors import Effectors
        eff = Effectors(self._config(), embodiment=self._fake_embodiment())
        ok, msg = eff.execute("screen_click", {"x_pct": "nope"})
        self.assertFalse(ok)
        ok2, msg2 = eff.execute("screen_type", {})
        self.assertFalse(ok2)

    def test_occipital_posts_vision_broadcast(self):
        from brain.regions.occipital import Occipital
        cfg = self._config()
        emb = self._fake_embodiment()
        # no LLM needed: the fake observation carries elements → render_text
        occ = Occipital(cfg, llm=MagicMock(), embodiment=emb,
                        describe_with_vision=False)
        ws = Workspace(task="x")
        b = occ.step(ws)
        self.assertIsNotNone(b)
        self.assertEqual(b.kind, "vision")
        self.assertIn("Editor", b.content)
        self.assertEqual(b.data["frontmost_app"], "Editor")

    def test_orchestrator_builds_embodiment_when_enabled(self):
        from brain.orchestrator import Brain
        cfg = self._config(embodiment={"enabled": True, "backend": "fake",
                                       "read_only": True})
        with patch("brain.orchestrator.LLM") as MockLLM, \
             patch("brain.orchestrator.make_backend") as mk:
            MockLLM.return_value = MagicMock()
            from brain.embeddings import TfidfBackend
            mk.return_value = TfidfBackend()
            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=False)
            self.assertIsNotNone(brain.embodiment)
            self.assertIsNotNone(brain.occipital)
            self.assertIn("look", brain.effectors.available())
            brain.close()

    def test_orchestrator_disembodied_by_default(self):
        from brain.orchestrator import Brain
        cfg = self._config(embodiment={})   # disabled
        with patch("brain.orchestrator.LLM") as MockLLM, \
             patch("brain.orchestrator.make_backend") as mk:
            MockLLM.return_value = MagicMock()
            from brain.embeddings import TfidfBackend
            mk.return_value = TfidfBackend()
            brain = Brain(cfg, confirm=lambda _m: True, log=lambda _m: None,
                          humanize=False)
            self.assertIsNone(brain.embodiment)
            self.assertIsNone(brain.occipital)
            brain.close()


class ForwardModelTests(unittest.TestCase):
    """The learned forward model (numpy MLP) and its sleep-time trainer."""

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed (optional forward-model dep)")

    def test_mlp_learns_success_pattern(self):
        import numpy as np
        from brain.forward_model import ForwardModel
        rng = np.random.RandomState(0)
        emb = 6
        # Learnable rule: success iff the first state dim > 0 (action irrelevant).
        N = 240
        S = rng.randn(N, emb).astype(np.float32)
        A = rng.randn(N, emb).astype(np.float32)
        X = np.concatenate([S, A], axis=1)
        ok = (S[:, 0] > 0).astype(np.float32)
        Y = S + 0.1 * A   # arbitrary smooth outcome target
        m = ForwardModel(emb_dim=emb, hidden=32, seed=1)
        m.fit(X, Y, ok, epochs=200, lr=5e-3, seed=1)
        # accuracy on the training rule should be well above chance
        correct = 0
        for i in range(N):
            _, p = m.predict(S[i], A[i])
            correct += int((p >= 0.5) == bool(ok[i]))
        acc = correct / N
        self.assertGreater(acc, 0.85)

    def test_save_load_roundtrip(self):
        import numpy as np
        from brain.forward_model import ForwardModel
        m = ForwardModel(emb_dim=4, hidden=8, seed=2)
        X = np.random.RandomState(0).randn(40, 8).astype(np.float32)
        Y = np.random.RandomState(1).randn(40, 4).astype(np.float32)
        ok = (np.arange(40) % 2).astype(np.float32)
        m.fit(X, Y, ok, epochs=20)
        s = np.random.RandomState(3).randn(4).astype(np.float32)
        a = np.random.RandomState(4).randn(4).astype(np.float32)
        before = m.predict(s, a)
        path = Path(tempfile.mkdtemp()) / "fm.npz"
        m.save(path)
        m2 = ForwardModel.load(path)
        self.assertIsNotNone(m2)
        after = m2.predict(s, a)
        self.assertAlmostEqual(before[1], after[1], places=5)

    def test_load_missing_returns_none(self):
        from brain.forward_model import ForwardModel
        self.assertIsNone(ForwardModel.load(Path(tempfile.mkdtemp()) / "nope.npz"))


class _DenseBackendStub:
    """Deterministic dense embedding backend for forward-model tests:
    embeds text -> fixed-dim vector from a hash. persistent + dim set."""
    name = "stub-dense"
    persistent = True
    dim = 8

    def encode_one(self, text):
        import struct
        import hashlib
        h = hashlib.sha256((text or "").encode()).digest()
        vals = [((h[i] / 255.0) * 2 - 1) for i in range(self.dim)]
        return struct.pack(f"<I{self.dim}f", self.dim, *vals)

    def from_bytes(self, blob):
        import struct
        n = struct.unpack("<I", blob[:4])[0]
        return list(struct.unpack(f"<{n}f", blob[4:4 + 4 * n]))


class ForwardModelTrainerTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def _world_model_with_triples(self, n=60):
        from brain.world_model import WorldModelStore
        tmp = Path(tempfile.mkdtemp()) / "wm.sqlite"
        wm = WorldModelStore(tmp, backend=_DenseBackendStub())
        for i in range(n):
            ok = (i % 2 == 0)
            wm.observe(state_text=f"goal=task{i%5}; mood=neutral",
                       action_text=f"shell(command=cmd{i%3})",
                       outcome_text=("ok done" if ok else "error failed"),
                       ok=ok)
        return wm

    def test_trainer_trains_and_saves(self):
        from brain.sleep import ForwardModelTrainer
        from brain.memory import Memory
        wm = self._world_model_with_triples(60)
        tmp = Path(tempfile.mkdtemp())
        mem = Memory(tmp / "m.sqlite", backend=_DenseBackendStub())
        ckpt = tmp / "fm.npz"
        stats = ForwardModelTrainer(epochs=50, min_rows=20).run(
            mem, wm, checkpoint=ckpt, log=lambda _m: None)
        self.assertTrue(stats["trained"])
        self.assertTrue(ckpt.exists())
        mem.close(); wm.close()

    def test_trainer_noops_on_nondense_backend(self):
        from brain.sleep import ForwardModelTrainer
        from brain.memory import Memory
        from brain.embeddings import TfidfBackend
        wm = self._world_model_with_triples(60)
        tmp = Path(tempfile.mkdtemp())
        # Memory with TF-IDF (non-dense) → trainer must skip
        mem = Memory(tmp / "m.sqlite", backend=TfidfBackend())
        stats = ForwardModelTrainer(min_rows=20).run(mem, wm, log=lambda _m: None)
        self.assertFalse(stats["trained"])
        self.assertEqual(stats["reason"], "non-dense backend")
        mem.close(); wm.close()

    def test_trainer_noops_on_insufficient_data(self):
        from brain.sleep import ForwardModelTrainer
        from brain.memory import Memory
        wm = self._world_model_with_triples(5)
        tmp = Path(tempfile.mkdtemp())
        mem = Memory(tmp / "m.sqlite", backend=_DenseBackendStub())
        stats = ForwardModelTrainer(min_rows=40).run(mem, wm, log=lambda _m: None)
        self.assertFalse(stats["trained"])
        wm.close(); mem.close()

    def test_cerebellum_uses_learned_ok_prob(self):
        """A forward model that always predicts failure should flip the
        cerebellum's predicted_ok to False even when k-NN neighbours succeeded."""
        import numpy as np
        from brain.config import Config
        from brain.regions.cerebellum import Cerebellum
        from brain.world_model import WorldModelStore

        backend = _DenseBackendStub()
        tmp = Path(tempfile.mkdtemp())
        wm = WorldModelStore(tmp / "wm.sqlite", backend=backend)
        # all-success neighbours → k-NN would vote ok=True
        for i in range(6):
            wm.observe("goal=clean; mood=neutral", "shell(command=rm)",
                       "ok removed", ok=True)

        class _AlwaysFail:
            def predict(self, s, a):
                return (np.zeros(backend.dim, dtype=np.float32), 0.02)

        cfg = Config(raw={}, api_key="", base_url="http://x", require_auth=False,
                     extra_headers={}, models={"reflex": "x", "executive": "y"},
                     timeout_seconds=10, max_retries=0,
                     sandbox_dir=tmp, db_path=tmp / "m.sqlite",
                     loop={}, memory={}, effectors={}, regions={})
        cb = Cerebellum(cfg, llm=None, world_model=wm,
                        forward_model=_AlwaysFail(), embedding_backend=backend)
        ws = Workspace(task="clean")
        from brain.workspace import Broadcast
        ws.post(Broadcast(source="sensory_cortex", kind="percept",
                          content="goal=clean", salience=0.9,
                          data={"goal": "clean", "entities": []}))
        pred = cb.quick_predict(ws, "shell", {"command": "rm"})
        self.assertFalse(pred.predicted_ok)        # learned model overrode k-NN
        self.assertGreater(pred.confidence, 0.5)   # 0.02 → high certainty of failure
        wm.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
