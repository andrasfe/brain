"""Face identity — recognize the individuals who use this machine, then
attribute screen/app activity to whoever was at the webcam.

This is the identity layer on top of the wellness shot. The existing wellness
path takes one webcam still while you're present, reads your visible state, and
DROPS the pixels. Here we add ONE more step on that same still, before the drop:
compute a local **face embedding** and record a *sighting*. Embeddings only —
the photo is never stored, same privacy spine as everything else.

Unsupervised by design: there is no enrollment and no names. Sightings are
clustered into individuals (`Individual #1`, `#2`, …) purely by face-embedding
similarity; you can label a cluster later if you want. Identity recognition of
*who* (vs the wellness path, which deliberately only reports presence/state).

Two-phase, matching the brain's wake/sleep split:
  - WAKE (present): `record_sighting()` does a cheap online cluster assignment
    so a sighting is attributed immediately. Runs in the worker that already
    handles the wellness shot.
  - NREM (away / screensaver on): `recluster()` recomputes stable clusters from
    ALL sightings (removing the order-dependence of online assignment) and
    `build_profiles()` joins sightings to screen observations on timestamp to
    build each individual's app-usage profile. This is the "train while the
    screensaver's on" pass — heavy work in the idle window, none at capture.

Storage stays SQLite (the operational store): three tables in the memory db —
`face_sighting`, `face_identity`, `face_profile`. The embedder is pluggable
behind `FaceEmbedder`; InsightFace (local ONNX) is the default backend and the
whole thing no-ops cleanly when the model/deps are absent or `face.enabled` is
false.

PRIVACY: local-only (InsightFace runs on-device; no pixels leave the machine),
the photo is dropped by the caller right after embedding, committed config is
OFF, and nothing here ever names a person unless you label a cluster yourself.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .embeddings import _pack_floats, _unpack_floats

# apps that are brain bookkeeping, not human usage (kept out of profiles)
_NON_USER_APPS = {"wellness", "survey", "unknown", ""}


# ── math (pure python; the store never needs numpy) ──────────────────────────
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return s / (na * nb)


def _mean(vectors: list[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in acc]


def _running_mean(centroid: Sequence[float], n: int,
                  emb: Sequence[float]) -> list[float]:
    return [(c * n + e) / (n + 1) for c, e in zip(centroid, emb)]


# ── embedder backends (pluggable) ────────────────────────────────────────────
class FaceEmbedder:
    """Return an L2-ish face embedding for the most prominent face in an image,
    or None when no usable face is found. Implementations stay local."""

    def embed(self, image_path: str) -> Optional[list[float]]:  # pragma: no cover
        raise NotImplementedError


class InsightFaceEmbedder(FaceEmbedder):
    """Local InsightFace (ArcFace) embeddings via onnxruntime. `buffalo_l`
    gives 512-d normed embeddings; same-person cosine ~0.5+, different <~0.3.
    Heavy model loaded once and reused. Optional dep — guarded by the factory."""

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640,
                 min_det_score: float = 0.5):
        import numpy as np  # noqa: F401  (insightface needs it)
        from insightface.app import FaceAnalysis
        self._np = np
        self.min_det_score = float(min_det_score)
        self._app = FaceAnalysis(name=model_name,
                                 allowed_modules=["detection", "recognition"])
        # ctx_id=-1 forces CPU; onnxruntime picks CoreML/CPU providers itself.
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def embed(self, image_path: str) -> Optional[list[float]]:
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            faces = self._app.get(img)
        except Exception:
            return None
        if not faces:
            return None
        # most prominent face = largest box with acceptable detector score
        faces = [f for f in faces if float(getattr(f, "det_score", 1.0))
                 >= self.min_det_score]
        if not faces:
            return None

        def _area(f):
            x1, y1, x2, y2 = f.bbox
            return (x2 - x1) * (y2 - y1)

        face = max(faces, key=_area)
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        return [float(x) for x in emb]


_EMBEDDER_CACHE: dict[str, Optional[FaceEmbedder]] = {}


def make_face_embedder(cfg) -> Optional[FaceEmbedder]:
    """Build (and cache) the configured face embedder, or None when face
    identity is disabled or the backend can't load. Never raises."""
    fc = (getattr(cfg, "raw", {}) or {}).get("face") or {}
    if not fc.get("enabled"):
        return None
    backend = str(fc.get("backend", "insightface")).lower()
    ck = f"{backend}:{fc.get('model', 'buffalo_l')}"
    if ck in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[ck]
    emb: Optional[FaceEmbedder] = None
    if backend == "insightface":
        try:
            emb = InsightFaceEmbedder(
                model_name=str(fc.get("model", "buffalo_l")),
                det_size=int(fc.get("det_size", 640)),
                min_det_score=float(fc.get("min_det_score", 0.5)))
        except Exception:
            emb = None
    _EMBEDDER_CACHE[ck] = emb
    return emb


# ── identity store (SQLite — the operational store) ──────────────────────────
class FaceIdentityStore:
    """Sightings, clusters, and per-individual app profiles. Opens its own
    connection to the memory db (WAL → safe alongside other readers/writers).
    All logic is pure-python so it's offline-testable with a fake embedder."""

    def __init__(self, db_path, *, sim_threshold: float = 0.42,
                 merge_threshold: float = 0.5, window_seconds: float = 120.0):
        self.db_path = str(db_path)
        self.sim_threshold = float(sim_threshold)
        self.merge_threshold = float(merge_threshold)
        self.window_seconds = float(window_seconds)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS face_identity (
                id           INTEGER PRIMARY KEY,
                label        TEXT,
                centroid     BLOB NOT NULL,
                n            INTEGER NOT NULL,
                created_ts   REAL NOT NULL,
                last_seen_ts REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_sighting (
                id          INTEGER PRIMARY KEY,
                ts          REAL NOT NULL,
                embedding   BLOB NOT NULL,
                identity_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_face_sighting_ts ON face_sighting(ts);
            CREATE TABLE IF NOT EXISTS face_profile (
                identity_id  INTEGER PRIMARY KEY,
                apps_json    TEXT,
                n_sightings  INTEGER NOT NULL,
                first_seen   REAL,
                last_seen    REAL,
                summary      TEXT,
                updated_ts   REAL NOT NULL
            );
            """)
        self.conn.commit()

    # ── online assignment (WAKE) ─────────────────────────────────────────────
    def _load_identities(self) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute(
                "SELECT id, label, centroid, n FROM face_identity"):
            out.append({"id": r["id"], "label": r["label"],
                        "centroid": _unpack_floats(r["centroid"]), "n": r["n"]})
        return out

    def record_sighting(self, ts: float, emb: Sequence[float]) -> int:
        """Attribute one face embedding to an individual (cheap, online). Joins
        the nearest cluster above `sim_threshold`, else opens a new one. Returns
        the identity id."""
        emb = [float(x) for x in emb]
        ids = self._load_identities()
        best, best_sim = None, -1.0
        for ident in ids:
            sim = cosine(emb, ident["centroid"])
            if sim > best_sim:
                best, best_sim = ident, sim
        if best is not None and best_sim >= self.sim_threshold:
            new_c = _running_mean(best["centroid"], best["n"], emb)
            self.conn.execute(
                "UPDATE face_identity SET centroid=?, n=n+1, last_seen_ts=? "
                "WHERE id=?", (_pack_floats(new_c), ts, best["id"]))
            iid = int(best["id"])
        else:
            cur = self.conn.execute(
                "INSERT INTO face_identity (label, centroid, n, created_ts, "
                "last_seen_ts) VALUES (NULL,?,?,?,?)",
                (_pack_floats(emb), 1, ts, ts))
            iid = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO face_sighting (ts, embedding, identity_id) "
            "VALUES (?,?,?)", (ts, _pack_floats(emb), iid))
        self.conn.commit()
        return iid

    # ── full reclustering (NREM) ─────────────────────────────────────────────
    def recluster(self) -> dict[str, Any]:
        """Recompute stable clusters from ALL sightings, removing the order
        dependence of online assignment. Greedy streaming assignment, then a
        merge pass on near-duplicate centroids. Inherits any human label from
        the closest prior identity. Reassigns every sighting. Returns stats."""
        rows = list(self.conn.execute(
            "SELECT id, embedding FROM face_sighting ORDER BY ts"))
        if not rows:
            return {"identities": 0, "sightings": 0, "merges": 0}
        sightings = [(r["id"], _unpack_floats(r["embedding"])) for r in rows]

        # snapshot old labeled centroids so labels survive the rebuild
        old_labeled = [
            (_unpack_floats(r["centroid"]), r["label"])
            for r in self.conn.execute(
                "SELECT centroid, label FROM face_identity "
                "WHERE label IS NOT NULL")]

        # greedy streaming clusters
        clusters: list[dict[str, Any]] = []  # {members:[sid], centroid:[...]}
        for sid, emb in sightings:
            best, best_sim = None, -1.0
            for c in clusters:
                sim = cosine(emb, c["centroid"])
                if sim > best_sim:
                    best, best_sim = c, sim
            if best is not None and best_sim >= self.sim_threshold:
                best["members"].append(sid)
                best["embs"].append(emb)
                best["centroid"] = _mean(best["embs"])
            else:
                clusters.append({"members": [sid], "embs": [emb],
                                 "centroid": list(emb)})

        # merge pass: collapse clusters whose centroids are near-identical
        merges = 0
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if cosine(clusters[i]["centroid"],
                              clusters[j]["centroid"]) >= self.merge_threshold:
                        clusters[i]["members"] += clusters[j]["members"]
                        clusters[i]["embs"] += clusters[j]["embs"]
                        clusters[i]["centroid"] = _mean(clusters[i]["embs"])
                        clusters.pop(j)
                        merges += 1
                        merged = True
                        break
                if merged:
                    break

        # rebuild identity table
        now = time.time()
        self.conn.execute("DELETE FROM face_identity")
        for c in clusters:
            centroid = c["centroid"]
            label = None
            best_sim = -1.0
            for old_c, old_label in old_labeled:
                sim = cosine(centroid, old_c)
                if sim > best_sim and sim >= self.merge_threshold:
                    best_sim, label = sim, old_label
            cur = self.conn.execute(
                "INSERT INTO face_identity (label, centroid, n, created_ts, "
                "last_seen_ts) VALUES (?,?,?,?,?)",
                (label, _pack_floats(centroid), len(c["members"]), now, now))
            iid = int(cur.lastrowid)
            self.conn.executemany(
                "UPDATE face_sighting SET identity_id=? WHERE id=?",
                [(iid, sid) for sid in c["members"]])
        self.conn.commit()
        return {"identities": len(clusters), "sightings": len(sightings),
                "merges": merges}

    # ── association: who used which apps (NREM) ──────────────────────────────
    def build_profiles(self) -> dict[str, Any]:
        """Attribute active screen observations to the individual whose sighting
        was nearest in time (within `window_seconds`), then aggregate per
        individual into an app-usage profile. Returns stats."""
        import bisect
        from collections import defaultdict

        srows = list(self.conn.execute(
            "SELECT ts, identity_id FROM face_sighting "
            "WHERE identity_id IS NOT NULL ORDER BY ts"))
        if not srows:
            return {"profiles": 0, "attributed": 0}
        s_ts = [r["ts"] for r in srows]
        s_id = [r["identity_id"] for r in srows]

        apps: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        attributed = 0
        for row in self.conn.execute(
                "SELECT ts, tags FROM episodes "
                "WHERE mem_type='observation' AND tags IS NOT NULL"):
            tags = row["tags"] or ""
            app = None
            active = False
            for t in tags.split(","):
                if t.startswith("app:"):
                    app = t[4:]
                elif t == "agency:active":
                    active = True
            if not active or not app or app in _NON_USER_APPS:
                continue
            t = row["ts"]
            i = bisect.bisect_left(s_ts, t)
            best_iid, best_d = None, self.window_seconds + 1.0
            for k in (i - 1, i):
                if 0 <= k < len(s_ts):
                    d = abs(s_ts[k] - t)
                    if d < best_d:
                        best_d, best_iid = d, s_id[k]
            if best_iid is not None and best_d <= self.window_seconds:
                apps[best_iid][app] += 1
                attributed += 1

        # sighting counts + spans per identity
        meta: dict[int, dict[str, float]] = defaultdict(
            lambda: {"n": 0, "first": 0.0, "last": 0.0})
        for ts, iid in zip(s_ts, s_id):
            m = meta[iid]
            m["n"] += 1
            m["first"] = ts if m["first"] == 0.0 else min(m["first"], ts)
            m["last"] = max(m["last"], ts)

        now = time.time()
        self.conn.execute("DELETE FROM face_profile")
        for iid, m in meta.items():
            app_counts = dict(sorted(apps.get(iid, {}).items(),
                                     key=lambda kv: -kv[1]))
            top = list(app_counts.items())[:3]
            label = self._label_for(iid)
            summary = (f"{label} — {int(m['n'])} sighting(s); "
                       + (", ".join(f"{a} ({c})" for a, c in top)
                          if top else "no app activity attributed yet"))
            self.conn.execute(
                "INSERT INTO face_profile (identity_id, apps_json, n_sightings, "
                "first_seen, last_seen, summary, updated_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (iid, json.dumps(app_counts), int(m["n"]), m["first"],
                 m["last"], summary, now))
        self.conn.commit()
        return {"profiles": len(meta), "attributed": attributed}

    def _label_for(self, identity_id: int) -> str:
        r = self.conn.execute(
            "SELECT label FROM face_identity WHERE id=?", (identity_id,)).fetchone()
        if r and r["label"]:
            return str(r["label"])
        return f"Individual #{identity_id}"

    # ── read side (CLI / UI / recall) ────────────────────────────────────────
    def label_identity(self, identity_id: int, name: str) -> bool:
        cur = self.conn.execute(
            "UPDATE face_identity SET label=? WHERE id=?",
            (name.strip()[:60], identity_id))
        self.conn.commit()
        return cur.rowcount > 0

    def profiles(self) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute(
                "SELECT p.identity_id, p.apps_json, p.n_sightings, p.first_seen, "
                "p.last_seen, p.summary, i.label "
                "FROM face_profile p JOIN face_identity i ON i.id=p.identity_id "
                "ORDER BY p.n_sightings DESC"):
            out.append({
                "identity_id": r["identity_id"],
                "label": r["label"] or f"Individual #{r['identity_id']}",
                "apps": json.loads(r["apps_json"] or "{}"),
                "n_sightings": r["n_sightings"],
                "first_seen": r["first_seen"], "last_seen": r["last_seen"],
                "summary": r["summary"]})
        return out

    def count_sightings(self) -> int:
        r = self.conn.execute("SELECT COUNT(*) AS c FROM face_sighting").fetchone()
        return int(r["c"]) if r else 0

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# ── CLI: who used this machine, and what did each person do? ─────────────────
def main() -> int:
    import argparse
    import sys
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config

    ap = argparse.ArgumentParser(description="Face identity report / labeling.")
    ap.add_argument("--label", nargs=2, metavar=("ID", "NAME"),
                    help="assign a name to Individual #ID")
    ap.add_argument("--recluster", action="store_true",
                    help="force a recluster + profile rebuild now")
    args = ap.parse_args()

    cfg = load_config()
    fc = (cfg.raw or {}).get("face") or {}
    store = FaceIdentityStore(
        cfg.db_path,
        sim_threshold=float(fc.get("sim_threshold", 0.42)),
        merge_threshold=float(fc.get("merge_threshold", 0.5)),
        window_seconds=float(fc.get("window_seconds", 120)))
    try:
        if args.label:
            ok = store.label_identity(int(args.label[0]), args.label[1])
            print("labeled." if ok else "no such identity.")
            return 0 if ok else 1
        if args.recluster:
            rc = store.recluster()
            pr = store.build_profiles()
            print(f"reclustered: {rc}; profiles: {pr}")
        profs = store.profiles()
        if not profs:
            print(f"No individuals yet ({store.count_sightings()} sightings). "
                  "Enable face.enabled + let the daemon run an NREM bout.")
            return 0
        print(f"👤 {len(profs)} individual(s), "
              f"{store.count_sightings()} sighting(s):\n")
        for p in profs:
            print(f"  {p['label']}  ({p['n_sightings']} sightings)")
            if p["apps"]:
                top = list(p["apps"].items())[:6]
                print("    apps: " + ", ".join(f"{a} ({c})" for a, c in top))
            else:
                print("    apps: (none attributed yet)")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
