"""ScreenSequenceTrainer (NREM) — learn the dynamics of the user's day.

Trains the `ScreenSequenceModel` on the ordered `observation` stream: from
consecutive screen embeddings build (embedding[t] + time-of-day) → embedding[t+1]
pairs and fit a next-screen predictor. Runs during NREM sleep, checkpoints,
and refreshes the live occipital model.

No-ops without numpy/MLX, without a dense embedding backend, or below
`min_pairs`. Consecutive observations are only paired when close in time (no
pairing across a sleep/away gap).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ..memory import OBSERVATION


def default_checkpoint(db_path) -> Path:
    return Path(db_path).parent / "screen_model"


class ScreenSequenceTrainer:
    name = "screen_sequence_trainer"

    def __init__(self, hidden: int = 256, depth: int = 2, epochs: int = 200,
                 lr: float = 1e-3, min_pairs: int = 40, window: int = 4000,
                 max_gap_seconds: float = 1800.0, backend: str = "auto"):
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.min_pairs = min_pairs
        self.window = window
        self.max_gap_seconds = max_gap_seconds
        self.backend = backend

    def run(self, memory, *, checkpoint: Optional[Path] = None,
            occipital=None, log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        try:
            import numpy as np  # noqa: F401
            from ..screen_model import make_screen_model, time_features
        except ImportError:
            return {"trained": False, "reason": "numpy missing"}

        backend = getattr(memory, "backend", None)
        if backend is None or not getattr(backend, "persistent", False) \
                or int(getattr(backend, "dim", 0) or 0) <= 0:
            log("  screen_model: needs a dense embedding backend; skipping")
            return {"trained": False, "reason": "non-dense backend"}

        rows = memory.conn.execute(
            "SELECT content, ts, embedding FROM episodes WHERE mem_type=? "
            "ORDER BY ts ASC LIMIT ?", (OBSERVATION, self.window)
        ).fetchall()
        if len(rows) < self.min_pairs + 1:
            log(f"  screen_model: only {len(rows)} observations; skipping")
            return {"trained": False, "reason": "insufficient data",
                    "rows": len(rows)}

        from .forward_model_trainer import _Embedder
        emb = _Embedder(backend)

        def vec_of(row):
            # Prefer the stored embedding (DINOv2 image vector when visual
            # embedding is enabled) — that IS the screen-state representation.
            # Else embed the description text via the backend (text mode).
            blob = row["embedding"]
            if blob:
                v = backend.from_bytes(blob)
                if v:
                    return np.asarray(v, np.float32)
            return emb(row["content"])

        import numpy as np
        from collections import Counter
        # Collect candidate transitions first. Embeddings can live in DIFFERENT
        # spaces — DINOv2 image vectors for rows captured while visual_embed was
        # on, text vectors otherwise — so dimensions vary ACROSS pairs even
        # though va/vb match WITHIN a pair. Training one model needs a single
        # homogeneous space, so lock onto the dominant dim and drop the rest
        # (stacking ragged vectors would otherwise raise an inhomogeneous-shape
        # ValueError, and mixing image/text spaces is meaningless anyway).
        cands = []  # (va, vb, hour)
        for a, b in zip(rows, rows[1:]):
            if (b["ts"] - a["ts"]) > self.max_gap_seconds:
                continue  # don't pair across a gap (sleep/away)
            va = vec_of(a); vb = vec_of(b)
            if va is None or vb is None or len(va) != len(vb):
                continue
            hour = time.localtime(a["ts"]).tm_hour + time.localtime(a["ts"]).tm_min / 60.0
            cands.append((va, vb, hour))
        if not cands:
            log("  screen_model: no usable transitions; skipping")
            return {"trained": False, "reason": "insufficient pairs", "pairs": 0}
        target_dim = Counter(len(va) for va, _, _ in cands).most_common(1)[0][0]
        X, Y = [], []
        for va, vb, hour in cands:
            if len(va) != target_dim or len(vb) != target_dim:
                continue
            X.append(np.concatenate([va, np.asarray(time_features(hour), np.float32)]))
            Y.append(vb)
        dropped = len(cands) - len(X)
        if dropped:
            log(f"  screen_model: training on {len(X)} pairs at dim {target_dim}; "
                f"dropped {dropped} pair(s) in other embedding spaces")
        if len(X) < self.min_pairs:
            log(f"  screen_model: {len(X)} usable pairs (< {self.min_pairs}); skipping")
            return {"trained": False, "reason": "insufficient pairs", "pairs": len(X)}

        emb_dim = len(Y[0])
        model = make_screen_model(emb_dim, backend=self.backend,
                                  hidden=self.hidden, depth=self.depth)
        stats = model.fit(np.asarray(X), np.asarray(Y),
                          epochs=self.epochs, lr=self.lr, log=log)
        ckpt = Path(checkpoint) if checkpoint else default_checkpoint(_db_path(memory))
        model.save(ckpt)
        if occipital is not None:
            occipital.screen_model = model
        log(f"  screen_model: saved -> {ckpt.name}")
        return {"trained": True, "pairs": len(X), **stats}


def _db_path(memory):
    try:
        row = memory.conn.execute("PRAGMA database_list").fetchone()
        return row["file"] if row and row["file"] else "screen_model"
    except Exception:
        return "screen_model"
