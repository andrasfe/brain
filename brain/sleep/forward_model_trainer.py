"""ForwardModelTrainer (NREM) — train the learned forward model during sleep.

Pulls the accumulated `(state, action, outcome, ok)` triples from the
`WorldModelStore`, embeds each text via the (dense) embedding backend, and
fits the `ForwardModel` MLP to predict `(outcome_emb, success_prob)` from
`(state_emb, action_emb)`. Best-val weights are saved to a checkpoint next to
the memory DB, and — when a live cerebellum is passed — its in-memory model
is swapped to the freshly trained one.

Preconditions (else it no-ops, by design):
  - numpy installed (optional brain dep),
  - a DENSE embedding backend (fixed-dim, persistent — EmbeddingGemma /
    sentence-transformers). TF-IDF is sparse/variable-dim → not trainable,
  - at least `min_rows` triples accumulated.

Real ML during real sleep: this is where the brain's world model actually
*learns* (gradients), as opposed to merely accumulating k-NN rows while awake.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def default_checkpoint(db_path) -> Path:
    # Stem (no suffix); each backend appends its own (.npz / .mlx.safetensors).
    return Path(db_path).parent / "forward_model"


class ForwardModelTrainer:
    name = "forward_model_trainer"

    def __init__(self, hidden: int = 256, depth: int = 2, epochs: int = 200,
                 lr: float = 1e-3, min_rows: int = 40, window: int = 4000,
                 backend: str = "auto"):
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.min_rows = min_rows
        self.window = window
        self.backend = backend

    def run(self, memory, world_model, *, checkpoint: Optional[Path] = None,
            cerebellum=None, log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        try:
            import numpy as np  # noqa: F401
            from ..forward_model import make_forward_model
        except ImportError:
            log("  forward_model: numpy not installed — skipping")
            return {"trained": False, "reason": "numpy missing"}

        backend = getattr(memory, "backend", None)
        if backend is None or not getattr(backend, "persistent", False) \
                or int(getattr(backend, "dim", 0) or 0) <= 0:
            log("  forward_model: needs a dense embedding backend "
                "(EmbeddingGemma); skipping")
            return {"trained": False, "reason": "non-dense backend"}

        rows = world_model.conn.execute(
            "SELECT state_text, action_text, outcome_text, ok FROM world_model "
            "ORDER BY ts DESC LIMIT ?", (self.window,)
        ).fetchall()
        if len(rows) < self.min_rows:
            log(f"  forward_model: only {len(rows)} triples "
                f"(< {self.min_rows}); skipping")
            return {"trained": False, "reason": "insufficient data",
                    "rows": len(rows)}

        emb = _Embedder(backend)
        import numpy as np
        X, Y, ok = [], [], []
        for r in rows:
            s = emb(r["state_text"]); a = emb(r["action_text"]); o = emb(r["outcome_text"])
            if s is None or a is None or o is None:
                continue
            X.append(np.concatenate([s, a]))
            Y.append(o)
            ok.append(1.0 if int(r["ok"]) else 0.0)
        if len(X) < self.min_rows:
            log(f"  forward_model: {len(X)} embeddable rows (< {self.min_rows}); "
                "skipping")
            return {"trained": False, "reason": "insufficient embeddable data",
                    "rows": len(X)}

        emb_dim = len(Y[0])
        model = make_forward_model(emb_dim, backend=self.backend,
                                   hidden=self.hidden, depth=self.depth)
        stats = model.fit(np.asarray(X), np.asarray(Y), np.asarray(ok),
                          epochs=self.epochs, lr=self.lr, log=log)

        ckpt = Path(checkpoint) if checkpoint else default_checkpoint(memory_db_path(memory))
        model.save(ckpt)
        if cerebellum is not None:
            cerebellum.forward_model = model
        log(f"  forward_model: saved checkpoint -> {ckpt.name}")
        return {"trained": True, "checkpoint": str(ckpt), **stats}


def memory_db_path(memory):
    # Memory keeps its sqlite connection; derive the file path from it.
    try:
        row = memory.conn.execute("PRAGMA database_list").fetchone()
        return row["file"] if row and row["file"] else "forward_model"
    except Exception:
        return "forward_model"


class _Embedder:
    """Embed arbitrary text -> dense float vector via the backend's
    encode_one/from_bytes round-trip (works for dense persistent backends)."""

    def __init__(self, backend):
        self.backend = backend
        self._cache: dict = {}

    def __call__(self, text: str):
        text = text or ""
        if text in self._cache:
            return self._cache[text]
        try:
            blob = self.backend.encode_one(text)
            vec = self.backend.from_bytes(blob) if blob else None
        except Exception:
            vec = None
        import numpy as np
        out = np.asarray(vec, dtype=np.float32) if vec else None
        self._cache[text] = out
        return out


# ── CLI: standalone training pass ────────────────────────────────────────────
def main() -> int:
    import argparse
    import sys
    repo = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config
    from brain.llm import LLM
    from brain.embeddings import make_backend
    from brain.memory import Memory
    from brain.world_model import WorldModelStore

    ap = argparse.ArgumentParser(description="Train the learned forward model.")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--min-rows", type=int, default=40)
    args = ap.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    backend = make_backend(cfg, llm=llm)
    memory = Memory(cfg.db_path, backend=backend)
    wm = WorldModelStore(cfg.db_path, backend=backend)
    try:
        stats = ForwardModelTrainer(hidden=args.hidden, epochs=args.epochs,
                                    min_rows=args.min_rows).run(
            memory, wm, log=lambda m: print(m, flush=True))
    finally:
        memory.close(); wm.close(); llm.close()
    print(f"\nstats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
