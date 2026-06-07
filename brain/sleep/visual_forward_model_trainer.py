"""VisualForwardModelTrainer (NREM) — train the visual JEPA world model.

The action-conditioned VISUAL world model: a frozen DINOv2 encoder (the screen
embedding) + a learned predictor in that visual latent space. Given the screen
the brain saw and the action it took, predict the screen it will see next:

    f([screen_vis ; action_text_emb]) -> (next_screen_vis_hat, success_prob)

This is true JEPA — we predict in *representation* space, never pixels, and the
encoder stays frozen (pretrained DINOv2), so there's no representation collapse
to guard against. The predictor is the same robust MLX residual MLP the text
forward model uses, just with asymmetric input dims (visual state + text action)
and a visual output dim.

Reads the action-conditioned visual triples the orchestrator records during
embodied screen actions (`WorldModelStore.visual_triples()`), embeds each
action's text with the configured dense backend, fits, saves a SEPARATE
checkpoint (`forward_model_visual.*`), and refreshes the live cerebellum's
visual model when one is passed.

Preconditions (else it no-ops, by design):
  - numpy installed (or MLX),
  - a DENSE embedding backend (for the action embedding),
  - at least `min_rows` visual triples accumulated (only grow when the brain
    actually drives the screen — hands enabled).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .forward_model_trainer import _Embedder, memory_db_path


def default_checkpoint(db_path) -> Path:
    # Stem (no suffix); each backend appends its own (.npz / .mlx.safetensors).
    return Path(db_path).parent / "forward_model_visual"


class VisualForwardModelTrainer:
    name = "visual_forward_model_trainer"

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
            log("  visual_forward_model: numpy not installed — skipping")
            return {"trained": False, "reason": "numpy missing"}

        backend = getattr(memory, "backend", None)
        if backend is None or not getattr(backend, "persistent", False) \
                or int(getattr(backend, "dim", 0) or 0) <= 0:
            log("  visual_forward_model: needs a dense embedding backend "
                "(for action text); skipping")
            return {"trained": False, "reason": "non-dense backend"}

        try:
            triples = world_model.visual_triples(limit=self.window)
        except Exception as e:  # noqa: BLE001
            log(f"  visual_forward_model: visual_triples failed ({e}); skipping")
            return {"trained": False, "reason": "query failed"}
        if len(triples) < self.min_rows:
            log(f"  visual_forward_model: only {len(triples)} visual triples "
                f"(< {self.min_rows}); skipping")
            return {"trained": False, "reason": "insufficient data",
                    "rows": len(triples)}

        emb = _Embedder(backend)
        import numpy as np
        X, Y, ok = [], [], []
        vis_dim = None
        act_dim = None
        for t in triples:
            s = t.get("state_vis")
            o = t.get("outcome_vis")
            if not s or not o:
                continue
            a = emb(t.get("action_text") or "")
            if a is None:
                continue
            if vis_dim is None:
                vis_dim, act_dim = len(s), len(a)
            # Guard against ragged dims (mixed encoders across a DB's history).
            if len(s) != vis_dim or len(o) != vis_dim or len(a) != act_dim:
                continue
            X.append(np.concatenate([np.asarray(s, np.float32),
                                     np.asarray(a, np.float32)]))
            Y.append(np.asarray(o, np.float32))
            ok.append(1.0 if int(t.get("ok", 1)) else 0.0)
        if len(X) < self.min_rows:
            log(f"  visual_forward_model: {len(X)} usable triples "
                f"(< {self.min_rows}); skipping")
            return {"trained": False, "reason": "insufficient usable data",
                    "rows": len(X)}

        in_dim = vis_dim + act_dim
        model = make_forward_model(vis_dim, backend=self.backend,
                                   hidden=self.hidden, depth=self.depth,
                                   in_dim=in_dim)
        stats = model.fit(np.asarray(X), np.asarray(Y), np.asarray(ok),
                          epochs=self.epochs, lr=self.lr, log=log)

        ckpt = Path(checkpoint) if checkpoint else \
            default_checkpoint(memory_db_path(memory))
        model.save(ckpt)
        if cerebellum is not None:
            cerebellum.visual_forward_model = model
        log(f"  visual_forward_model: saved checkpoint -> {ckpt.name} "
            f"(vis={vis_dim}, act={act_dim})")
        return {"trained": True, "checkpoint": str(ckpt),
                "vis_dim": vis_dim, "act_dim": act_dim, **stats}


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

    ap = argparse.ArgumentParser(description="Train the visual forward model.")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min-rows", type=int, default=40)
    args = ap.parse_args()

    cfg = load_config()
    llm = LLM(cfg)
    backend = make_backend(cfg, llm=llm)
    memory = Memory(cfg.db_path, backend=backend)
    wm = WorldModelStore(cfg.db_path, backend=backend)
    try:
        stats = VisualForwardModelTrainer(
            hidden=args.hidden, epochs=args.epochs, min_rows=args.min_rows).run(
            memory, wm, log=lambda m: print(m, flush=True))
    finally:
        memory.close(); wm.close(); llm.close()
    print(f"\nstats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
