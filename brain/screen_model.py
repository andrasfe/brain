"""ScreenSequenceModel — a learned predictor of *what you'll do next*.

Self-supervised, JEPA-style: given the current screen embedding + time-of-day,
predict the **next** screen embedding. Trained during NREM sleep on the
ordered `observation` stream the ScreenObserver produces. This is the piece
that makes the brain learn the *dynamics* of your computer use (transitions,
routines) rather than just remembering snapshots.

Two uses at runtime (occipital region):
  - **novelty**: distance between what the model predicted last step and the
    screen that actually appeared → a "this is unexpected" signal.
  - **anticipation**: the predicted next-embedding, matched to the nearest
    past observation, gives "you usually do X next."

Encoder stays frozen (the screenshot embedding — EmbeddingGemma today, DINOv2
when enabled); only this predictor learns. MLX on Apple silicon; numpy
fallback via the same factory pattern as the action forward model. Optional
dependency, guarded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple


def time_features(hour: float) -> List[float]:
    """Cyclic encoding of time-of-day (so 23:59 ≈ 00:01)."""
    theta = 2.0 * math.pi * (float(hour) % 24.0) / 24.0
    return [math.sin(theta), math.cos(theta)]


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def make_screen_model(emb_dim: int, *, backend: str = "auto",
                      hidden: int = 256, depth: int = 2, dropout: float = 0.1,
                      seed: int = 0):
    backend = (backend or "auto").lower()
    if backend in ("auto", "mlx") and mlx_available():
        return _MLXScreenModel(emb_dim, hidden=hidden, depth=depth,
                               dropout=dropout, seed=seed)
    return _NumpyScreenModel(emb_dim, hidden=min(hidden, 128), seed=seed)


def load_screen_model(base_path: Path):
    base = Path(base_path)
    meta = base.with_suffix(".json")
    kind = None
    if meta.exists():
        try:
            kind = json.loads(meta.read_text()).get("kind")
        except Exception:
            kind = None
    if (kind == "mlx" or base.with_suffix(".mlx.safetensors").exists()) and mlx_available():
        try:
            return _MLXScreenModel.load(base)
        except Exception:
            pass
    return _NumpyScreenModel.load(base)


# ── numpy fallback ───────────────────────────────────────────────────────────
class _NumpyScreenModel:
    kind = "numpy"

    def __init__(self, emb_dim: int, hidden: int = 128, seed: int = 0):
        import numpy as np
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        d_in = emb_dim + 2  # embedding + 2 time features
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(d_in, hidden) * np.sqrt(2.0 / d_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.randn(hidden, emb_dim) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(emb_dim, dtype=np.float32)
        self.mu = np.zeros(d_in, dtype=np.float32)
        self.sd = np.ones(d_in, dtype=np.float32)
        self.trained_rows = 0

    def _fwd(self, X):
        import numpy as np
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            Xn = (X - self.mu) / self.sd
            h = np.maximum(0.0, Xn @ self.W1 + self.b1)
            return h @ self.W2 + self.b2

    def predict_next(self, emb, hour: float):
        import numpy as np
        x = np.concatenate([np.asarray(emb, np.float32),
                            np.asarray(time_features(hour), np.float32)])[None, :]
        return self._fwd(x)[0]

    def fit(self, X, Y, *, epochs=150, lr=1e-3, batch=64, seed=0, log=None):
        import numpy as np
        log = log or (lambda _m: None)
        X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
        n = X.shape[0]
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        rng = np.random.RandomState(seed)
        mW1 = mW2 = mb1 = mb2 = 0
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            for ep in range(epochs):
                perm = rng.permutation(n)
                for s in range(0, n, batch):
                    bi = perm[s:s + batch]
                    Xn = (X[bi] - self.mu) / self.sd
                    z = Xn @ self.W1 + self.b1
                    h = np.maximum(0.0, z)
                    out = h @ self.W2 + self.b2
                    d_out = (2.0 / len(bi)) * (out - Y[bi])
                    gW2 = h.T @ d_out; gb2 = d_out.sum(0)
                    dh = (d_out @ self.W2.T) * (z > 0)
                    gW1 = Xn.T @ dh; gb1 = dh.sum(0)
                    for p, g in ((self.W1, gW1), (self.b1, gb1),
                                 (self.W2, gW2), (self.b2, gb2)):
                        np.clip(g, -5, 5, out=g)
                        p -= lr * g
        self.trained_rows = n
        log(f"screen_model[numpy]: trained on {n} pairs")
        return {"rows": n, "backend": "numpy"}

    def save(self, path: Path):
        import numpy as np
        path = Path(path)
        np.savez(str(path.with_suffix(".npz")), W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2, mu=self.mu, sd=self.sd)
        path.with_suffix(".json").write_text(json.dumps(
            {"kind": "numpy", "emb_dim": self.emb_dim, "hidden": self.hidden,
             "trained_rows": self.trained_rows}))

    @classmethod
    def load(cls, path: Path):
        import numpy as np
        path = Path(path)
        npz, meta = path.with_suffix(".npz"), path.with_suffix(".json")
        if not npz.exists() or not meta.exists():
            return None
        try:
            m_ = json.loads(meta.read_text())
            m = cls(emb_dim=int(m_["emb_dim"]), hidden=int(m_["hidden"]))
            d = np.load(str(npz))
            for k in ("W1", "b1", "W2", "b2", "mu", "sd"):
                setattr(m, k, d[k])
            m.trained_rows = int(m_.get("trained_rows", 0))
            return m
        except Exception:
            return None


# ── MLX (default on Apple silicon) ───────────────────────────────────────────
class _MLXScreenModel:
    kind = "mlx"

    def __init__(self, emb_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, seed: int = 0):
        import mlx.core as mx
        import mlx.nn as nn
        from .forward_model_mlx import _ResidualBlock
        mx.random.seed(seed)
        self.emb_dim = int(emb_dim); self.hidden = int(hidden); self.depth = int(depth)
        d_in = emb_dim + 2

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(d_in, hidden)
                self.blocks = [_ResidualBlock(hidden, dropout) for _ in range(depth)]
                self.norm = nn.LayerNorm(hidden)
                self.out = nn.Linear(hidden, emb_dim)

            def __call__(self, x):
                h = self.proj(x)
                for b in self.blocks:
                    h = b(h)
                return self.out(self.norm(h))

        self.net = _Net()
        mx.eval(self.net.parameters())
        self.mu = mx.zeros((d_in,)); self.sd = mx.ones((d_in,))
        self.trained_rows = 0

    def predict_next(self, emb, hour: float):
        import mlx.core as mx
        import numpy as np
        vec = np.concatenate([np.asarray(emb, np.float32),
                              np.asarray(time_features(hour), np.float32)])
        self.net.eval()
        x = (mx.array(vec.reshape(1, -1)) - self.mu) / self.sd
        return np.asarray(self.net(x)[0].tolist(), dtype=np.float32)

    def fit(self, X, Y, *, epochs=200, lr=1e-3, batch=64, weight_decay=1e-4,
            val_frac=0.15, patience=25, seed=0, log=None):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from .forward_model_mlx import _clone_tree
        log = log or (lambda _m: None)
        X = mx.array(np.asarray(X, np.float32)); Y = mx.array(np.asarray(Y, np.float32))
        n = X.shape[0]
        self.mu = mx.mean(X, 0); self.sd = mx.std(X, 0) + 1e-6
        Xn = (X - self.mu) / self.sd
        rng = np.random.RandomState(seed); idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac)) if n >= 8 else 0
        vi = mx.array(idx[:n_val]) if n_val else None
        tr = idx[n_val:]
        opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        def loss_fn(net, xb, yb):
            return mx.mean((net(xb) - yb) ** 2)
        lg = nn.value_and_grad(self.net, loss_fn)

        def vloss():
            if not n_val:
                return None
            self.net.eval()
            return float(mx.mean((self.net(Xn[vi]) - Y[vi]) ** 2).item())

        best = float("inf"); best_w = None; bad = 0
        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(len(tr))
            for s in range(0, len(tr), batch):
                bi = mx.array(tr[perm[s:s + batch]])
                _, g = lg(self.net, Xn[bi], Y[bi])
                g = optim.clip_grad_norm(g, 5.0)[0]
                opt.update(self.net, g)
                mx.eval(self.net.parameters(), opt.state)
            vl = vloss()
            if vl is not None:
                if vl < best - 1e-6:
                    best, bad, best_w = vl, 0, _clone_tree(self.net.parameters())
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_w is not None:
            self.net.update(best_w); mx.eval(self.net.parameters())
        self.trained_rows = int(n)
        log(f"screen_model[mlx]: trained on {n} pairs ({self.depth}x{self.hidden}), "
            f"val={round(best,5) if n_val else None}")
        return {"rows": int(n), "val_loss": round(best, 5) if n_val else None,
                "backend": "mlx"}

    def save(self, path: Path):
        import mlx.core as mx
        from .forward_model_mlx import _flatten
        path = Path(path)
        flat = dict(_flatten(self.net.parameters()))
        flat["__mu__"] = self.mu; flat["__sd__"] = self.sd
        mx.save_safetensors(str(path.with_suffix(".mlx.safetensors")), flat)
        path.with_suffix(".json").write_text(json.dumps(
            {"kind": "mlx", "emb_dim": self.emb_dim, "hidden": self.hidden,
             "depth": self.depth, "trained_rows": self.trained_rows}))

    @classmethod
    def load(cls, path: Path):
        import mlx.core as mx
        from .forward_model_mlx import _unflatten
        path = Path(path)
        meta, w = path.with_suffix(".json"), path.with_suffix(".mlx.safetensors")
        if not meta.exists() or not w.exists():
            return None
        try:
            m_ = json.loads(meta.read_text())
            m = cls(emb_dim=int(m_["emb_dim"]), hidden=int(m_["hidden"]),
                    depth=int(m_["depth"]))
            flat = mx.load(str(w))
            m.mu = flat.pop("__mu__"); m.sd = flat.pop("__sd__")
            m.net.update(_unflatten(flat)); mx.eval(m.net.parameters())
            m.trained_rows = int(m_.get("trained_rows", 0))
            return m
        except Exception:
            return None


def cosine_distance(a, b) -> float:
    import numpy as np
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))
