"""MLXForwardModel — a robust, GPU-trained world model (Apple MLX).

The gradient-trained forward model, upgraded from the hand-rolled numpy MLP to
a proper residual network trained on the Mac's GPU via MLX (Apple silicon,
unified memory). Same contract as the numpy fallback so the cerebellum /
trainer don't care which is loaded:

    predict(state_emb, action_emb) -> (predicted_outcome_emb, success_prob)

Architecture: input standardization → projection → N pre-norm residual MLP
blocks (LayerNorm → GELU → Linear → Dropout, gated residual) → two heads
(outcome-embedding regression, success-probability classification). Training:
AdamW + weight decay, cosine LR with warmup, dropout, gradient clipping,
best-val checkpointing with early-stopping patience. This scales to real
embedding dims (EmbeddingGemma 768 → 1536 concat) and sizeable hidden widths
without the numerical fragility of the toy.

MLX is an optional dependency (Apple-silicon only). `make_forward_model` /
`load_forward_model` in `brain/forward_model.py` fall back to the numpy model
when MLX is unavailable, so nothing here is a hard requirement.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def __call__(self, x):
        h = self.norm(x)
        h = nn.gelu(self.fc1(h))
        h = self.drop(self.fc2(h))
        return x + h


class _Net(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: int, depth: int,
                 dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = [_ResidualBlock(hidden, dropout) for _ in range(depth)]
        self.head_norm = nn.LayerNorm(hidden)
        self.out_head = nn.Linear(hidden, emb_dim)   # predicted outcome embedding
        self.ok_head = nn.Linear(hidden, 1)          # success logit

    def __call__(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        h = self.head_norm(h)
        return self.out_head(h), self.ok_head(h)


class MLXForwardModel:
    kind = "mlx"

    def __init__(self, emb_dim: int, hidden: int = 256, depth: int = 2,
                 dropout: float = 0.1, seed: int = 0):
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.dropout = float(dropout)
        mx.random.seed(seed)
        self.net = _Net(2 * self.emb_dim, self.emb_dim, hidden, depth, dropout)
        mx.eval(self.net.parameters())
        self.mu = mx.zeros((2 * self.emb_dim,))
        self.sd = mx.ones((2 * self.emb_dim,))
        self.trained_rows = 0

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, state_emb, action_emb) -> Tuple[list, float]:
        import numpy as _np
        vec = _np.concatenate([_np.asarray(state_emb, dtype=_np.float32),
                              _np.asarray(action_emb, dtype=_np.float32)])
        x = mx.array(vec.reshape(1, -1))
        xn = (x - self.mu) / self.sd
        self.net.eval()
        out, logit = self.net(xn)
        prob = float(mx.sigmoid(logit)[0, 0].item())
        return (out[0].tolist(), prob)

    # ── training ──────────────────────────────────────────────────────────────
    def fit(self, X, Y_emb, y_ok, *, epochs: int = 200, lr: float = 1e-3,
            batch: int = 64, val_frac: float = 0.15, weight_decay: float = 1e-4,
            warmup: int = 5, patience: int = 25, ok_weight: float = 1.0,
            seed: int = 0, log=None) -> dict:
        log = log or (lambda _m: None)
        X = mx.array(X, dtype=mx.float32)
        Y = mx.array(Y_emb, dtype=mx.float32)
        ok = mx.array(y_ok, dtype=mx.float32).reshape(-1)
        n = X.shape[0]

        self.mu = mx.mean(X, axis=0)
        self.sd = mx.std(X, axis=0) + 1e-6
        Xn = (X - self.mu) / self.sd

        import numpy as _np
        rng = _np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac)) if n >= 8 else 0
        val_idx = mx.array(idx[:n_val]) if n_val else None
        tr_idx = idx[n_val:]

        steps_per_epoch = max(1, len(tr_idx) // batch)
        total_steps = epochs * steps_per_epoch
        sched = optim.cosine_decay(lr, max(1, total_steps - warmup * steps_per_epoch))
        opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        def loss_fn(net, xb, yb, kb):
            out, logit = net(xb)
            mse = mx.mean((out - yb) ** 2)
            p = mx.sigmoid(logit).reshape(-1)
            bce = mx.mean(-(kb * mx.log(p + 1e-7) + (1 - kb) * mx.log(1 - p + 1e-7)))
            return mse + ok_weight * bce

        loss_and_grad = nn.value_and_grad(self.net, loss_fn)

        def val_loss():
            if not n_val:
                return None
            self.net.eval()
            out, logit = self.net(Xn[val_idx])
            mse = mx.mean((out - Y[val_idx]) ** 2)
            p = mx.sigmoid(logit).reshape(-1)
            kb = ok[val_idx]
            bce = mx.mean(-(kb * mx.log(p + 1e-7) + (1 - kb) * mx.log(1 - p + 1e-7)))
            return float((mse + ok_weight * bce).item())

        best = float("inf")
        best_w = None
        bad = 0
        gstep = 0
        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(len(tr_idx))
            for s in range(0, len(tr_idx), batch):
                bi = tr_idx[perm[s:s + batch]]
                if len(bi) == 0:
                    continue
                mbi = mx.array(bi)
                # warmup then cosine
                if gstep < warmup * steps_per_epoch:
                    opt.learning_rate = lr * (gstep + 1) / max(1, warmup * steps_per_epoch)
                else:
                    opt.learning_rate = sched(gstep - warmup * steps_per_epoch)
                loss, grads = loss_and_grad(self.net, Xn[mbi], Y[mbi], ok[mbi])
                grads = optim.clip_grad_norm(grads, 5.0)[0]
                opt.update(self.net, grads)
                mx.eval(self.net.parameters(), opt.state)
                gstep += 1
            vl = val_loss()
            if vl is not None:
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_w = _clone_tree(self.net.parameters())
                else:
                    bad += 1
                    if bad >= patience:
                        log(f"forward_model[mlx]: early stop @epoch {ep} (val={best:.5f})")
                        break
        if best_w is not None:
            self.net.update(best_w)
            mx.eval(self.net.parameters())
        self.trained_rows = int(n)
        stats = {"rows": int(n), "val_loss": round(best, 5) if n_val else None,
                 "backend": "mlx"}
        log(f"forward_model[mlx]: trained on {n} rows ({self.depth}x{self.hidden}), "
            f"val_loss={stats['val_loss']}")
        return stats

    # ── persistence ─────────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights_path = path.with_suffix(".mlx.safetensors")
        # Flatten params for save_weights
        flat = dict(_flatten(self.net.parameters()))
        flat["__mu__"] = self.mu
        flat["__sd__"] = self.sd
        mx.save_safetensors(str(weights_path), flat)
        meta = {"kind": "mlx", "emb_dim": self.emb_dim, "hidden": self.hidden,
                "depth": self.depth, "dropout": self.dropout,
                "trained_rows": self.trained_rows}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> Optional["MLXForwardModel"]:
        path = Path(path)
        meta_p = path.with_suffix(".json")
        weights_p = path.with_suffix(".mlx.safetensors")
        if not meta_p.exists() or not weights_p.exists():
            return None
        try:
            meta = json.loads(meta_p.read_text())
            m = cls(emb_dim=int(meta["emb_dim"]), hidden=int(meta["hidden"]),
                    depth=int(meta["depth"]), dropout=float(meta.get("dropout", 0.1)))
            flat = mx.load(str(weights_p))
            m.mu = flat.pop("__mu__")
            m.sd = flat.pop("__sd__")
            m.net.update(_unflatten(flat))
            mx.eval(m.net.parameters())
            m.trained_rows = int(meta.get("trained_rows", 0))
            return m
        except Exception:
            return None


# ── param-tree helpers (MLX parameters() is a nested dict/list tree) ─────────
def _flatten(tree, prefix=""):
    items = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            items += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            items += _flatten(v, f"{prefix}.{i}")
    else:
        items.append((prefix, tree))
    return items


def _unflatten(flat: dict):
    root: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = root
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            nxt_is_idx = (not last) and parts[i + 1].isdigit()
            if p.isdigit():
                p = int(p)
            if last:
                _tset(cur, p, val)
            else:
                child = _tget(cur, p)
                if child is None:
                    child = [] if nxt_is_idx else {}
                    _tset(cur, p, child)
                cur = child
    return root


def _tget(container, key):
    if isinstance(container, list):
        return container[key] if isinstance(key, int) and key < len(container) else None
    return container.get(key)


def _tset(container, key, val):
    if isinstance(container, list):
        while len(container) <= key:
            container.append(None)
        container[key] = val
    else:
        container[key] = val


def _clone_tree(tree):
    if isinstance(tree, dict):
        return {k: _clone_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_clone_tree(v) for v in tree]
    return mx.array(tree)
