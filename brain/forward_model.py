"""ForwardModel — a small learned world model (JEPA-lite).

A 2-layer MLP that maps a state+action embedding to:
  - `outcome_emb_hat`: the predicted next-state representation (regression, MSE)
  - `ok_logit`:        P(action succeeds in this state) (classification, BCE)

This is the gradient-trained complement to the k-NN `WorldModelStore`. The
k-NN substrate predicts by *retrieval* over a frozen embedding space; this
predicts by a *learned* function of the same embeddings — so it generalizes
between observed triples instead of only interpolating to the nearest one.
It's "JEPA-lite": the encoder (the embedding backend) stays frozen; we learn
the *predictor* in that latent space. Training the encoder itself (full
DINO/JEPA) is deliberately out of scope.

numpy-only (manual forward/backward + Adam). numpy is an optional dependency
of the brain — this module is imported lazily by the trainer and the
cerebellum, both of which degrade to k-NN when numpy or a checkpoint is
absent. Trains in milliseconds on the data volumes a single brain
accumulates; the architecture is intentionally tiny to resist overfitting
(handsneyes lesson: hidden=64 generalized; hidden=128 overfit on ~2k rows).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def _gelu_grad(x):
    # derivative of the tanh GELU approximation
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    dt = 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * dt


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ForwardModel:
    """Tiny MLP: [state_emb ; action_emb] -> (outcome_emb_hat, ok_prob)."""

    def __init__(self, emb_dim: int, hidden: int = 64, seed: int = 0):
        self.emb_dim = int(emb_dim)
        self.hidden = int(hidden)
        d_in = 2 * self.emb_dim
        rng = np.random.RandomState(seed)
        # He-ish init
        self.W1 = rng.randn(d_in, hidden).astype(np.float32) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.Wo = rng.randn(hidden, self.emb_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.bo = np.zeros(self.emb_dim, dtype=np.float32)
        self.Wk = rng.randn(hidden, 1).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.bk = np.zeros(1, dtype=np.float32)
        # input normalization (set during fit)
        self.mu = np.zeros(d_in, dtype=np.float32)
        self.sd = np.ones(d_in, dtype=np.float32)
        self.trained_rows = 0

    # ── forward ───────────────────────────────────────────────────────────────
    def _forward(self, X):
        Xn = (X - self.mu) / self.sd
        z1 = Xn @ self.W1 + self.b1
        h = _gelu(z1)
        out_emb = h @ self.Wo + self.bo
        ok_logit = (h @ self.Wk + self.bk).ravel()
        return Xn, z1, h, out_emb, ok_logit

    def predict(self, state_emb, action_emb) -> Tuple[np.ndarray, float]:
        """Return (predicted_outcome_emb, ok_prob) for one (state, action)."""
        X = np.concatenate([np.asarray(state_emb, dtype=np.float32),
                            np.asarray(action_emb, dtype=np.float32)])[None, :]
        _, _, _, out_emb, ok_logit = self._forward(X)
        return out_emb[0], float(_sigmoid(ok_logit)[0])

    # ── training (Adam, MSE + BCE) ──────────────────────────────────────────────
    def fit(self, X, Y_emb, y_ok, *, epochs: int = 300, lr: float = 3e-3,
            batch: int = 32, val_frac: float = 0.15, ok_weight: float = 1.0,
            seed: int = 0, log=None) -> dict:
        log = log or (lambda _m: None)
        X = np.asarray(X, dtype=np.float32)
        Y_emb = np.asarray(Y_emb, dtype=np.float32)
        y_ok = np.asarray(y_ok, dtype=np.float32).ravel()
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        # normalization from training data
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sd = (X.std(axis=0) + 1e-6).astype(np.float32)

        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac)) if n >= 8 else 0
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        params = [self.W1, self.b1, self.Wo, self.bo, self.Wk, self.bk]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1a, b2a, eps = 0.9, 0.999, 1e-8
        t = 0
        best_val = float("inf")
        best = None

        def loss_on(ix):
            if len(ix) == 0:
                return 0.0
            _, _, _, oe, kl = self._forward(X[ix])
            mse = float(np.mean((oe - Y_emb[ix]) ** 2))
            p = _sigmoid(kl)
            bce = float(np.mean(-(y_ok[ix] * np.log(p + 1e-7)
                                  + (1 - y_ok[ix]) * np.log(1 - p + 1e-7))))
            return mse + ok_weight * bce

        for ep in range(epochs):
            perm = rng.permutation(len(tr_idx))
            for s in range(0, len(tr_idx), batch):
                bi = tr_idx[perm[s:s + batch]]
                if len(bi) == 0:
                    continue
                Xb, Yb, kb = X[bi], Y_emb[bi], y_ok[bi]
                Xn, z1, h, oe, kl = self._forward(Xb)
                nb = Xb.shape[0]
                # grads — output (MSE) head
                d_oe = (2.0 / nb) * (oe - Yb)                  # (nb, emb)
                gWo = h.T @ d_oe
                gbo = d_oe.sum(axis=0)
                # ok (BCE) head
                p = _sigmoid(kl)
                d_kl = (ok_weight / nb) * (p - kb)             # (nb,)
                gWk = h.T @ d_kl[:, None]
                gbk = np.array([d_kl.sum()], dtype=np.float32)
                # backprop into hidden
                dh = d_oe @ self.Wo.T + d_kl[:, None] @ self.Wk.T
                dz1 = dh * _gelu_grad(z1)
                gW1 = Xn.T @ dz1
                gb1 = dz1.sum(axis=0)
                grads = [gW1, gb1, gWo, gbo, gWk, gbk]
                # Adam
                t += 1
                for i, (pgrad) in enumerate(grads):
                    m[i] = b1a * m[i] + (1 - b1a) * pgrad
                    v[i] = b2a * v[i] + (1 - b2a) * (pgrad ** 2)
                    mhat = m[i] / (1 - b1a ** t)
                    vhat = v[i] / (1 - b2a ** t)
                    params[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            if n_val:
                vl = loss_on(val_idx)
                if vl < best_val:
                    best_val = vl
                    best = [p.copy() for p in params]
        # restore best-val checkpoint
        if best is not None:
            for p, bp in zip(params, best):
                p[...] = bp
        self.trained_rows = int(n)
        stats = {"rows": n, "val_loss": round(best_val, 5) if n_val else None,
                 "train_loss": round(loss_on(tr_idx), 5)}
        log(f"forward_model: trained on {n} rows, val_loss={stats['val_loss']}")
        return stats

    # ── persistence ─────────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path),
                 W1=self.W1, b1=self.b1, Wo=self.Wo, bo=self.bo,
                 Wk=self.Wk, bk=self.bk, mu=self.mu, sd=self.sd)
        meta = {"emb_dim": self.emb_dim, "hidden": self.hidden,
                "trained_rows": self.trained_rows}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> Optional["ForwardModel"]:
        path = Path(path)
        npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
        meta_path = path.with_suffix(".json")
        if not npz.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            data = np.load(str(npz))
            m = cls(emb_dim=int(meta["emb_dim"]), hidden=int(meta["hidden"]))
            for k in ("W1", "b1", "Wo", "bo", "Wk", "bk", "mu", "sd"):
                setattr(m, k, data[k])
            m.trained_rows = int(meta.get("trained_rows", 0))
            return m
        except Exception:
            return None
