"""VisualEmbedder — DINOv2 image embeddings for screen state (local-only).

A frozen, pretrained DINOv2 encoder turns a screenshot directly into a dense
vector — a far better *visual* state representation than embedding a one-line
text description. Used as the screen-observation embedding when enabled, so
the screen-sequence model learns over real visual features.

PRIVACY: runs entirely locally via torch (no uploads). The image is read from
disk, embedded, and the pixels are dropped as usual by the observer.

Optional dependency: needs `torch` + `torchvision` (and a one-time DINOv2
weight download via torch.hub). Everything is guarded — `available()` is False
when torch/model aren't present, and callers fall back to the text-embedding
path. Mac uses the MPS device when available, else CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


class VisualEmbedder:
    def __init__(self, model_name: str = "dinov2_vits14"):
        self.model_name = model_name
        self._model = None
        self._transform = None
        self._device = None
        self.dim = 0
        self._tried = False
        self._ok = False

    def _lazy_init(self) -> bool:
        if self._tried:
            return self._ok
        self._tried = True
        try:
            import torch
            from torchvision import transforms
            self._device = (
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            # DINOv2 from the official hub repo (downloads weights once).
            model = torch.hub.load("facebookresearch/dinov2", self.model_name,
                                   verbose=False)
            model.eval().to(self._device)
            self._model = model
            self._transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            # infer dim from a dry run
            import numpy as np  # noqa: F401
            self.dim = int(getattr(model, "embed_dim", 0)) or 384
            self._ok = True
        except Exception:
            self._ok = False
        return self._ok

    @property
    def available(self) -> bool:
        return self._lazy_init()

    def embed(self, image_path: str) -> Optional[List[float]]:
        if not self._lazy_init():
            return None
        try:
            import torch
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            x = self._transform(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(x)            # (1, dim)
            vec = feat[0].float().cpu().tolist()
            self.dim = len(vec)
            return vec
        except Exception:
            return None


def make_visual_embedder(mode: str = "none",
                         model_name: str = "dinov2_vits14") -> Optional[VisualEmbedder]:
    """mode: 'none' (disabled) | 'dinov2' | 'auto' (use if torch+model load).
    Returns a ready VisualEmbedder or None (callers fall back to text-embed)."""
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    emb = VisualEmbedder(model_name=model_name)
    if mode == "dinov2":
        return emb            # caller will discover availability lazily
    if mode == "auto":
        return emb if emb.available else None
    return None
