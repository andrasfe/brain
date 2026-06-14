"""FaceTrainer (NREM) — the "train while the screensaver's on" pass.

During WAKE the worker records cheap, order-dependent online sightings. This
sleep agent does the heavy, stable work in the idle window: recompute clusters
from ALL sightings (so identities don't drift with arrival order) and rebuild
each individual's app-usage profile by joining sightings to screen observations
on time. No capture happens here — purely consolidation of what was seen while
the user was present.

Owns ONE job, like the other sleep agents. No LLM calls. No-ops when face
identity is disabled or there are no sightings yet.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class FaceTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        fc = (getattr(cfg, "raw", {}) or {}).get("face") or {}
        self.enabled = bool(fc.get("enabled"))
        self.sim_threshold = float(fc.get("sim_threshold", 0.42))
        self.merge_threshold = float(fc.get("merge_threshold", 0.5))
        self.window_seconds = float(fc.get("window_seconds", 120))

    def run(self, *, log: Optional[Callable[[str], None]] = None
            ) -> dict[str, Any]:
        if not self.enabled:
            return {"ran": False, "reason": "disabled"}
        from brain.face import FaceIdentityStore
        store = FaceIdentityStore(
            self.cfg.db_path, sim_threshold=self.sim_threshold,
            merge_threshold=self.merge_threshold,
            window_seconds=self.window_seconds)
        try:
            if store.count_sightings() == 0:
                return {"ran": False, "reason": "no sightings"}
            rc = store.recluster()
            pr = store.build_profiles()
            if log:
                log(f"  👤 face: {rc['identities']} individual(s) from "
                    f"{rc['sightings']} sighting(s) "
                    f"({rc['merges']} merges); {pr['attributed']} app "
                    f"observation(s) attributed")
            return {"ran": True, **rc, **pr}
        finally:
            store.close()
