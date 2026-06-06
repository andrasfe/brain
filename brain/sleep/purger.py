"""ScreenPurger (NREM) — keep the observation stream from growing forever.

The capture loop already drops raw pixels at capture time, so the stream is
embeddings + short descriptions, not PNGs. This agent enforces the rest of the
retention policy during sleep:

  1. **Row retention** — delete `observation` memory rows older than
     `max_age_days`, and keep at most `max_rows` (newest wins).
  2. **Orphaned files** — delete any stray `*.png` left in the capture dir
     (defensive: a crash between capture and pixel-drop).
  3. **Disk budget** — if the capture dir still exceeds `max_dir_mb`, delete
     oldest files until under budget.

All bounds are configurable. Deletions are logged so a silent purge never
looks like data loss.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ..memory import OBSERVATION, Memory


class ScreenPurger:
    name = "screen_purger"

    def __init__(self, max_age_days: float = 30.0, max_rows: int = 20000,
                 capture_dir: Optional[str] = None, max_dir_mb: float = 200.0):
        self.max_age_days = max_age_days
        self.max_rows = max_rows
        self.capture_dir = capture_dir
        self.max_dir_mb = max_dir_mb

    def run(self, memory: Memory, *, log=None) -> dict[str, Any]:
        log = log or (lambda _m: None)
        now = time.time()
        pruned_age = pruned_cap = files_removed = 0

        # 1. age-based row pruning
        cutoff = now - self.max_age_days * 86400.0
        cur = memory.conn.execute(
            "DELETE FROM episodes WHERE mem_type=? AND ts < ?",
            (OBSERVATION, cutoff))
        pruned_age = cur.rowcount or 0

        # 2. count cap — keep newest max_rows
        row = memory.conn.execute(
            "SELECT COUNT(*) AS n FROM episodes WHERE mem_type=?",
            (OBSERVATION,)).fetchone()
        n = int(row["n"])
        if n > self.max_rows:
            excess = n - self.max_rows
            memory.conn.execute(
                "DELETE FROM episodes WHERE id IN ("
                "  SELECT id FROM episodes WHERE mem_type=? "
                "  ORDER BY ts ASC LIMIT ?)",
                (OBSERVATION, excess))
            pruned_cap = excess
        memory.conn.commit()

        # 3. orphaned PNGs + disk budget
        if self.capture_dir:
            files_removed = self._purge_files(Path(self.capture_dir), log)

        if pruned_age or pruned_cap or files_removed:
            log(f"  screen_purger: rows -{pruned_age} (age) -{pruned_cap} (cap), "
                f"files -{files_removed}")
        return {"pruned_age": pruned_age, "pruned_cap": pruned_cap,
                "files_removed": files_removed,
                "remaining": max(0, n - pruned_cap)}

    def _purge_files(self, d: Path, log) -> int:
        if not d.exists():
            return 0
        pngs = []
        try:
            for p in d.glob("afferent_frame_*.png"):
                try:
                    pngs.append((p, p.stat().st_mtime, p.stat().st_size))
                except OSError:
                    continue
        except OSError:
            return 0
        removed = 0
        # Any frame file older than 5 min is orphaned (capture drops within ms)
        cutoff = time.time() - 300
        survivors = []
        for p, mtime, size in pngs:
            if mtime < cutoff:
                try:
                    p.unlink(); removed += 1
                except OSError:
                    survivors.append((p, mtime, size))
            else:
                survivors.append((p, mtime, size))
        # disk budget: delete oldest survivors until under max_dir_mb
        budget = self.max_dir_mb * 1024 * 1024
        total = sum(s for _, _, s in survivors)
        survivors.sort(key=lambda t: t[1])  # oldest first
        i = 0
        while total > budget and i < len(survivors):
            p, _, size = survivors[i]
            try:
                p.unlink(); removed += 1; total -= size
            except OSError:
                pass
            i += 1
        return removed
