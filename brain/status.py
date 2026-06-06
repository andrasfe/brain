"""Brain status — a snapshot of what the brain is doing, plus a tiny UI.

Three surfaces, zero new dependencies:
  - `gather_status(cfg)` → a structured dict (works whether or not the daemon
    is running — it reads the SQLite DBs directly + a `status.json` the daemon
    writes each tick for its live wake/sleep state + affect).
  - `python -m brain.status` → pretty terminal print (`--json` for raw).
  - `python -m brain.status --serve 8800` → a stdlib http.server dashboard
    that auto-refreshes (open http://localhost:8800).

The daemon writes `status.json` next to the memory DB via `write_status()`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def status_path(cfg) -> Path:
    return Path(cfg.db_path).parent / "status.json"


def write_status(cfg, payload: dict) -> None:
    """Atomic-ish write of the live daemon state."""
    p = status_path(cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        tmp.replace(p)
    except OSError:
        pass


def _dir_size(path: Path) -> "tuple[int, int]":
    """Return (total_bytes, file_count) for a directory (non-recursive glob of pngs)."""
    if not path or not path.exists():
        return (0, 0)
    total = count = 0
    try:
        for f in path.glob("*.png"):
            try:
                total += f.stat().st_size
                count += 1
            except OSError:
                pass
    except OSError:
        pass
    return (total, count)


def _safe_counts(db: Path) -> dict:
    out: dict[str, Any] = {"memory_by_type": {}, "skills": 0, "world_model": 0}
    if not db.exists():
        return out
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return out
    try:
        try:
            rows = conn.execute(
                "SELECT mem_type, COUNT(*) AS n FROM episodes GROUP BY mem_type"
            ).fetchall()
            out["memory_by_type"] = {r["mem_type"]: int(r["n"]) for r in rows}
        except sqlite3.Error:
            pass
        for tbl, key in (("skills", "skills"), ("world_model", "world_model")):
            try:
                r = conn.execute(f"SELECT COUNT(*) AS n FROM {tbl}").fetchone()
                out[key] = int(r["n"])
            except sqlite3.Error:
                pass
    finally:
        conn.close()
    return out


def gather_status(cfg) -> dict:
    db = Path(cfg.db_path)
    snap: dict[str, Any] = {"generated_ts": time.time()}

    # live daemon state (if the daemon is/was running)
    sp = status_path(cfg)
    daemon = None
    if sp.exists():
        try:
            daemon = json.loads(sp.read_text())
        except (OSError, ValueError):
            daemon = None
    if daemon:
        age = time.time() - float(daemon.get("ts", 0))
        daemon["age_seconds"] = round(age, 1)
        daemon["live"] = age < 120  # heartbeat fresh within 2 min
    snap["daemon"] = daemon or {"state": "offline", "live": False}

    # DB counts
    snap.update(_safe_counts(db))

    # disk — screenshot frames are contained in <sandbox>/frames
    sandbox = cfg.raw.get("sandbox_dir") if isinstance(cfg.raw, dict) else None
    base = Path(os.path.expanduser(sandbox)) if sandbox else db.parent
    cap_dir = base / "frames"
    cap_bytes, cap_files = _dir_size(cap_dir)
    snap["disk"] = {
        "memory_db_mb": round(db.stat().st_size / 1e6, 2) if db.exists() else 0.0,
        "capture_dir": str(cap_dir),
        "capture_pngs": cap_files,
        "capture_png_mb": round(cap_bytes / 1e6, 2),
    }
    return snap


# ── rendering ────────────────────────────────────────────────────────────────
def render_text(snap: dict) -> str:
    d = snap.get("daemon", {})
    lines = ["═" * 52, " BRAIN STATUS", "═" * 52]
    live = "● live" if d.get("live") else "○ offline"
    lines.append(f" daemon : {d.get('state', '?'):<10} {live}"
                 + (f"  (tick {d.get('tick')}, {d.get('age_seconds')}s ago)"
                    if d.get("tick") is not None else ""))
    af = d.get("affect") or {}
    if af:
        lines.append(f" mood   : {af.get('mood_label','?'):<10} "
                     f"val={af.get('valence',0):+.2f} arou={af.get('arousal',0):.2f} "
                     f"stress={af.get('stress',0):.2f} fatigue={af.get('fatigue',0):.2f}")
    mt = snap.get("memory_by_type", {})
    if mt:
        lines.append(" memory : " + ", ".join(f"{k}={v}" for k, v in sorted(mt.items())))
    lines.append(f" skills : {snap.get('skills', 0)}    "
                 f"world-model rows: {snap.get('world_model', 0)}")
    if "user_present" in d and d.get("user_present") is not None:
        lines.append(f" present: {'yes' if d['user_present'] else 'no (away → sleeping)'}")
    cap = d.get("capture") or {}
    if cap:
        lines.append(f" capture: captured={cap.get('captured',0)} "
                     f"deduped={cap.get('deduped',0)} "
                     f"skipped={cap.get('skipped',0)} "
                     f"privacy_ok={cap.get('privacy_ok')}"
                     + ("" if cap.get("privacy_ok", True) else f" [{cap.get('reason','')}]"))
    dk = snap.get("disk", {})
    lines.append(f" disk   : memory={dk.get('memory_db_mb',0)}MB  "
                 f"capture pngs={dk.get('capture_pngs',0)} "
                 f"({dk.get('capture_png_mb',0)}MB)")
    st = d.get("stats") or {}
    if st:
        keep = {k: st[k] for k in (
            "tasks_processed", "spontaneous_thoughts", "dreams_written",
            "episodes_pruned", "facts_consolidated", "forward_model_trains")
            if k in st}
        if keep:
            lines.append(" sleep  : " + ", ".join(f"{k}={v}" for k, v in keep.items()))
    lines.append("═" * 52)
    return "\n".join(lines)


def render_html(snap: dict) -> str:
    import html
    body = render_text(snap)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='5'>"
        "<title>brain status</title>"
        "<style>body{background:#0b0e14;color:#b3e283;font:14px/1.5 ui-monospace,"
        "Menlo,monospace;padding:24px}pre{white-space:pre-wrap}"
        "h1{color:#7aa2f7;font-size:16px}</style></head><body>"
        f"<h1>brain status <small style='color:#555'>(auto-refresh 5s)</small></h1>"
        f"<pre>{html.escape(body)}</pre>"
        "</body></html>"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def _serve(cfg, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            return

        def do_GET(self):  # noqa: N802
            snap = gather_status(cfg)
            if self.path.rstrip("/") == "/json":
                body = json.dumps(snap, default=str, indent=2).encode()
                ctype = "application/json"
            else:
                body = render_html(snap).encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"brain status dashboard → http://localhost:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


def main() -> int:
    import argparse
    import sys
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from brain.config import load_config

    ap = argparse.ArgumentParser(description="Show brain status.")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--serve", type=int, metavar="PORT",
                    help="serve an auto-refreshing web dashboard on PORT")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.serve:
        _serve(cfg, args.serve)
        return 0
    snap = gather_status(cfg)
    print(json.dumps(snap, default=str, indent=2) if args.json else render_text(snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
