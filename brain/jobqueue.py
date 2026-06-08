"""JobQueue — a durable, local, SQLite-backed work queue.

Decouples fast perception from slow comprehension: the capture loop enqueues a
job (e.g. "deep-read this screenshot with the strong model") and returns
immediately; a worker drains the queue on its own schedule — throttled while
you're active, aggressive during sleep. The strong model can take its sweet
time without ever blocking the daemon tick.

Why SQLite and not RabbitMQ: this is a single-machine, single-user brain whose
whole substrate is already SQLite (memory, skills, world model). A networked
broker would be a server to babysit for a queue that holds a few dozen jobs.
This gives the same guarantees that matter here — durability across restarts,
atomic claim with leases, priority, dedup, TTL — with zero extra services. The
public surface is deliberately broker-shaped (`enqueue` / `claim` / `complete`
/ `fail`) so a Redis/RabbitMQ backend can slot in behind it later if the brain
ever outgrows one machine.

Thread-safe: one connection (check_same_thread=False) guarded by a re-entrant
lock, since the producer (daemon tick) and the worker thread share it. Throughput
is tiny, so a process-wide lock is the right simplicity/safety trade.

Time is injected (`now` arg, default time.time()) so the lease/TTL logic is
deterministically testable offline.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

PENDING = "pending"
LEASED = "leased"
DONE = "done"
FAILED = "failed"      # dead-letter: exhausted retries
DROPPED = "dropped"    # expired before processing (TTL)


class JobQueue:
    def __init__(self, db_path: Path, *, done_retention_seconds: float = 86400.0):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.done_retention_seconds = float(done_retention_seconds)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind         TEXT NOT NULL,
                    payload      TEXT NOT NULL DEFAULT '{}',
                    priority     INTEGER NOT NULL DEFAULT 0,
                    dedup_key    TEXT,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    enqueued_ts  REAL NOT NULL,
                    not_after_ts REAL,
                    lease_owner  TEXT,
                    lease_until  REAL,
                    attempts     INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    last_error   TEXT,
                    finished_ts  REAL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, priority DESC, enqueued_ts);
                CREATE INDEX IF NOT EXISTS idx_jobs_dedup ON jobs(dedup_key);
            """)
            self.conn.commit()

    # ── produce ─────────────────────────────────────────────────────────────
    def enqueue(self, kind: str, payload: Optional[dict] = None, *,
                priority: int = 0, dedup_key: Optional[str] = None,
                not_after_ts: Optional[float] = None, max_attempts: int = 3,
                now: Optional[float] = None) -> int:
        """Add a job; return its id. When `dedup_key` matches an existing
        PENDING job, coalesce: keep one job, lifting it to the max priority and
        latest payload/TTL (so a burst of near-identical frames collapses to
        the freshest one) and return that job's id."""
        now = time.time() if now is None else now
        payload_json = json.dumps(payload or {}, default=str)
        with self._lock:
            if dedup_key:
                row = self.conn.execute(
                    "SELECT id, priority FROM jobs WHERE dedup_key=? AND status=? "
                    "ORDER BY id LIMIT 1", (dedup_key, PENDING)).fetchone()
                if row is not None:
                    self.conn.execute(
                        "UPDATE jobs SET payload=?, priority=MAX(priority,?), "
                        "not_after_ts=?, enqueued_ts=? WHERE id=?",
                        (payload_json, int(priority), not_after_ts, now,
                         int(row["id"])))
                    self.conn.commit()
                    return int(row["id"])
            cur = self.conn.execute(
                "INSERT INTO jobs (kind, payload, priority, dedup_key, status, "
                "enqueued_ts, not_after_ts, max_attempts) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (kind, payload_json, int(priority), dedup_key, PENDING, now,
                 not_after_ts, int(max_attempts)))
            self.conn.commit()
            return int(cur.lastrowid)

    # ── consume ─────────────────────────────────────────────────────────────
    def claim(self, kinds: Optional[list[str]] = None, *, owner: str = "worker",
              lease_seconds: float = 300.0, now: Optional[float] = None
              ) -> Optional[dict]:
        """Atomically claim the highest-priority (then oldest) ready job, lease
        it, and return {id, kind, payload, attempts}. Reaps expired leases and
        drops TTL-expired jobs first. Returns None when nothing is ready."""
        now = time.time() if now is None else now
        with self._lock:
            self._reap(now)
            self._drop_expired(now)
            where = "status=?"
            params: list[Any] = [PENDING]
            if kinds:
                where += " AND kind IN (%s)" % ",".join("?" for _ in kinds)
                params += list(kinds)
            row = self.conn.execute(
                f"SELECT * FROM jobs WHERE {where} "
                "ORDER BY priority DESC, enqueued_ts ASC, id ASC LIMIT 1",
                params).fetchone()
            if row is None:
                return None
            self.conn.execute(
                "UPDATE jobs SET status=?, lease_owner=?, lease_until=?, "
                "attempts=attempts+1 WHERE id=?",
                (LEASED, owner, now + float(lease_seconds), int(row["id"])))
            self.conn.commit()
            return {"id": int(row["id"]), "kind": row["kind"],
                    "payload": json.loads(row["payload"]),
                    "attempts": int(row["attempts"]) + 1,
                    "priority": int(row["priority"])}

    def complete(self, job_id: int, *, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status=?, lease_owner=NULL, lease_until=NULL, "
                "finished_ts=? WHERE id=?", (DONE, now, int(job_id)))
            self.conn.commit()

    def fail(self, job_id: int, error: str = "", *, now: Optional[float] = None
             ) -> str:
        """Release a failed job: back to PENDING for retry, or dead-letter
        (FAILED) once attempts reach max_attempts. Returns the new status."""
        now = time.time() if now is None else now
        with self._lock:
            row = self.conn.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id=?",
                (int(job_id),)).fetchone()
            if row is None:
                return "missing"
            dead = int(row["attempts"]) >= int(row["max_attempts"])
            status = FAILED if dead else PENDING
            self.conn.execute(
                "UPDATE jobs SET status=?, lease_owner=NULL, lease_until=NULL, "
                "last_error=?, finished_ts=? WHERE id=?",
                (status, error[:500], now if dead else None, int(job_id)))
            self.conn.commit()
            return status

    # ── maintenance ───────────────────────────────────────────────────────────
    def _reap(self, now: float) -> int:
        """Expired leases (crashed/stuck worker) → back to PENDING."""
        cur = self.conn.execute(
            "UPDATE jobs SET status=?, lease_owner=NULL, lease_until=NULL "
            "WHERE status=? AND lease_until IS NOT NULL AND lease_until < ?",
            (PENDING, LEASED, now))
        return cur.rowcount

    def _drop_expired(self, now: float) -> int:
        """Pending jobs past their TTL → DROPPED (never worth processing)."""
        cur = self.conn.execute(
            "UPDATE jobs SET status=?, finished_ts=? "
            "WHERE status=? AND not_after_ts IS NOT NULL AND not_after_ts < ?",
            (DROPPED, now, PENDING, now))
        return cur.rowcount

    def reap_stale_leases(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            n = self._reap(now)
            self.conn.commit()
            return n

    def purge(self, now: Optional[float] = None) -> dict[str, int]:
        """Drop TTL-expired pending jobs and delete old finished rows."""
        now = time.time() if now is None else now
        with self._lock:
            dropped = self._drop_expired(now)
            cur = self.conn.execute(
                "DELETE FROM jobs WHERE status IN (?,?,?) AND finished_ts IS NOT NULL "
                "AND finished_ts < ?",
                (DONE, FAILED, DROPPED, now - self.done_retention_seconds))
            self.conn.commit()
            return {"dropped": dropped, "deleted": cur.rowcount}

    # ── inspection ──────────────────────────────────────────────────────────
    def depth(self, kind: Optional[str] = None) -> int:
        with self._lock:
            if kind:
                row = self.conn.execute(
                    "SELECT COUNT(*) AS n FROM jobs WHERE status=? AND kind=?",
                    (PENDING, kind)).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT COUNT(*) AS n FROM jobs WHERE status=?",
                    (PENDING,)).fetchone()
            return int(row["n"])

    def oldest_age(self, now: Optional[float] = None) -> Optional[float]:
        now = time.time() if now is None else now
        with self._lock:
            row = self.conn.execute(
                "SELECT MIN(enqueued_ts) AS t FROM jobs WHERE status=?",
                (PENDING,)).fetchone()
            return (now - float(row["t"])) if row and row["t"] is not None else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
            return {r["status"]: int(r["n"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            self.conn.close()
