"""Joplin Data API client — the brain's knowledge-publishing backend.

The brain keeps its operational state in SQLite (typed memory, embeddings, the
world model, skills, the job queue). On top of that, the human-readable layer —
the daily story of what you did, read, and how you looked — is PUBLISHED as
markdown notes into Joplin, so it syncs to all your devices via Joplin Server.

Architecture note: Joplin *Server* is the sync backend; notes are created
through a Joplin *client's* Data API (the desktop Web Clipper service on
:41184, or the headless terminal app), which then syncs to the Server. So this
client talks to the Data API; Joplin Server distributes it.

Data API: every request carries `?token=`. Folders (notebooks) and notes are
markdown-native. Upsert is idempotent via a hidden marker line embedded in the
note body (`<!-- brain:KEY -->`) located through full-text search — so re-syncs
update the same daily page instead of duplicating it. No state kept here.

best-effort: any failure returns None/False rather than raising; the daemon
treats Joplin as an optional publish target, never a hard dependency.
"""
from __future__ import annotations

import re
from typing import Any, Optional

MARKER_RE = re.compile(r"<!--\s*brain:([^\s>]+)\s*-->")


def marker_line(key: str) -> str:
    return f"<!-- brain:{key} -->"


class JoplinClient:
    def __init__(self, base_url: str = "http://localhost:41184",
                 token: str = "", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._client = None
        self._folder_cache: dict[str, str] = {}

    # httpx is created lazily so importing this module never requires it.
    def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def _params(self, **kw) -> dict:
        p = {"token": self.token}
        p.update(kw)
        return p

    def ping(self) -> bool:
        try:
            r = self._http().get("/ping")
            return r.status_code == 200 and "JoplinClipperServer" in r.text
        except Exception:
            return False

    # ── folders (notebooks) ─────────────────────────────────────────────────
    def _all_folders(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        try:
            while True:
                r = self._http().get("/folders", params=self._params(page=page))
                r.raise_for_status()
                d = r.json()
                out.extend(d.get("items", []))
                if not d.get("has_more"):
                    break
                page += 1
        except Exception:
            return out
        return out

    def ensure_notebook(self, title: str,
                        parent_id: Optional[str] = None) -> Optional[str]:
        """Return the id of the notebook `title` under `parent_id`, creating it
        if needed. Cached per (parent,title)."""
        ck = f"{parent_id or ''}/{title}"
        if ck in self._folder_cache:
            return self._folder_cache[ck]
        for f in self._all_folders():
            if f.get("title") == title and (f.get("parent_id") or "") == (parent_id or ""):
                self._folder_cache[ck] = f["id"]
                return f["id"]
        body = {"title": title}
        if parent_id:
            body["parent_id"] = parent_id
        try:
            r = self._http().post("/folders", params=self._params(), json=body)
            r.raise_for_status()
            fid = r.json()["id"]
            self._folder_cache[ck] = fid
            return fid
        except Exception:
            return None

    # ── notes (upsert by embedded marker) ───────────────────────────────────
    def find_note_by_marker(self, key: str) -> Optional[str]:
        """Locate a note carrying `<!-- brain:KEY -->` via full-text search.
        Verifies the marker in the returned bodies (search is fuzzy)."""
        try:
            r = self._http().get("/search", params=self._params(
                query=f"brain:{key}", type="note", fields="id,body"))
            r.raise_for_status()
            for it in r.json().get("items", []):
                m = MARKER_RE.search(it.get("body", "") or "")
                if m and m.group(1) == key:
                    return it["id"]
        except Exception:
            return None
        return None

    def upsert_note(self, parent_id: str, title: str, body: str,
                    key: str) -> Optional[str]:
        """Create or update the note identified by `key`. The marker is embedded
        once at the top of the body. Returns the note id."""
        marker = marker_line(key)
        full = body if marker in body else f"{marker}\n\n{body}"
        nid = self.find_note_by_marker(key)
        try:
            if nid:
                r = self._http().put(f"/notes/{nid}", params=self._params(),
                                     json={"title": title, "body": full})
                r.raise_for_status()
                return nid
            r = self._http().post("/notes", params=self._params(),
                                  json={"title": title, "body": full,
                                        "parent_id": parent_id,
                                        "source_url": f"brain://{key}"})
            r.raise_for_status()
            return r.json()["id"]
        except Exception:
            return None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
