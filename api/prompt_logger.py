"""
api/prompt_logger.py
----------------------
Structured prompt + output logging for the Genesis API.

Logs every generation request with:
  - prompt, negative prompt, parameters
  - generated image paths (saved to disk)
  - latency, model used, seed
  - outcome (success/fail/fallback)

Storage: SQLite (single file, WAL mode, thread-safe).
All images are saved to outputs/api_images/<date>/<request_id>/.

The log feeds:
  - Feedback loop (which prompts succeed/fail)
  - Dataset collection (high-rated prompts → dataset samples)
  - Monitoring dashboards
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_log (
    request_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'anonymous',
    created_at      REAL NOT NULL,
    prompt          TEXT NOT NULL,
    negative_prompt TEXT DEFAULT '',
    model_key       TEXT NOT NULL,
    steps           INTEGER,
    guidance        REAL,
    width           INTEGER,
    height          INTEGER,
    seed            INTEGER,
    quality_tier    TEXT,
    n_images        INTEGER DEFAULT 1,
    duration_s      REAL,
    success         INTEGER NOT NULL DEFAULT 0,
    fallback_used   INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    image_paths     TEXT,   -- JSON array
    meta            TEXT    -- JSON blob for extra fields
);

CREATE INDEX IF NOT EXISTS idx_user      ON prompt_log(user_id);
CREATE INDEX IF NOT EXISTS idx_created   ON prompt_log(created_at);
CREATE INDEX IF NOT EXISTS idx_model     ON prompt_log(model_key);
CREATE INDEX IF NOT EXISTS idx_success   ON prompt_log(success);
"""


@dataclass
class PromptLogEntry:
    request_id: str
    user_id: str
    created_at: float
    prompt: str
    model_key: str
    negative_prompt: str = ""
    steps: int = 0
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    quality_tier: str = "fast"
    n_images: int = 1
    duration_s: Optional[float] = None
    success: bool = False
    fallback_used: bool = False
    error: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_row(self) -> tuple:
        return (
            self.request_id, self.user_id, self.created_at,
            self.prompt, self.negative_prompt, self.model_key,
            self.steps, self.guidance, self.width, self.height,
            self.seed, self.quality_tier, self.n_images,
            self.duration_s, int(self.success), int(self.fallback_used),
            self.error, json.dumps(self.image_paths),
            json.dumps(self.meta),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "PromptLogEntry":
        (rid, uid, ts, prompt, neg, model, steps, guide, w, h,
         seed, tier, n_imgs, dur, succ, fallback, err, paths, meta) = row
        return cls(
            request_id=rid, user_id=uid, created_at=ts,
            prompt=prompt, negative_prompt=neg or "",
            model_key=model, steps=steps or 0,
            guidance=guide or 7.5, width=w or 512, height=h or 512,
            seed=seed, quality_tier=tier or "fast",
            n_images=n_imgs or 1, duration_s=dur,
            success=bool(succ), fallback_used=bool(fallback),
            error=err,
            image_paths=json.loads(paths) if paths else [],
            meta=json.loads(meta) if meta else {},
        )


class PromptLogger:
    """
    SQLite-backed prompt and output logger.

    Usage:
        plog = PromptLogger("outputs/api_logs/prompts.db")
        plog.log(entry)
        entries = plog.recent(n=50)
        stats = plog.stats()
    """

    def __init__(self, db_path: str = "outputs/api_logs/prompts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def log(self, entry: PromptLogEntry) -> None:
        sql = """
        INSERT OR REPLACE INTO prompt_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            conn.execute(sql, entry.to_row())

    def log_result(
        self,
        request_id: str,
        prompt: str,
        model_key: str,
        duration_s: float,
        image_paths: List[str],
        success: bool,
        user_id: str = "anonymous",
        negative_prompt: str = "",
        steps: int = 0,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        quality_tier: str = "fast",
        fallback_used: bool = False,
        error: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> PromptLogEntry:
        """Convenience method for logging generation results directly."""
        entry = PromptLogEntry(
            request_id=request_id, user_id=user_id,
            created_at=time.time(), prompt=prompt,
            negative_prompt=negative_prompt, model_key=model_key,
            steps=steps, guidance=guidance, width=width, height=height,
            seed=seed, quality_tier=quality_tier,
            n_images=len(image_paths),
            duration_s=duration_s, success=success,
            fallback_used=fallback_used, error=error,
            image_paths=image_paths, meta=meta or {},
        )
        self.log(entry)
        return entry

    def get(self, request_id: str) -> Optional[PromptLogEntry]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_log WHERE request_id=?", (request_id,)
            ).fetchone()
        return PromptLogEntry.from_row(tuple(row)) if row else None

    def recent(
        self,
        n: int = 50,
        user_id: Optional[str] = None,
        success_only: bool = False,
    ) -> List[PromptLogEntry]:
        conditions = []
        params: list = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if success_only:
            conditions.append("success = 1")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM prompt_log {where} ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [PromptLogEntry.from_row(tuple(r)) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM prompt_log").fetchone()[0]
            success  = conn.execute("SELECT COUNT(*) FROM prompt_log WHERE success=1").fetchone()[0]
            failed   = total - success
            avg_dur  = conn.execute(
                "SELECT AVG(duration_s) FROM prompt_log WHERE success=1"
            ).fetchone()[0] or 0.0
            by_model = dict(conn.execute(
                "SELECT model_key, COUNT(*) FROM prompt_log GROUP BY model_key"
            ).fetchall())
            by_tier  = dict(conn.execute(
                "SELECT quality_tier, COUNT(*) FROM prompt_log GROUP BY quality_tier"
            ).fetchall())
            by_user  = dict(conn.execute(
                "SELECT user_id, COUNT(*) FROM prompt_log GROUP BY user_id ORDER BY 2 DESC LIMIT 20"
            ).fetchall())
        return {
            "total": total, "success": success, "failed": failed,
            "success_rate": round(success / max(total, 1), 3),
            "avg_duration_s": round(avg_dur, 2),
            "by_model": by_model,
            "by_tier": by_tier,
            "top_users": by_user,
        }

    def failed_prompts(self, limit: int = 100) -> List[str]:
        """Return unique prompts that frequently fail — useful for weakness detection."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT prompt, COUNT(*) as n FROM prompt_log "
                "WHERE success=0 GROUP BY prompt HAVING n >= 2 "
                "ORDER BY n DESC LIMIT ?", (limit,)
            ).fetchall()
        return [row[0] for row in rows]

    def export_successful(
        self,
        output_path: str,
        min_duration: float = 0.0,
        limit: Optional[int] = None,
    ) -> int:
        """Export successful prompt+image pairs as JSONL for dataset use."""
        sql = ("SELECT * FROM prompt_log WHERE success=1 "
               f"AND duration_s >= {min_duration} "
               "ORDER BY created_at DESC")
        if limit:
            sql += f" LIMIT {limit}"
        count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()
        with open(output_path, "w") as f:
            for row in rows:
                entry = PromptLogEntry.from_row(tuple(row))
                for path in entry.image_paths:
                    f.write(json.dumps({
                        "image_path": path,
                        "caption": entry.prompt,
                        "source": "api_generated",
                        "model": entry.model_key,
                        "seed": entry.seed,
                    }) + "\n")
                    count += 1
        logger.info(f"Exported {count} prompt-image pairs → {output_path}")
        return count
