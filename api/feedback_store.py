"""
api/feedback_store.py
-----------------------
Stores user feedback on generated images and feeds it back into
the active learning pipeline.

Feedback types:
  rating      — 1-5 star rating on a generated image
  thumbs      — simple up/down signal
  category    — user labels the image category (used for weakness detection)
  correction  — user provides a better prompt (model didn't understand original)
  report      — user flags inappropriate content

All feedback is stored in SQLite.
Feedback aggregation feeds directly into:
  - WeaknessDetector (which categories get low ratings)
  - QueryGenerator (which concepts to collect more data for)
  - DatasetExpander (trigger collection for poorly-rated categories)

Usage:
    store = FeedbackStore("outputs/api_logs/feedback.db")
    store.submit(FeedbackEntry(
        request_id="abc123",
        user_id="user_01",
        feedback_type="rating",
        rating=2,
        category="animals",
        note="The dog looks unnatural",
    ))
    weak = store.weak_categories(threshold=3.0)  # categories with mean rating < 3
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id    TEXT PRIMARY KEY,
    request_id     TEXT NOT NULL,
    user_id        TEXT NOT NULL DEFAULT 'anonymous',
    created_at     REAL NOT NULL,
    feedback_type  TEXT NOT NULL,  -- rating | thumbs | category | correction | report
    rating         INTEGER,        -- 1-5 for 'rating' type
    thumbs_up      INTEGER,        -- 1=up, 0=down for 'thumbs' type
    category       TEXT,           -- user-provided or inferred
    prompt         TEXT,           -- original prompt (denormalized for queries)
    correction     TEXT,           -- better prompt from user
    note           TEXT,           -- free-text note
    image_path     TEXT,           -- which image was rated
    meta           TEXT            -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_fb_request  ON feedback(request_id);
CREATE INDEX IF NOT EXISTS idx_fb_user     ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_fb_created  ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_fb_category ON feedback(category);
CREATE INDEX IF NOT EXISTS idx_fb_type     ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_fb_rating   ON feedback(rating);

CREATE TABLE IF NOT EXISTS category_stats (
    category    TEXT PRIMARY KEY,
    n_ratings   INTEGER DEFAULT 0,
    sum_ratings REAL    DEFAULT 0,
    n_thumbs_up INTEGER DEFAULT 0,
    n_thumbs    INTEGER DEFAULT 0,
    updated_at  REAL
);
"""


@dataclass
class FeedbackEntry:
    request_id: str
    feedback_type: str            # rating | thumbs | category | correction | report
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = "anonymous"
    created_at: float = field(default_factory=time.time)
    rating: Optional[int] = None          # 1-5
    thumbs_up: Optional[bool] = None      # True=up, False=down
    category: Optional[str] = None
    prompt: Optional[str] = None
    correction: Optional[str] = None      # better prompt
    note: Optional[str] = None
    image_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_row(self) -> tuple:
        return (
            self.feedback_id, self.request_id, self.user_id,
            self.created_at, self.feedback_type,
            self.rating,
            int(self.thumbs_up) if self.thumbs_up is not None else None,
            self.category, self.prompt, self.correction,
            self.note, self.image_path,
            json.dumps(self.meta),
        )

    @classmethod
    def from_row(cls, row) -> "FeedbackEntry":
        tu = row[6]
        return cls(
            feedback_id=row[0], request_id=row[1], user_id=row[2],
            created_at=row[3], feedback_type=row[4],
            rating=row[5],
            thumbs_up=bool(tu) if tu is not None else None,
            category=row[7], prompt=row[8], correction=row[9],
            note=row[10], image_path=row[11],
            meta=json.loads(row[12]) if row[12] else {},
        )


@dataclass
class CategoryFeedbackSummary:
    category: str
    n_ratings: int
    mean_rating: float        # 1.0 - 5.0
    thumbs_up_rate: float     # 0.0 - 1.0
    is_weak: bool             # True if mean_rating < threshold
    needs_data: bool          # True if weak + enough samples to be confident

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "n_ratings": self.n_ratings,
            "mean_rating": round(self.mean_rating, 2),
            "thumbs_up_rate": round(self.thumbs_up_rate, 2),
            "is_weak": self.is_weak,
            "needs_data": self.needs_data,
        }


class FeedbackStore:
    """
    Persistent user feedback store with aggregation for active learning.

    Thread-safe (WAL + per-request connections).
    """

    def __init__(
        self,
        db_path: str = "outputs/api_logs/feedback.db",
        weak_rating_threshold: float = 3.0,
        min_samples_for_weakness: int = 3,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.weak_threshold    = weak_rating_threshold
        self.min_samples       = min_samples_for_weakness
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Write ──────────────────────────────────────────────────

    def submit(self, entry: FeedbackEntry) -> str:
        """Store a feedback entry and update category stats."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                entry.to_row(),
            )
            # Update category_stats materialized view
            if entry.category:
                self._update_category_stats(conn, entry)
        logger.debug(f"Feedback stored: {entry.feedback_id} ({entry.feedback_type})")
        return entry.feedback_id

    def _update_category_stats(
        self, conn: sqlite3.Connection, entry: FeedbackEntry
    ) -> None:
        cat = entry.category
        existing = conn.execute(
            "SELECT * FROM category_stats WHERE category=?", (cat,)
        ).fetchone()

        if existing is None:
            conn.execute(
                "INSERT INTO category_stats VALUES (?,0,0,0,0,?)",
                (cat, time.time())
            )
            existing = conn.execute(
                "SELECT * FROM category_stats WHERE category=?", (cat,)
            ).fetchone()

        n_rat, sum_rat, n_up, n_th = existing[1], existing[2], existing[3], existing[4]

        if entry.rating is not None:
            n_rat   += 1
            sum_rat += entry.rating
        if entry.thumbs_up is not None:
            n_th += 1
            if entry.thumbs_up:
                n_up += 1

        conn.execute(
            "UPDATE category_stats SET n_ratings=?, sum_ratings=?, "
            "n_thumbs_up=?, n_thumbs=?, updated_at=? WHERE category=?",
            (n_rat, sum_rat, n_up, n_th, time.time(), cat),
        )

    # ── Read ───────────────────────────────────────────────────

    def get_by_request(self, request_id: str) -> List[FeedbackEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE request_id=? ORDER BY created_at",
                (request_id,),
            ).fetchall()
        return [FeedbackEntry.from_row(r) for r in rows]

    def recent(self, n: int = 100, feedback_type: Optional[str] = None) -> List[FeedbackEntry]:
        sql = "SELECT * FROM feedback"
        params: list = []
        if feedback_type:
            sql += " WHERE feedback_type=?"
            params.append(feedback_type)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [FeedbackEntry.from_row(r) for r in rows]

    # ── Aggregation for active learning ───────────────────────

    def category_summaries(self) -> List[CategoryFeedbackSummary]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT category, n_ratings, sum_ratings, n_thumbs_up, n_thumbs "
                "FROM category_stats ORDER BY n_ratings DESC"
            ).fetchall()
        summaries = []
        for cat, n_rat, sum_rat, n_up, n_th in rows:
            mean_rat = (sum_rat / n_rat) if n_rat > 0 else 0.0
            thumbs   = (n_up / n_th) if n_th > 0 else 0.0
            is_weak  = mean_rat < self.weak_threshold and n_rat > 0
            needs    = is_weak and n_rat >= self.min_samples
            summaries.append(CategoryFeedbackSummary(
                category=cat, n_ratings=n_rat, mean_rating=mean_rat,
                thumbs_up_rate=thumbs, is_weak=is_weak, needs_data=needs,
            ))
        return summaries

    def weak_categories(
        self, threshold: Optional[float] = None, min_samples: int = 1,
    ) -> List[str]:
        """
        Return categories with mean rating below threshold.
        Used by DatasetExpander.expand_from_feedback().
        """
        t = threshold or self.weak_threshold
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT category FROM category_stats "
                "WHERE n_ratings >= ? AND sum_ratings / n_ratings < ? "
                "ORDER BY sum_ratings / n_ratings ASC",
                (min_samples, t),
            ).fetchall()
        return [r[0] for r in rows]

    def correction_prompts(self, limit: int = 50) -> List[Tuple[str, str]]:
        """Return (original_prompt, correction) pairs from user corrections."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT prompt, correction FROM feedback "
                "WHERE feedback_type='correction' AND correction IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return list(rows)

    def stats(self) -> dict:
        with self._conn() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            by_type  = dict(conn.execute(
                "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type"
            ).fetchall())
            avg_rat  = conn.execute(
                "SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL"
            ).fetchone()[0] or 0.0
            n_cats   = conn.execute(
                "SELECT COUNT(*) FROM category_stats"
            ).fetchone()[0]
        weak = self.weak_categories()
        return {
            "total_feedback": total,
            "by_type": by_type,
            "avg_rating": round(avg_rat, 2),
            "n_categories_tracked": n_cats,
            "weak_categories": weak,
        }

    def to_active_learning_signal(self) -> dict:
        """
        Produce a structured signal for the DatasetExpander / WeaknessDetector.
        Called by the feedback loop in the orchestrator.
        """
        weak = self.weak_categories()
        summaries = [s.to_dict() for s in self.category_summaries() if s.needs_data]
        corrections = self.correction_prompts(limit=20)
        return {
            "weak_categories": weak,
            "category_summaries": summaries,
            "corrections": [
                {"original": o, "correction": c} for o, c in corrections
            ],
            "generated_at": time.time(),
        }
