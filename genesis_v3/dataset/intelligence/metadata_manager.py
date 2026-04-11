"""
dataset/intelligence/metadata_manager.py
------------------------------------------
SQLite-backed provenance metadata for every dataset item.

Schema: image_id, source_url, caption, timestamp, domain,
        embedding_id, dataset_version, quality_score,
        source_type, phash, file_size, width, height,
        filter_results, tags, file_path, md5

Thread-safe: WAL journal mode + per-call connection.
"""
from __future__ import annotations
import json, logging, sqlite3, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    image_id        TEXT PRIMARY KEY,
    source_url      TEXT DEFAULT '',
    caption         TEXT DEFAULT '',
    timestamp       REAL,
    domain          TEXT DEFAULT 'unknown',
    embedding_id    INTEGER,
    dataset_version TEXT DEFAULT 'v1',
    quality_score   REAL DEFAULT 0.0,
    source_type     TEXT DEFAULT 'crawled',
    phash           TEXT DEFAULT '',
    file_size       INTEGER DEFAULT 0,
    width           INTEGER DEFAULT 0,
    height          INTEGER DEFAULT 0,
    filter_results  TEXT DEFAULT '{}',
    tags            TEXT DEFAULT '[]',
    file_path       TEXT DEFAULT '',
    md5             TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_domain   ON items(domain);
CREATE INDEX IF NOT EXISTS idx_version  ON items(dataset_version);
CREATE INDEX IF NOT EXISTS idx_source   ON items(source_type);
CREATE INDEX IF NOT EXISTS idx_quality  ON items(quality_score);
CREATE INDEX IF NOT EXISTS idx_ts       ON items(timestamp);
CREATE INDEX IF NOT EXISTS idx_phash    ON items(phash);
"""

_INSERT = """
INSERT OR IGNORE INTO items
(image_id,source_url,caption,timestamp,domain,embedding_id,
 dataset_version,quality_score,source_type,phash,file_size,
 width,height,filter_results,tags,file_path,md5)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


@dataclass
class MetadataRecord:
    image_id:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    source_url:      str   = ""
    caption:         str   = ""
    timestamp:       float = field(default_factory=time.time)
    domain:          str   = "unknown"
    embedding_id:    Optional[int] = None
    dataset_version: str   = "v1"
    quality_score:   float = 0.0
    source_type:     str   = "crawled"
    phash:           str   = ""
    file_size:       int   = 0
    width:           int   = 0
    height:          int   = 0
    filter_results:  dict  = field(default_factory=dict)
    tags:            list  = field(default_factory=list)
    file_path:       str   = ""
    md5:             str   = ""

    def _row(self) -> tuple:
        return (self.image_id, self.source_url, self.caption, self.timestamp,
                self.domain, self.embedding_id, self.dataset_version,
                self.quality_score, self.source_type, self.phash,
                self.file_size, self.width, self.height,
                json.dumps(self.filter_results), json.dumps(self.tags),
                self.file_path, self.md5)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id, "source_url": self.source_url,
            "caption": self.caption, "timestamp": self.timestamp,
            "domain": self.domain, "embedding_id": self.embedding_id,
            "dataset_version": self.dataset_version,
            "quality_score": self.quality_score, "source_type": self.source_type,
            "phash": self.phash, "file_size": self.file_size,
            "width": self.width, "height": self.height,
            "filter_results": self.filter_results, "tags": self.tags,
            "file_path": self.file_path, "md5": self.md5,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MetadataRecord":
        return cls(
            image_id=row["image_id"], source_url=row["source_url"] or "",
            caption=row["caption"] or "", timestamp=row["timestamp"] or 0.0,
            domain=row["domain"] or "unknown",
            embedding_id=row["embedding_id"],
            dataset_version=row["dataset_version"] or "v1",
            quality_score=float(row["quality_score"] or 0.0),
            source_type=row["source_type"] or "crawled",
            phash=row["phash"] or "", file_size=int(row["file_size"] or 0),
            width=int(row["width"] or 0), height=int(row["height"] or 0),
            filter_results=json.loads(row["filter_results"] or "{}"),
            tags=json.loads(row["tags"] or "[]"),
            file_path=row["file_path"] or "", md5=row["md5"] or "",
        )

    @classmethod
    def from_builder_meta(cls, filename: str, meta: dict,
                          caption: str, version: str = "v1") -> "MetadataRecord":
        return cls(
            source_url=meta.get("url", ""), caption=caption,
            timestamp=meta.get("timestamp", time.time()),
            domain=meta.get("extra_meta", {}).get("category", "unknown"),
            dataset_version=version,
            source_type=meta.get("source", "crawled"),
            phash=meta.get("phash", ""),
            width=meta.get("width", 0), height=meta.get("height", 0),
            file_path=filename, md5=meta.get("md5", ""),
        )


class MetadataManager:
    """SQLite-backed metadata store with CRUD + rich queries."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
        logger.info(f"MetadataManager: {self.db_path} ({self.count()} records)")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD ──────────────────────────────────────────────────

    def insert(self, r: MetadataRecord) -> str:
        with self._conn() as c:
            c.execute(_INSERT, r._row())
        return r.image_id

    def insert_batch(self, records: List[MetadataRecord]) -> int:
        with self._conn() as c:
            c.executemany(_INSERT, [r._row() for r in records])
        return len(records)

    def get(self, image_id: str) -> Optional[MetadataRecord]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM items WHERE image_id=?",
                            (image_id,)).fetchone()
        return MetadataRecord.from_row(row) if row else None

    def delete(self, image_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM items WHERE image_id=?", (image_id,))

    def delete_batch(self, image_ids: List[str]) -> int:
        ph = ",".join("?" * len(image_ids))
        with self._conn() as c:
            c.execute(f"DELETE FROM items WHERE image_id IN ({ph})", image_ids)
        return len(image_ids)

    def update_field(self, image_id: str, field_name: str, value) -> None:
        allowed = {"embedding_id", "quality_score", "filter_results",
                   "tags", "domain", "dataset_version"}
        if field_name not in allowed:
            raise ValueError(f"Cannot update field: {field_name}")
        val = json.dumps(value) if isinstance(value, (dict, list)) else value
        with self._conn() as c:
            c.execute(f"UPDATE items SET {field_name}=? WHERE image_id=?",
                      (val, image_id))

    # ── Queries ───────────────────────────────────────────────

    def count(self, dataset_version: Optional[str] = None) -> int:
        with self._conn() as c:
            if dataset_version:
                return c.execute(
                    "SELECT COUNT(*) FROM items WHERE dataset_version=?",
                    (dataset_version,)).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def query(
        self,
        domain: Optional[str] = None,
        source_type: Optional[str] = None,
        dataset_version: Optional[str] = None,
        min_quality: float = 0.0,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = "timestamp DESC",
    ) -> List[MetadataRecord]:
        clauses, params = [], []
        if domain:
            clauses.append("domain=?"); params.append(domain)
        if source_type:
            clauses.append("source_type=?"); params.append(source_type)
        if dataset_version:
            clauses.append("dataset_version=?"); params.append(dataset_version)
        if min_quality > 0:
            clauses.append("quality_score>=?"); params.append(min_quality)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        lim   = f"LIMIT {limit} OFFSET {offset}" if limit else ""
        sql   = f"SELECT * FROM items {where} ORDER BY {order_by} {lim}"
        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [MetadataRecord.from_row(r) for r in rows]

    def get_by_phash(self, phash: str) -> Optional[MetadataRecord]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM items WHERE phash=?",
                            (phash,)).fetchone()
        return MetadataRecord.from_row(row) if row else None

    def domain_distribution(self) -> Dict[str, int]:
        with self._conn() as c:
            return {r[0]: r[1] for r in c.execute(
                "SELECT domain, COUNT(*) FROM items GROUP BY domain")}

    def version_distribution(self) -> Dict[str, int]:
        with self._conn() as c:
            return {r[0]: r[1] for r in c.execute(
                "SELECT dataset_version, COUNT(*) FROM items GROUP BY dataset_version")}

    def export_jsonl(self, output_path: str,
                     dataset_version: Optional[str] = None,
                     min_quality: float = 0.0) -> int:
        records = self.query(dataset_version=dataset_version, min_quality=min_quality)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in records:
                f.write(r.to_jsonl() + "\n")
        logger.info(f"Exported {len(records)} records → {output_path}")
        return len(records)

    def stats(self) -> dict:
        return {
            "total": self.count(),
            "by_domain": self.domain_distribution(),
            "by_version": self.version_distribution(),
            "db_path": str(self.db_path),
            "db_size_mb": round(self.db_path.stat().st_size / 1024**2, 2)
                if self.db_path.exists() else 0,
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(f"\n{'='*50}")
        print(f"  MetadataManager: {s['total']:,} records")
        print(f"  by_domain  : {s['by_domain']}")
        print(f"  by_version : {s['by_version']}")
        print(f"  db: {s['db_path']} ({s['db_size_mb']} MB)")
        print(f"{'='*50}\n")
