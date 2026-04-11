"""
dataset/lifecycle/retention.py
--------------------------------
Dataset retention policies — controls how much data is kept and for how long.

Retention policies prevent unbounded dataset growth by:
  1. Enforcing a maximum dataset size (bytes or sample count)
  2. Time-based expiry — remove samples older than N days
  3. Source-based quotas — cap each data source at N samples
  4. Category-based balancing — keep proportional representation
  5. Quality-weighted retention — preferentially keep high-quality samples

When retention limits are exceeded, the RetentionManager determines
which records to evict using a configurable eviction strategy:
  lru_eviction      Remove least recently accessed records
  quality_eviction  Remove lowest quality_score records first
  age_eviction      Remove oldest records first
  random_eviction   Random subset (for diversity preservation)

Usage:
    policy = RetentionPolicy(
        max_samples=50_000,
        max_size_gb=20.0,
        max_age_days=90,
        source_quota={"active_learning": 5000, "openverse": 20000},
    )
    manager = RetentionManager("outputs/dataset", policy)
    eviction_result = manager.apply(dry_run=True)
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Defines what to keep in the dataset."""
    max_samples: Optional[int]       = None   # absolute record cap
    max_size_gb: Optional[float]     = None   # total images disk cap
    max_age_days: Optional[float]    = None   # remove records older than this
    source_quota: Dict[str, int]     = field(default_factory=dict)     # {source: max_count}
    category_quota: Dict[str, int]   = field(default_factory=dict)     # {category: max_count}
    min_quality_score: float         = 0.0
    eviction_strategy: str           = "quality_eviction"  # lru | quality | age | random
    protect_sources: Set[str]        = field(default_factory=set)   # never evict these

    def to_dict(self) -> dict:
        return {
            "max_samples": self.max_samples,
            "max_size_gb": self.max_size_gb,
            "max_age_days": self.max_age_days,
            "source_quota": self.source_quota,
            "category_quota": self.category_quota,
            "min_quality_score": self.min_quality_score,
            "eviction_strategy": self.eviction_strategy,
            "protect_sources": list(self.protect_sources),
        }


@dataclass
class RetentionResult:
    policy_applied: str
    total_before: int = 0
    total_after: int = 0
    evicted: int = 0
    bytes_freed: int = 0
    dry_run: bool = True
    duration_s: float = 0.0
    eviction_reasons: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        mode = "DRY RUN" if self.dry_run else "LIVE"
        return (
            f"[{mode}] Retention({self.policy_applied}): "
            f"before={self.total_before} after={self.total_after} "
            f"evicted={self.evicted} freed={self.bytes_freed/1024**2:.1f}MB "
            f"in {self.duration_s:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "policy_applied": self.policy_applied,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "evicted": self.evicted,
            "bytes_freed_mb": round(self.bytes_freed/1024**2, 2),
            "dry_run": self.dry_run,
            "duration_s": round(self.duration_s, 1),
            "eviction_reasons": self.eviction_reasons,
        }


class RetentionManager:
    """
    Applies retention policies to a Genesis dataset.

    Reads index.jsonl, identifies records to evict, optionally
    deletes image files and rewrites index.
    """

    def __init__(
        self,
        dataset_root: str,
        policy: RetentionPolicy,
        archive_dir: str = "outputs/archive",
    ):
        self.dataset_root = Path(dataset_root)
        self.policy       = policy
        self.archive_dir  = Path(archive_dir)
        self._index_path  = self.dataset_root / "index.jsonl"

    def _load_index(self) -> List[dict]:
        if not self._index_path.exists():
            return []
        records = []
        with open(self._index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _write_index(self, records: List[dict]) -> None:
        with open(self._index_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def _path_of(self, rec: dict) -> Optional[Path]:
        p = rec.get("path") or rec.get("file_path") or rec.get("image_path")
        return Path(p) if p else None

    def _file_size(self, rec: dict) -> int:
        p = self._path_of(rec)
        if p and p.exists():
            return p.stat().st_size
        return 0

    # ── Candidate identification ──────────────────────────────

    def _identify_eviction_candidates(
        self, records: List[dict]
    ) -> Dict[str, List[dict]]:
        """
        Returns dict of {reason: [records to evict]} for each violated policy.
        Does NOT evict — just identifies candidates.
        """
        candidates: Dict[str, List[dict]] = {}
        now = time.time()

        # 1. Time-based expiry
        if self.policy.max_age_days is not None:
            cutoff = now - self.policy.max_age_days * 86400
            expired = [
                r for r in records
                if float(r.get("timestamp", r.get("created_at", now))) < cutoff
                and r.get("source") not in self.policy.protect_sources
            ]
            if expired:
                candidates["age_expired"] = expired

        # 2. Quality floor
        if self.policy.min_quality_score > 0:
            low_q = [
                r for r in records
                if float(r.get("quality_score", 1.0)) < self.policy.min_quality_score
                and r.get("source") not in self.policy.protect_sources
            ]
            if low_q:
                candidates["quality_floor"] = low_q

        # 3. Source quotas
        for source, quota in self.policy.source_quota.items():
            source_records = [r for r in records if r.get("source") == source]
            if len(source_records) > quota:
                excess = len(source_records) - quota
                to_evict = self._select_eviction_subset(source_records, excess)
                candidates[f"source_quota:{source}"] = to_evict

        # 4. Category quotas
        for cat, quota in self.policy.category_quota.items():
            cat_records = [r for r in records if r.get("category") == cat]
            if len(cat_records) > quota:
                excess = len(cat_records) - quota
                to_evict = self._select_eviction_subset(cat_records, excess)
                candidates[f"category_quota:{cat}"] = to_evict

        # 5. Total sample cap
        if self.policy.max_samples and len(records) > self.policy.max_samples:
            excess = len(records) - self.policy.max_samples
            protected = {
                id(r) for r in records
                if r.get("source") in self.policy.protect_sources
            }
            eligible = [r for r in records if id(r) not in protected]
            to_evict = self._select_eviction_subset(eligible, excess)
            candidates["max_samples"] = to_evict

        return candidates

    def _select_eviction_subset(
        self, records: List[dict], n: int
    ) -> List[dict]:
        """Select n records to evict using the configured eviction strategy."""
        strategy = self.policy.eviction_strategy
        if strategy == "quality_eviction":
            # Remove lowest quality first
            sorted_recs = sorted(records,
                                  key=lambda r: float(r.get("quality_score", 0.5)))
            return sorted_recs[:n]
        elif strategy == "age_eviction":
            # Remove oldest first
            sorted_recs = sorted(records,
                                  key=lambda r: float(r.get("timestamp", 0)))
            return sorted_recs[:n]
        elif strategy == "lru_eviction":
            # Remove least recently accessed (use timestamp as proxy)
            sorted_recs = sorted(records,
                                  key=lambda r: float(r.get("last_accessed", r.get("timestamp", 0))))
            return sorted_recs[:n]
        else:  # random
            return random.sample(records, min(n, len(records)))

    # ── Apply ─────────────────────────────────────────────────

    def apply(self, dry_run: bool = True) -> RetentionResult:
        """
        Apply retention policy to dataset.

        Args:
            dry_run: If True, compute evictions but don't modify files/index.

        Returns:
            RetentionResult with full stats.
        """
        records = self._load_index()
        result  = RetentionResult(
            policy_applied=self.policy.eviction_strategy,
            total_before=len(records),
            dry_run=dry_run,
        )
        t0 = time.time()

        candidates = self._identify_eviction_candidates(records)

        # Union of all eviction candidates (avoid double-counting)
        evict_ids: Set[int] = set()
        for reason, recs in candidates.items():
            for r in recs:
                evict_ids.add(id(r))
            result.eviction_reasons[reason] = len(recs)

        evict_set = [r for r in records if id(r) in evict_ids]
        surviving  = [r for r in records if id(r) not in evict_ids]

        # Execute evictions
        for rec in evict_set:
            p = self._path_of(rec)
            size = 0
            if p and p.exists():
                size = p.stat().st_size
                if not dry_run:
                    # Archive instead of delete
                    self.archive_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(p), str(self.archive_dir / p.name))
                    except Exception:
                        p.unlink(missing_ok=True)
            result.bytes_freed += size
            result.evicted     += 1

        if not dry_run and evict_set:
            self._write_index(surviving)

        result.total_after = len(surviving)
        result.duration_s  = time.time() - t0

        logger.info(result.summary())

        # Persist report
        rp = self.archive_dir / f"retention_report_{int(t0)}.json"
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(result.to_dict(), indent=2))

        return result

    def dataset_stats(self) -> dict:
        """Return current dataset statistics relevant to retention."""
        records = self._load_index()
        now = time.time()

        total_size = sum(self._file_size(r) for r in records)
        by_source: Dict[str, int] = {}
        by_cat:    Dict[str, int] = {}
        ages: List[float] = []

        for r in records:
            src = r.get("source", "unknown")
            cat = r.get("category", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
            by_cat[cat]    = by_cat.get(cat, 0) + 1
            ts = float(r.get("timestamp", r.get("created_at", now)))
            ages.append((now - ts) / 86400)

        return {
            "n_samples": len(records),
            "total_size_gb": round(total_size / 1024**3, 2),
            "by_source": by_source,
            "by_category": by_cat,
            "mean_age_days": round(sum(ages)/len(ages), 1) if ages else 0,
            "max_age_days": round(max(ages), 1) if ages else 0,
        }

    def suggest_policy(
        self,
        target_samples: int = 50_000,
        target_size_gb: float = 20.0,
    ) -> RetentionPolicy:
        """
        Auto-generate a sensible retention policy based on current dataset stats.
        """
        stats = self.dataset_stats()
        source_quota = {
            src: min(count, target_samples // max(len(stats["by_source"]), 1))
            for src, count in stats["by_source"].items()
        }
        return RetentionPolicy(
            max_samples=target_samples,
            max_size_gb=target_size_gb,
            source_quota=source_quota,
            eviction_strategy="quality_eviction",
        )
