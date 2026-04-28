"""Chroma-backed cache for permanent positioning and 24h intel TTL."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .schemas import Positioning


class IntelCache:
    """Small convenience wrapper around a local persistent Chroma collection."""

    def __init__(self, db_path: str = "chroma_db") -> None:
        self.collection: Any | None = None
        self.memory_cache: dict[str, dict[str, Any]] = {}
        try:
            import chromadb

            Path(db_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=db_path)
            self.collection = client.get_or_create_collection("market_intel_cache")
        except Exception:
            # Streamlit Cloud may briefly run unsupported Python builds; keep live runs functional.
            self.collection = None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _id(kind: str, company: str, query: str = "") -> str:
        suffix = hashlib.md5(query.encode("utf-8")).hexdigest()[:12] if query else "static"
        return f"{kind}::{company.lower().strip()}::{suffix}"

    def get_positioning(self, company: str) -> Positioning | None:
        if self.collection is None:
            row = self.memory_cache.get(self._id("positioning", company))
            return Positioning.model_validate_json(row["document"]) if row else None
        result = self.collection.get(ids=[self._id("positioning", company)])
        if not result.get("documents"):
            return None
        return Positioning.model_validate_json(result["documents"][0])

    def set_positioning(self, item: Positioning) -> None:
        if self.collection is None:
            self.memory_cache[self._id("positioning", item.company_name)] = {
                "document": item.model_dump_json(),
                "metadata": {"kind": "positioning", "cached_at": self._now_iso(), "ttl_hours": -1},
            }
            return
        self.collection.upsert(
            ids=[self._id("positioning", item.company_name)],
            documents=[item.model_dump_json()],
            metadatas=[{"kind": "positioning", "cached_at": self._now_iso(), "ttl_hours": -1}],
        )

    def get_ttl(self, kind: str, company: str, query: str, ttl_hours: int = 24) -> Any | None:
        if self.collection is None:
            row = self.memory_cache.get(self._id(kind, company, query))
            if not row:
                return None
            cached_at = datetime.fromisoformat(row["metadata"]["cached_at"])
            if datetime.now(timezone.utc) - cached_at > timedelta(hours=ttl_hours):
                return None
            return json.loads(row["document"])
        result = self.collection.get(ids=[self._id(kind, company, query)])
        if not result.get("documents"):
            return None
        meta = result["metadatas"][0]
        cached_at = datetime.fromisoformat(meta["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=ttl_hours):
            return None
        return json.loads(result["documents"][0])

    def set_ttl(self, kind: str, company: str, query: str, value: Any, ttl_hours: int = 24) -> None:
        metadata = {"kind": kind, "cached_at": self._now_iso(), "ttl_hours": ttl_hours, "company": company}
        if self.collection is None:
            self.memory_cache[self._id(kind, company, query)] = {
                "document": json.dumps(value),
                "metadata": metadata,
            }
            return
        self.collection.upsert(
            ids=[self._id(kind, company, query)],
            documents=[json.dumps(value)],
            metadatas=[metadata],
        )

    def clear_ttl_entries(self) -> int:
        if self.collection is None:
            ttl_ids = [
                key for key, row in self.memory_cache.items() if row["metadata"].get("ttl_hours", -1) > 0
            ]
            for key in ttl_ids:
                del self.memory_cache[key]
            return len(ttl_ids)
        rows = self.collection.get(include=["metadatas"])
        ids = [
            row_id
            for row_id, meta in zip(rows.get("ids", []), rows.get("metadatas", []), strict=False)
            if meta.get("ttl_hours", -1) > 0
        ]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
