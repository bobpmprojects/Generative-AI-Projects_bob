"""Chroma-backed cache for permanent positioning and 24h intel TTL."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb

from .schemas import Positioning


class IntelCache:
    """Small convenience wrapper around a local persistent Chroma collection."""

    def __init__(self, db_path: str = "chroma_db") -> None:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection("market_intel_cache")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _id(kind: str, company: str, query: str = "") -> str:
        suffix = hashlib.md5(query.encode("utf-8")).hexdigest()[:12] if query else "static"
        return f"{kind}::{company.lower().strip()}::{suffix}"

    def get_positioning(self, company: str) -> Positioning | None:
        result = self.collection.get(ids=[self._id("positioning", company)])
        if not result.get("documents"):
            return None
        return Positioning.model_validate_json(result["documents"][0])

    def set_positioning(self, item: Positioning) -> None:
        self.collection.upsert(
            ids=[self._id("positioning", item.company_name)],
            documents=[item.model_dump_json()],
            metadatas=[{"kind": "positioning", "cached_at": self._now_iso(), "ttl_hours": -1}],
        )

    def get_ttl(self, kind: str, company: str, query: str, ttl_hours: int = 24) -> Any | None:
        result = self.collection.get(ids=[self._id(kind, company, query)])
        if not result.get("documents"):
            return None
        meta = result["metadatas"][0]
        cached_at = datetime.fromisoformat(meta["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=ttl_hours):
            return None
        return json.loads(result["documents"][0])

    def set_ttl(self, kind: str, company: str, query: str, value: Any, ttl_hours: int = 24) -> None:
        self.collection.upsert(
            ids=[self._id(kind, company, query)],
            documents=[json.dumps(value)],
            metadatas=[
                {"kind": kind, "cached_at": self._now_iso(), "ttl_hours": ttl_hours, "company": company}
            ],
        )

    def clear_ttl_entries(self) -> int:
        rows = self.collection.get(include=["metadatas"])
        ids = [
            row_id
            for row_id, meta in zip(rows.get("ids", []), rows.get("metadatas", []), strict=False)
            if meta.get("ttl_hours", -1) > 0
        ]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
