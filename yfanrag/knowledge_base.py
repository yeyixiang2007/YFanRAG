"""Knowledge base service helpers for UI and tooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
import re
import sqlite3

from .chunking import FixedChunker, RecursiveChunker
from .embedders import HashingEmbedder
from .fts import SqliteFtsIndex
from .loaders.text import TextFileLoader
from .pipeline import SimplePipeline
from .retrievers import HybridRetriever
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.sqlite_vec import SqliteVecStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

STORE_CHOICES = ("sqlite-vec1", "sqlite-vec", "duckdb-vss")
QUERY_MODE_CHOICES = ("vector", "hybrid", "fts")
DEFAULT_TABLES = {
    "sqlite-vec1": "vec1_chunks_data",
    "sqlite-vec": "vec_chunks",
    "duckdb-vss": "vss_chunks",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    db_path: str = "yfanrag_kb.db"
    store: str = "sqlite-vec1"
    table: str | None = None
    fts_table: str = "fts_chunks"
    dims: int = 8
    chunker: str = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120
    enable_fts: bool = True
    distance_metric: str = "l2"
    disable_sqlite_extension: bool = True
    sqlite_extension_path: str | None = None
    extension_whitelist: Sequence[str] = field(default_factory=tuple)
    disable_vss_extension: bool = False
    vss_persistent_index: bool = False
    hybrid_alpha: float = 0.5
    score_norm: str = "minmax"
    vector_top_k: int | None = None
    fts_top_k: int | None = None
    path_whitelist: Sequence[str] | None = None


@dataclass(frozen=True)
class KnowledgeBaseStats:
    db_path: str
    store: str
    table: str
    chunk_count: int
    doc_count: int


@dataclass(frozen=True)
class KnowledgeBaseHit:
    rank: int
    source: str
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    score: float | None = None
    distance: float | None = None
    vector_score: float | None = None
    fts_score: float | None = None


@dataclass(frozen=True)
class KnowledgeBaseIngestResult:
    document_count: int
    chunk_count: int
    doc_ids: list[str]


@dataclass(frozen=True)
class KnowledgeBaseDeleteResult:
    doc_ids: list[str]
    vector_deleted: int
    fts_deleted: int


class KnowledgeBaseManager:
    """Orchestrates ingest/query/delete operations for local KB data."""

    def ingest_paths(
        self,
        paths: Sequence[str],
        config: KnowledgeBaseConfig,
    ) -> KnowledgeBaseIngestResult:
        normalized_paths = self._normalize_paths(paths)
        if not normalized_paths:
            raise ValueError("at least one file or directory path is required")

        loader = TextFileLoader(
            paths=normalized_paths,
            path_whitelist=list(config.path_whitelist or []),
        )
        documents = loader.load()
        if not documents:
            raise ValueError("no supported .txt/.md documents found in selected paths")

        chunker = self._build_chunker(config)
        embedder = HashingEmbedder(dims=config.dims)
        store = self._build_store(config)
        try:
            pipeline = SimplePipeline(
                chunker=chunker,
                embedder=embedder,
                store=store,
            )
            if config.enable_fts and config.store != "duckdb-vss":
                fts = SqliteFtsIndex(
                    path=config.db_path,
                    table=self._safe_identifier(config.fts_table, label="fts table"),
                )
                try:
                    chunks = pipeline.upsert(documents, fts_index=fts)
                finally:
                    fts.close()
            else:
                chunks = pipeline.upsert(documents)
        finally:
            self._close_store(store)

        return KnowledgeBaseIngestResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            doc_ids=[doc.doc_id for doc in documents],
        )

    def delete_doc_ids(
        self,
        doc_ids: Sequence[str],
        config: KnowledgeBaseConfig,
    ) -> KnowledgeBaseDeleteResult:
        normalized_ids = [doc_id.strip() for doc_id in doc_ids if doc_id and doc_id.strip()]
        if not normalized_ids:
            raise ValueError("at least one doc_id is required")

        store = self._build_store(config)
        try:
            vector_deleted = int(store.delete_by_doc_ids(normalized_ids))
        finally:
            self._close_store(store)

        fts_deleted = 0
        if config.enable_fts and config.store != "duckdb-vss":
            fts = SqliteFtsIndex(
                path=config.db_path,
                table=self._safe_identifier(config.fts_table, label="fts table"),
            )
            try:
                fts_deleted = int(fts.delete_by_doc_ids(normalized_ids))
            finally:
                fts.close()

        return KnowledgeBaseDeleteResult(
            doc_ids=normalized_ids,
            vector_deleted=vector_deleted,
            fts_deleted=fts_deleted,
        )

    def query(
        self,
        query_text: str,
        top_k: int,
        mode: str,
        config: KnowledgeBaseConfig,
    ) -> list[KnowledgeBaseHit]:
        text = query_text.strip()
        if not text:
            return []
        if top_k <= 0:
            return []
        if mode not in QUERY_MODE_CHOICES:
            raise ValueError(f"unsupported query mode: {mode}")

        if mode == "fts":
            return self._query_fts(text, top_k, config)
        if mode == "hybrid":
            return self._query_hybrid(text, top_k, config)
        return self._query_vector(text, top_k, config)

    def list_doc_ids(
        self,
        config: KnowledgeBaseConfig,
        limit: int = 200,
    ) -> list[str]:
        if limit <= 0:
            return []
        table = self._table_name(config)

        if config.store == "duckdb-vss":
            return self._list_doc_ids_duckdb(config.db_path, table, limit)
        return self._list_doc_ids_sqlite(config.db_path, table, limit)

    def stats(self, config: KnowledgeBaseConfig) -> KnowledgeBaseStats:
        table = self._table_name(config)
        if config.store == "duckdb-vss":
            chunk_count, doc_count = self._stats_duckdb(config.db_path, table)
        else:
            chunk_count, doc_count = self._stats_sqlite(config.db_path, table)
        return KnowledgeBaseStats(
            db_path=config.db_path,
            store=config.store,
            table=table,
            chunk_count=chunk_count,
            doc_count=doc_count,
        )

    def _query_vector(
        self,
        query_text: str,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> list[KnowledgeBaseHit]:
        embedder = HashingEmbedder(dims=config.dims)
        store = self._build_store(config)
        try:
            embedding = embedder.embed([query_text])[0]
            chunks = store.query(embedding, top_k)
        finally:
            self._close_store(store)

        hits: list[KnowledgeBaseHit] = []
        for idx, chunk in enumerate(chunks, start=1):
            distance = chunk.metadata.get("distance")
            if isinstance(distance, (int, float)):
                score = -float(distance)
                distance_value = float(distance)
            else:
                score = None
                distance_value = None
            hits.append(
                KnowledgeBaseHit(
                    rank=idx,
                    source="vector",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                    score=score,
                    distance=distance_value,
                )
            )
        return hits

    def _query_hybrid(
        self,
        query_text: str,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> list[KnowledgeBaseHit]:
        if config.store == "duckdb-vss":
            raise ValueError("hybrid mode currently requires a sqlite store with FTS")

        embedder = HashingEmbedder(dims=config.dims)
        store = self._build_store(config)
        fts = SqliteFtsIndex(
            path=config.db_path,
            table=self._safe_identifier(config.fts_table, label="fts table"),
        )
        try:
            retriever = HybridRetriever(
                embedder=embedder,
                vector_store=store,
                fts_index=fts,
                alpha=config.hybrid_alpha,
                score_norm=config.score_norm,
            )
            rows = retriever.retrieve_with_scores(
                query=query_text,
                top_k=top_k,
                vector_top_k=config.vector_top_k,
                fts_top_k=config.fts_top_k,
            )
        finally:
            fts.close()
            self._close_store(store)

        hits: list[KnowledgeBaseHit] = []
        for idx, hit in enumerate(rows, start=1):
            chunk = hit.chunk
            distance = chunk.metadata.get("distance")
            hits.append(
                KnowledgeBaseHit(
                    rank=idx,
                    source="hybrid",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    start=chunk.start,
                    end=chunk.end,
                    score=float(hit.fused_score),
                    distance=float(distance) if isinstance(distance, (int, float)) else None,
                    vector_score=float(hit.vector_score),
                    fts_score=float(hit.fts_score),
                )
            )
        return hits

    def _query_fts(
        self,
        query_text: str,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> list[KnowledgeBaseHit]:
        if config.store == "duckdb-vss":
            raise ValueError("fts mode currently requires sqlite")

        fts = SqliteFtsIndex(
            path=config.db_path,
            table=self._safe_identifier(config.fts_table, label="fts table"),
        )
        try:
            rows = fts.query(query_text, top_k)
        finally:
            fts.close()

        hits: list[KnowledgeBaseHit] = []
        for idx, row in enumerate(rows, start=1):
            hits.append(
                KnowledgeBaseHit(
                    rank=idx,
                    source="fts",
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id or "",
                    text=row.text,
                    start=0,
                    end=len(row.text),
                    score=float(row.score),
                )
            )
        return hits

    def _build_store(self, config: KnowledgeBaseConfig) -> object:
        table = self._table_name(config)
        if config.store == "sqlite-vec1":
            return SqliteVec1Store(
                path=config.db_path,
                table=table,
                embedding_dim=config.dims,
                distance_metric=config.distance_metric,
                load_extension=not config.disable_sqlite_extension,
                extension_path=config.sqlite_extension_path,
                extension_whitelist=list(config.extension_whitelist or []),
            )
        if config.store == "sqlite-vec":
            return SqliteVecStore(
                path=config.db_path,
                table=table,
                embedding_dim=config.dims,
                distance_metric=config.distance_metric,
            )
        if config.store == "duckdb-vss":
            return DuckDbVssStore(
                path=config.db_path,
                table=table,
                embedding_dim=config.dims,
                distance_metric=config.distance_metric,
                enable_vss=not config.disable_vss_extension,
                persistent_index=config.vss_persistent_index,
                fail_if_no_vss=False,
            )
        raise ValueError(f"unsupported store backend: {config.store}")

    @staticmethod
    def _close_store(store: object) -> None:
        if hasattr(store, "close"):
            store.close()

    @staticmethod
    def _normalize_paths(paths: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for item in paths:
            if not item:
                continue
            raw = item.strip()
            if not raw:
                continue
            normalized.append(str(Path(raw)))
        return normalized

    @staticmethod
    def _build_chunker(config: KnowledgeBaseConfig) -> FixedChunker | RecursiveChunker:
        if config.chunker == "recursive":
            return RecursiveChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        if config.chunker == "fixed":
            return FixedChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        raise ValueError(f"unsupported chunker: {config.chunker}")

    def _table_name(self, config: KnowledgeBaseConfig) -> str:
        if config.store not in STORE_CHOICES:
            raise ValueError(f"unsupported store backend: {config.store}")
        base = config.table or DEFAULT_TABLES[config.store]
        return self._safe_identifier(base, label="vector table")

    @staticmethod
    def _safe_identifier(name: str, label: str) -> str:
        value = (name or "").strip()
        if not value:
            raise ValueError(f"{label} cannot be empty")
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError(
                f"{label} contains invalid characters: {value!r}; use letters, digits, underscore"
            )
        return value

    @staticmethod
    def _stats_sqlite(path: str, table: str) -> tuple[int, int]:
        conn = sqlite3.connect(path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                return (0, 0)
            row = conn.execute(
                f"SELECT COUNT(*) AS chunk_count, COUNT(DISTINCT doc_id) AS doc_count FROM {table}"
            ).fetchone()
            if row is None:
                return (0, 0)
            return int(row[0]), int(row[1])
        finally:
            conn.close()

    @staticmethod
    def _list_doc_ids_sqlite(path: str, table: str, limit: int) -> list[str]:
        conn = sqlite3.connect(path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                return []
            rows = conn.execute(
                f"SELECT DISTINCT doc_id FROM {table} WHERE doc_id IS NOT NULL "
                "ORDER BY doc_id LIMIT ?",
                (limit,),
            ).fetchall()
            return [str(row[0]) for row in rows if row and row[0]]
        finally:
            conn.close()

    @staticmethod
    def _stats_duckdb(path: str, table: str) -> tuple[int, int]:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("duckdb is not installed") from exc

        conn = duckdb.connect(path)
        try:
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) AS chunk_count, COUNT(DISTINCT doc_id) AS doc_count FROM {table}"
                ).fetchone()
            except Exception:
                return (0, 0)
            if row is None:
                return (0, 0)
            return int(row[0]), int(row[1])
        finally:
            conn.close()

    @staticmethod
    def _list_doc_ids_duckdb(path: str, table: str, limit: int) -> list[str]:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("duckdb is not installed") from exc

        conn = duckdb.connect(path)
        try:
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT doc_id FROM {table} WHERE doc_id IS NOT NULL "
                    "ORDER BY doc_id LIMIT ?",
                    [limit],
                ).fetchall()
            except Exception:
                return []
            return [str(row[0]) for row in rows if row and row[0]]
        finally:
            conn.close()

