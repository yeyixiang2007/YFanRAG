"""Knowledge base service helpers for UI and tooling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence
import importlib
import json
import math
import re
import sqlite3
import threading
import urllib.error
import urllib.request

from .chunking import FixedChunker, RecursiveChunker, StructureAwareChunker
from .embedders import (
    DEFAULT_FASTEMBED_MODEL,
    FastEmbedder,
    HashingEmbedder,
    embed_queries,
    embedder_dims,
)
from .fts import SqliteFtsIndex
from .io_utils import write_text_atomic
from .loaders.text import TextFileLoader
from .pipeline import SimplePipeline
from .retrievers import HybridRetriever
from .vectorstores.duckdb_vss import DuckDbVssStore
from .vectorstores.sqlite_vec import SqliteVecStore
from .vectorstores.sqlite_vec1 import SqliteVec1Store

STORE_CHOICES = ("sqlite-vec1", "sqlite-vec", "duckdb-vss")
QUERY_MODE_CHOICES = ("auto", "vector", "hybrid", "fts")
DEFAULT_TABLES = {
    "sqlite-vec1": "vec1_chunks_data",
    "sqlite-vec": "vec_chunks",
    "duckdb-vss": "vss_chunks",
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_SEMANTIC_HINT_RE = re.compile(
    r"(?:why|how|explain|difference|compare|summary|overview|design|trade-?off|best practice|"
    r"什么时候|为什么|如何|解释|区别|对比|总结|原理|建议)",
    re.IGNORECASE,
)
_KEYWORD_OP_RE = re.compile(r"\b(?:and|or|not)\b|[+\-|]", re.IGNORECASE)
_ERROR_HINT_RE = re.compile(
    r"(?:error|exception|traceback|stack trace|failed|errno|http\s*\d{3}|line\s*\d+|报错|异常|堆栈)",
    re.IGNORECASE,
)
_PATH_HINT_RE = re.compile(
    r"(?:[A-Za-z]:\\|/)[^\s]+|[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|ini|sql|js|ts|java|go|rs)\b",
    re.IGNORECASE,
)
_CODE_HINT_RE = re.compile(r"(?:`[^`]+`|::|->|=>|==|!=|<=|>=|[{}()\[\]<>])")
_RERANK_BACKEND_CHOICES = {"auto", "flashrank", "cross-encoder", "api", "heuristic", "none"}
_CROSS_ENCODER_CACHE: dict[str, object] = {}
_CROSS_ENCODER_FAILED_MODELS: set[str] = set()
_CROSS_ENCODER_LOADING_EVENTS: dict[str, threading.Event] = {}
_CROSS_ENCODER_CACHE_LOCK = threading.Lock()
_CROSS_ENCODER_CACHE_LIMIT = 2
_FLASHRANK_CACHE: dict[str, object] = {}
_FLASHRANK_FAILED_MODELS: set[str] = set()
_FLASHRANK_CACHE_LOCK = threading.Lock()
_FLASHRANK_CACHE_LIMIT = 2
_DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
_PENDING_FTS_RECOVERY_VERSION = 1
_CONTEXT_CODE_HINT_RE = re.compile(
    r"(?:```|^\s{4,}|[{}();]|=>|::|\b(?:def|class|return|import|from|SELECT|INSERT|UPDATE|DELETE)\b)",
    re.IGNORECASE | re.MULTILINE,
)
_COMMON_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
}
_EN_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "not",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "from",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
    "why",
}
_CN_STOPWORDS = {"的", "了", "和", "或", "与", "及", "在", "对", "是", "请", "并", "给出"}


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    db_path: str = "yfanrag_kb.db"
    store: str = "sqlite-vec1"
    table: str | None = None
    fts_table: str = "fts_chunks"
    dims: int = 384
    embedding_provider: str = "auto"
    embedding_model: str = DEFAULT_FASTEMBED_MODEL
    embedding_batch_size: int = 256
    chunker: str = "structured"
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
    score_norm: str = "sigmoid"
    vector_top_k: int | None = None
    fts_top_k: int | None = None
    multi_query_enabled: bool = True
    multi_query_count: int = 4
    multi_query_rrf_k: int = 60
    multi_query_candidate_top_k: int | None = None
    reranker_enabled: bool = True
    reranker_backend: str = "auto"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_flashrank_model: str = _DEFAULT_FLASHRANK_MODEL
    reranker_endpoint: str | None = None
    reranker_api_key: str | None = None
    reranker_api_key_header: str = "Authorization"
    reranker_timeout_seconds: int = 30
    reranker_candidate_top_k: int = 50
    context_compress_enabled: bool = True
    context_dedup_similarity: float = 0.9
    context_max_sentences_per_chunk: int = 3
    context_max_chars_per_chunk: int = 800
    context_max_total_chars: int = 4800
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
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class KnowledgeBaseQueryPlan:
    requested_mode: str
    resolved_mode: str
    query_type: str
    alpha: float | None = None
    vector_top_k: int | None = None
    fts_top_k: int | None = None
    query_variants: tuple[str, ...] = ()
    fusion: str | None = None
    rrf_k: int | None = None
    candidate_top_k: int | None = None
    reranker_backend: str | None = None
    reranker_candidate_top_k: int | None = None
    reranker_top_k: int | None = None


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


@dataclass(frozen=True)
class KnowledgeBaseContextCompression:
    input_chunks: int
    output_chunks: int
    duplicate_removed: int
    chars_before: int
    chars_after: int


@dataclass(frozen=True)
class _PendingFtsRecovery:
    store: str
    table: str
    fts_table: str
    doc_ids: tuple[str, ...]


class KnowledgeBaseManager:
    """Orchestrates ingest/query/delete operations for local KB data."""

    def __init__(self) -> None:
        self._last_query_plan: KnowledgeBaseQueryPlan | None = None

    @property
    def last_query_plan(self) -> KnowledgeBaseQueryPlan | None:
        return self._last_query_plan

    def ingest_paths(
        self,
        paths: Sequence[str],
        config: KnowledgeBaseConfig,
    ) -> KnowledgeBaseIngestResult:
        self._recover_pending_fts_consistency(config)
        normalized_paths = self._normalize_paths(paths)
        if not normalized_paths:
            raise ValueError("at least one file or directory path is required")

        loader = TextFileLoader(
            paths=normalized_paths,
            path_whitelist=list(config.path_whitelist or []),
        )
        documents = loader.load()
        if not documents:
            allowed = ", ".join(sorted({ext.lower() for ext in loader.allow_extensions}))
            raise ValueError(
                "no supported text/code documents found in selected paths; "
                f"allowed extensions: {allowed}"
            )

        chunker = self._build_chunker(config)
        embedder = self._build_embedder(config)
        store = self._build_store(config, embedding_dim=self._resolved_embedding_dims(config, embedder))
        try:
            pipeline = SimplePipeline(
                chunker=chunker,
                embedder=embedder,
                store=store,
            )
            prepared = pipeline.prepare_upsert(documents)
            if self._fts_enabled_for_config(config):
                fts = SqliteFtsIndex(
                    path=config.db_path,
                    table=self._safe_identifier(config.fts_table, label="fts table"),
                )
                try:
                    self._write_pending_fts_recovery(config, prepared.doc_ids)
                    try:
                        pipeline.replace_vectors(prepared)
                    except Exception:
                        self._clear_pending_fts_recovery(config)
                        raise
                    pipeline.replace_fts(prepared, fts_index=fts)
                    self._clear_pending_fts_recovery(config)
                    chunks = prepared.chunks
                finally:
                    fts.close()
            else:
                pipeline.replace_vectors(prepared)
                chunks = prepared.chunks
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
        self._recover_pending_fts_consistency(config)
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
        self._recover_pending_fts_consistency(config)
        text = query_text.strip()
        if not text:
            self._last_query_plan = None
            return []
        if top_k <= 0:
            self._last_query_plan = None
            return []
        requested_mode = (mode or "").strip().lower()
        if requested_mode not in QUERY_MODE_CHOICES:
            raise ValueError(f"unsupported query mode: {mode}")

        plan = self._build_query_plan(text, top_k, requested_mode, config)
        rerank_enabled = self._reranker_enabled(config)
        rerank_candidate_top_k = self._resolve_reranker_candidate_top_k(top_k, config)

        variant_count = self._resolve_multi_query_count(config)
        variants = self._expand_query_variants(text, variant_count)
        if not variants:
            variants = [text]
        if len(variants) == 1:
            coarse_hits = self._execute_query_by_plan(
                variants[0],
                rerank_candidate_top_k,
                plan,
                config,
            )
            reranked, reranker_backend = self._maybe_rerank_hits(
                query_text=text,
                hits=coarse_hits,
                top_k=top_k,
                config=config,
            )
            final_plan = replace(
                plan,
                query_variants=tuple(variants),
                reranker_backend=reranker_backend,
                reranker_candidate_top_k=rerank_candidate_top_k if rerank_enabled else None,
                reranker_top_k=top_k if reranker_backend is not None else None,
            )
            self._last_query_plan = final_plan
            return reranked

        candidate_top_k = self._resolve_multi_query_candidate_top_k(top_k, config)
        if rerank_enabled and candidate_top_k < rerank_candidate_top_k:
            candidate_top_k = rerank_candidate_top_k
        rrf_k = self._normalize_rrf_k(config.multi_query_rrf_k)
        runs: list[list[KnowledgeBaseHit]] = []
        for variant in variants:
            hits = self._execute_query_by_plan(variant, candidate_top_k, plan, config)
            if hits:
                runs.append(hits)
        if not runs:
            final_plan = replace(
                plan,
                query_variants=tuple(variants),
                fusion="rrf",
                rrf_k=rrf_k,
                candidate_top_k=candidate_top_k,
                reranker_candidate_top_k=rerank_candidate_top_k if rerank_enabled else None,
            )
            self._last_query_plan = final_plan
            return []

        fused_hits = self._fuse_rrf_hits(
            runs=runs,
            top_k=rerank_candidate_top_k,
            rrf_k=rrf_k,
        )
        reranked, reranker_backend = self._maybe_rerank_hits(
            query_text=text,
            hits=fused_hits,
            top_k=top_k,
            config=config,
        )
        final_plan = replace(
            plan,
            query_variants=tuple(variants),
            fusion="rrf",
            rrf_k=rrf_k,
            candidate_top_k=candidate_top_k,
            reranker_backend=reranker_backend,
            reranker_candidate_top_k=rerank_candidate_top_k if rerank_enabled else None,
            reranker_top_k=top_k if reranker_backend is not None else None,
        )
        self._last_query_plan = final_plan
        return reranked

    def compress_hits_for_context(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
        config: KnowledgeBaseConfig,
        max_chunks: int,
    ) -> tuple[list[KnowledgeBaseHit], KnowledgeBaseContextCompression]:
        source_hits = [hit for hit in hits if hit.text and hit.text.strip()]
        input_chunks = len(source_hits)
        if max_chunks <= 0 or input_chunks == 0:
            return [], KnowledgeBaseContextCompression(
                input_chunks=input_chunks,
                output_chunks=0,
                duplicate_removed=0,
                chars_before=0,
                chars_after=0,
            )

        pool_size = max(max_chunks * 6, 24)
        candidates = source_hits[:pool_size]
        chars_before = sum(len(" ".join(hit.text.split())) for hit in candidates)
        if not config.context_compress_enabled:
            clipped = self._truncate_hits(candidates, max_chunks)
            chars_after = sum(len(hit.text) for hit in clipped)
            return clipped, KnowledgeBaseContextCompression(
                input_chunks=input_chunks,
                output_chunks=len(clipped),
                duplicate_removed=0,
                chars_before=chars_before,
                chars_after=chars_after,
            )

        threshold = self._normalize_context_similarity_threshold(config.context_dedup_similarity)
        sentence_limit = self._normalize_context_sentence_limit(config.context_max_sentences_per_chunk)
        per_chunk_chars = self._normalize_context_chars_per_chunk(config.context_max_chars_per_chunk)
        total_chars = self._resolve_context_total_chars(
            max_chunks=max_chunks,
            per_chunk_chars=per_chunk_chars,
            configured_total_chars=config.context_max_total_chars,
        )

        query_terms = [
            term.lower()
            for term in _WORD_RE.findall(query_text)
            if self._is_query_keyword(term)
        ]
        unique_query_terms = list(dict.fromkeys(query_terms))

        dedup_hits: list[KnowledgeBaseHit] = []
        dedup_texts: list[str] = []
        dedup_vectors: list[list[float]] = []
        vector_dims = max(128, min(1024, config.dims * 8))
        duplicate_removed = 0

        for hit in candidates:
            normalized_text = " ".join(hit.text.split())
            if not normalized_text:
                continue
            compressed_text = self._extract_key_sentences(
                text=normalized_text,
                query_terms=unique_query_terms,
                max_sentences=sentence_limit,
                max_chars=per_chunk_chars,
            )
            if not compressed_text:
                compressed_text = self._clip_text(normalized_text, per_chunk_chars)
            candidate_vector = self._token_hash_vector(compressed_text, dims=vector_dims)
            if self._is_semantic_duplicate(
                text=compressed_text,
                vector=candidate_vector,
                existing_texts=dedup_texts,
                existing_vectors=dedup_vectors,
                similarity_threshold=threshold,
            ):
                duplicate_removed += 1
                continue
            dedup_texts.append(compressed_text)
            dedup_vectors.append(candidate_vector)
            dedup_hits.append(replace(hit, text=compressed_text))
            if len(dedup_hits) >= max_chunks:
                break

        if not dedup_hits:
            first = candidates[0]
            fallback_text = self._extract_key_sentences(
                text=" ".join(first.text.split()),
                query_terms=unique_query_terms,
                max_sentences=sentence_limit,
                max_chars=per_chunk_chars,
            )
            if not fallback_text:
                fallback_text = self._clip_text(" ".join(first.text.split()), per_chunk_chars)
            dedup_hits = [replace(first, text=fallback_text)]

        budgeted_hits = self._apply_context_char_budget(
            hits=dedup_hits,
            max_total_chars=total_chars,
        )
        final_hits = self._truncate_hits(budgeted_hits, max_chunks)
        chars_after = sum(len(hit.text) for hit in final_hits)
        return final_hits, KnowledgeBaseContextCompression(
            input_chunks=input_chunks,
            output_chunks=len(final_hits),
            duplicate_removed=duplicate_removed,
            chars_before=chars_before,
            chars_after=chars_after,
        )

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
        embedder = self._build_embedder(config)
        store = self._build_store(config, embedding_dim=self._resolved_embedding_dims(config, embedder))
        try:
            embedding = embed_queries(embedder, [query_text])[0]
            chunks = store.query(embedding, top_k)
        finally:
            self._close_store(store)

        hits: list[KnowledgeBaseHit] = []
        for idx, chunk in enumerate(chunks, start=1):
            distance = chunk.metadata.get("distance")
            if isinstance(distance, (int, float)):
                distance_value = max(0.0, float(distance))
                score = 1.0 / (1.0 + distance_value)
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
        alpha: float | None = None,
        vector_top_k: int | None = None,
        fts_top_k: int | None = None,
    ) -> list[KnowledgeBaseHit]:
        if config.store == "duckdb-vss":
            raise ValueError("hybrid mode currently requires a sqlite store with FTS")

        alpha_value = self._clamp(
            config.hybrid_alpha if alpha is None else float(alpha),
            0.0,
            1.0,
        )
        vector_limit = self._normalize_candidate_top_k(vector_top_k, top_k)
        if vector_limit is None:
            vector_limit = self._normalize_candidate_top_k(config.vector_top_k, top_k)
        fts_limit = self._normalize_candidate_top_k(fts_top_k, top_k)
        if fts_limit is None:
            fts_limit = self._normalize_candidate_top_k(config.fts_top_k, top_k)

        embedder = self._build_embedder(config)
        store = self._build_store(config, embedding_dim=self._resolved_embedding_dims(config, embedder))
        fts = SqliteFtsIndex(
            path=config.db_path,
            table=self._safe_identifier(config.fts_table, label="fts table"),
        )
        try:
            retriever = HybridRetriever(
                embedder=embedder,
                vector_store=store,
                fts_index=fts,
                alpha=alpha_value,
                score_norm=config.score_norm,
            )
            rows = retriever.retrieve_with_scores(
                query=query_text,
                top_k=top_k,
                vector_top_k=vector_limit,
                fts_top_k=fts_limit,
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

    def _execute_query_by_plan(
        self,
        query_text: str,
        top_k: int,
        plan: KnowledgeBaseQueryPlan,
        config: KnowledgeBaseConfig,
    ) -> list[KnowledgeBaseHit]:
        if plan.resolved_mode == "fts":
            return self._query_fts(query_text, top_k, config)
        if plan.resolved_mode == "hybrid":
            return self._query_hybrid(
                query_text,
                top_k,
                config,
                alpha=plan.alpha,
                vector_top_k=plan.vector_top_k,
                fts_top_k=plan.fts_top_k,
            )
        return self._query_vector(query_text, top_k, config)

    @staticmethod
    def _fuse_rrf_hits(
        runs: Sequence[Sequence[KnowledgeBaseHit]],
        top_k: int,
        rrf_k: int,
    ) -> list[KnowledgeBaseHit]:
        if top_k <= 0:
            return []
        accumulator: dict[tuple[str, str], dict[str, object]] = {}
        for run_index, run in enumerate(runs):
            for rank, hit in enumerate(run, start=1):
                key = (hit.doc_id, hit.chunk_id)
                bonus = 1.0 / float(rrf_k + rank)
                item = accumulator.get(key)
                if item is None:
                    accumulator[key] = {
                        "rrf_score": bonus,
                        "best_rank": rank,
                        "run_index": run_index,
                        "hit": hit,
                    }
                    continue
                item["rrf_score"] = float(item["rrf_score"]) + bonus
                if rank < int(item["best_rank"]):
                    item["best_rank"] = rank
                    item["run_index"] = run_index
                    item["hit"] = hit
        ranked = sorted(
            accumulator.values(),
            key=lambda item: (
                float(item["rrf_score"]),
                -int(item["best_rank"]),
                -int(item["run_index"]),
                str(getattr(item["hit"], "chunk_id", "")),
            ),
            reverse=True,
        )
        merged: list[KnowledgeBaseHit] = []
        for idx, item in enumerate(ranked[:top_k], start=1):
            hit = item["hit"]
            if not isinstance(hit, KnowledgeBaseHit):
                continue
            merged.append(
                replace(
                    hit,
                    rank=idx,
                    rrf_score=float(item["rrf_score"]),
                )
            )
        return merged

    def _maybe_rerank_hits(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> tuple[list[KnowledgeBaseHit], str | None]:
        if top_k <= 0:
            return [], None
        if not hits:
            return [], None
        if not self._reranker_enabled(config):
            return self._truncate_hits(hits, top_k), None

        backend = self._normalize_reranker_backend(config.reranker_backend)
        candidate_top_k = self._resolve_reranker_candidate_top_k(top_k, config)
        candidates = list(hits[:candidate_top_k])
        if not candidates:
            return [], None
        if len(candidates) == 1:
            single = replace(
                candidates[0],
                rank=1,
                rerank_score=float(candidates[0].rrf_score or candidates[0].score or 0.0),
            )
            return [single], "heuristic"

        if backend in {"auto", "flashrank"}:
            scores = self._try_flashrank_rerank_scores(
                query_text=query_text,
                hits=candidates,
                config=config,
            )
            if scores:
                return self._apply_rerank_scores(candidates, scores, top_k), "flashrank"

        if backend in {"auto", "cross-encoder"}:
            scores = self._try_cross_encoder_rerank_scores(
                query_text=query_text,
                hits=candidates,
                config=config,
            )
            if scores:
                return self._apply_rerank_scores(candidates, scores, top_k), "cross-encoder"

        if backend in {"auto", "api"}:
            scores = self._try_api_rerank_scores(
                query_text=query_text,
                hits=candidates,
                top_k=top_k,
                config=config,
            )
            if scores:
                return self._apply_rerank_scores(candidates, scores, top_k), "api"

        scores = self._heuristic_rerank_scores(query_text=query_text, hits=candidates)
        return self._apply_rerank_scores(candidates, scores, top_k), "heuristic"

    @staticmethod
    def _truncate_hits(hits: Sequence[KnowledgeBaseHit], top_k: int) -> list[KnowledgeBaseHit]:
        output: list[KnowledgeBaseHit] = []
        for idx, hit in enumerate(hits[:top_k], start=1):
            output.append(replace(hit, rank=idx))
        return output

    @staticmethod
    def _apply_rerank_scores(
        hits: Sequence[KnowledgeBaseHit],
        scores: dict[int, float],
        top_k: int,
    ) -> list[KnowledgeBaseHit]:
        def _fallback_score(idx: int, hit: KnowledgeBaseHit) -> float:
            if idx in scores:
                return float(scores[idx])
            if hit.rrf_score is not None:
                return float(hit.rrf_score)
            if hit.score is not None:
                return float(hit.score)
            return 0.0

        ranked = sorted(
            enumerate(hits),
            key=lambda pair: (
                _fallback_score(pair[0], pair[1]),
                -int(pair[1].rank),
                -pair[0],
            ),
            reverse=True,
        )
        output: list[KnowledgeBaseHit] = []
        for idx, (source_index, hit) in enumerate(ranked[:top_k], start=1):
            output.append(
                replace(
                    hit,
                    rank=idx,
                    rerank_score=_fallback_score(source_index, hit),
                )
            )
        return output

    def _heuristic_rerank_scores(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
    ) -> dict[int, float]:
        query_terms = [
            term.lower()
            for term in _WORD_RE.findall(query_text)
            if self._is_query_keyword(term)
        ]
        query_unique = list(dict.fromkeys(query_terms))
        query_joined = " ".join(query_terms)

        scores: dict[int, float] = {}
        for idx, hit in enumerate(hits):
            doc_terms = [term.lower() for term in _WORD_RE.findall(hit.text)]
            if not doc_terms:
                prior = float(hit.rrf_score or hit.score or 0.0)
                scores[idx] = prior
                continue

            doc_set = set(doc_terms)
            overlap = sum(1 for term in query_unique if term in doc_set)
            overlap_ratio = overlap / float(max(1, len(query_unique)))

            frequency = sum(doc_terms.count(term) for term in query_unique[:10])
            phrase_bonus = 0.0
            if query_joined:
                compact_doc = " ".join(doc_terms)
                if query_joined in compact_doc:
                    phrase_bonus = 1.5
            adjacency_bonus = self._adjacency_bonus(query_unique, doc_terms)
            prior = float(hit.rrf_score or hit.score or 0.0)
            score = (
                overlap_ratio * 3.2
                + float(frequency) * 0.08
                + phrase_bonus
                + adjacency_bonus
                + prior * 0.05
            )
            scores[idx] = score
        return scores

    @staticmethod
    def _adjacency_bonus(query_terms: Sequence[str], doc_terms: Sequence[str]) -> float:
        if len(query_terms) < 2 or not doc_terms:
            return 0.0
        compact_doc = " ".join(doc_terms)
        bonus = 0.0
        for idx in range(0, len(query_terms) - 1):
            phrase = f"{query_terms[idx]} {query_terms[idx + 1]}"
            if phrase and phrase in compact_doc:
                bonus += 0.25
        return min(1.0, bonus)

    def _try_flashrank_rerank_scores(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
        config: KnowledgeBaseConfig,
    ) -> dict[int, float] | None:
        model_name = (config.reranker_flashrank_model or "").strip() or _DEFAULT_FLASHRANK_MODEL
        if model_name in _FLASHRANK_FAILED_MODELS:
            return None
        try:
            ranker = self._get_or_load_flashrank_ranker(model_name)
            if ranker is None:
                return None
            module = importlib.import_module("flashrank")
            rerank_request = getattr(module, "RerankRequest", None)
            if rerank_request is None:
                return None
            passages = [
                {
                    "id": idx,
                    "text": hit.text,
                    "meta": {
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                    },
                }
                for idx, hit in enumerate(hits)
            ]
            request = rerank_request(query=query_text, passages=passages)
            ranked = ranker.rerank(request)
            return self._parse_flashrank_scores(ranked, len(hits))
        except (ImportError, OSError, RuntimeError, ValueError, TypeError):
            _FLASHRANK_FAILED_MODELS.add(model_name)
            return None

    def _get_or_load_flashrank_ranker(self, model_name: str) -> object | None:
        with _FLASHRANK_CACHE_LOCK:
            ranker = _FLASHRANK_CACHE.get(model_name)
            if ranker is not None:
                _FLASHRANK_CACHE.pop(model_name, None)
                _FLASHRANK_CACHE[model_name] = ranker
                return ranker
        module = importlib.import_module("flashrank")
        ranker_cls = getattr(module, "Ranker", None)
        if ranker_cls is None:
            return None
        ranker = ranker_cls(model_name=model_name)
        with _FLASHRANK_CACHE_LOCK:
            _FLASHRANK_CACHE[model_name] = ranker
            while len(_FLASHRANK_CACHE) > _FLASHRANK_CACHE_LIMIT:
                oldest = next(iter(_FLASHRANK_CACHE))
                if oldest == model_name and len(_FLASHRANK_CACHE) == 1:
                    break
                _FLASHRANK_CACHE.pop(oldest, None)
        return ranker

    @staticmethod
    def _parse_flashrank_scores(payload: object, item_count: int) -> dict[int, float] | None:
        if not isinstance(payload, list):
            return None
        scores: dict[int, float] = {}
        for pos, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            idx: int | None = None
            raw_idx = row.get("id")
            if isinstance(raw_idx, int):
                idx = raw_idx
            elif isinstance(raw_idx, str) and raw_idx.strip().isdigit():
                idx = int(raw_idx.strip())
            if idx is None:
                meta = row.get("meta")
                if isinstance(meta, dict):
                    meta_idx = meta.get("id") or meta.get("index")
                    if isinstance(meta_idx, int):
                        idx = meta_idx
                    elif isinstance(meta_idx, str) and meta_idx.strip().isdigit():
                        idx = int(meta_idx.strip())
            if idx is None or idx < 0 or idx >= item_count:
                continue
            raw_score = row.get("score")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            elif isinstance(raw_score, str):
                try:
                    score = float(raw_score.strip())
                except ValueError:
                    score = 1.0 / float(1 + pos)
            else:
                score = 1.0 / float(1 + pos)
            scores[idx] = score
        return scores or None

    def _try_cross_encoder_rerank_scores(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
        config: KnowledgeBaseConfig,
    ) -> dict[int, float] | None:
        model_name = (config.reranker_model or "").strip()
        if not model_name:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if model_name in _CROSS_ENCODER_FAILED_MODELS:
            return None
        try:
            model = self._get_or_load_cross_encoder(model_name)
            if model is None:
                return None
            pairs = [(query_text, hit.text) for hit in hits]
            raw_scores = model.predict(pairs)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scores: dict[int, float] = {}
            for idx, raw in enumerate(raw_scores):
                value = raw
                if isinstance(value, (list, tuple)) and value:
                    value = value[0]
                try:
                    scores[idx] = float(value)
                except (TypeError, ValueError):
                    continue
            return scores or None
        except (ImportError, OSError, RuntimeError, ValueError):
            _CROSS_ENCODER_FAILED_MODELS.add(model_name)
            return None

    @staticmethod
    def _get_cached_cross_encoder(model_name: str) -> object | None:
        with _CROSS_ENCODER_CACHE_LOCK:
            model = _CROSS_ENCODER_CACHE.get(model_name)
            if model is None:
                return None
            _CROSS_ENCODER_CACHE.pop(model_name, None)
            _CROSS_ENCODER_CACHE[model_name] = model
            return model

    def _get_or_load_cross_encoder(self, model_name: str) -> object | None:
        cached = self._get_cached_cross_encoder(model_name)
        if cached is not None:
            return cached

        should_load = False
        with _CROSS_ENCODER_CACHE_LOCK:
            model = _CROSS_ENCODER_CACHE.get(model_name)
            if model is not None:
                _CROSS_ENCODER_CACHE.pop(model_name, None)
                _CROSS_ENCODER_CACHE[model_name] = model
                return model
            event = _CROSS_ENCODER_LOADING_EVENTS.get(model_name)
            if event is None:
                event = threading.Event()
                _CROSS_ENCODER_LOADING_EVENTS[model_name] = event
                should_load = True

        if not should_load:
            event.wait()
            return self._get_cached_cross_encoder(model_name)

        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = CrossEncoder(model_name)
            with _CROSS_ENCODER_CACHE_LOCK:
                _CROSS_ENCODER_CACHE[model_name] = model
                while len(_CROSS_ENCODER_CACHE) > _CROSS_ENCODER_CACHE_LIMIT:
                    oldest = next(iter(_CROSS_ENCODER_CACHE))
                    if oldest == model_name and len(_CROSS_ENCODER_CACHE) == 1:
                        break
                    _CROSS_ENCODER_CACHE.pop(oldest, None)
            return model
        except (ImportError, OSError, RuntimeError, ValueError):
            _CROSS_ENCODER_FAILED_MODELS.add(model_name)
            return None
        finally:
            with _CROSS_ENCODER_CACHE_LOCK:
                event = _CROSS_ENCODER_LOADING_EVENTS.pop(model_name, None)
                if event is not None:
                    event.set()

    def _try_api_rerank_scores(
        self,
        query_text: str,
        hits: Sequence[KnowledgeBaseHit],
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> dict[int, float] | None:
        endpoint = (config.reranker_endpoint or "").strip()
        if not endpoint:
            return None

        payload = {
            "query": query_text,
            "top_k": top_k,
            "documents": [
                {
                    "index": idx,
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "text": hit.text,
                }
                for idx, hit in enumerate(hits)
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = (config.reranker_api_key or "").strip()
        if api_key:
            key_header = (config.reranker_api_key_header or "Authorization").strip()
            if not key_header:
                key_header = "Authorization"
            if key_header.lower() == "authorization" and not api_key.lower().startswith("bearer "):
                headers[key_header] = f"Bearer {api_key}"
            else:
                headers[key_header] = api_key

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )
        timeout_seconds = int(config.reranker_timeout_seconds or 30)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None

        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return self._parse_api_rerank_scores(response_payload, len(hits))

    @staticmethod
    def _parse_api_rerank_scores(payload: object, item_count: int) -> dict[int, float] | None:
        rows: list[object] | None = None
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("results", "data", "items", "ranked"):
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
        if not rows:
            return None

        scores: dict[int, float] = {}
        for pos, row in enumerate(rows):
            idx: int | None = None
            score: float | None = None
            if isinstance(row, dict):
                for key in ("index", "idx", "id", "document_index"):
                    raw_idx = row.get(key)
                    if isinstance(raw_idx, int):
                        idx = raw_idx
                        break
                    if isinstance(raw_idx, str) and raw_idx.strip().isdigit():
                        idx = int(raw_idx.strip())
                        break
                raw_score = row.get("score")
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                elif isinstance(raw_score, str):
                    try:
                        score = float(raw_score.strip())
                    except ValueError:
                        score = None
                if score is None:
                    raw_rank = row.get("rank")
                    if isinstance(raw_rank, (int, float)):
                        score = 1.0 / float(1 + int(raw_rank))
            elif isinstance(row, int):
                idx = row
            elif isinstance(row, str) and row.strip().isdigit():
                idx = int(row.strip())

            if idx is None:
                continue
            if idx < 0 or idx >= item_count:
                continue
            if score is None:
                score = 1.0 / float(1 + pos)
            scores[idx] = score
        return scores or None

    @staticmethod
    def _normalize_reranker_backend(value: str) -> str:
        backend = (value or "").strip().lower()
        if not backend:
            return "auto"
        if backend not in _RERANK_BACKEND_CHOICES:
            return "auto"
        return backend

    def _reranker_enabled(self, config: KnowledgeBaseConfig) -> bool:
        if not config.reranker_enabled:
            return False
        backend = self._normalize_reranker_backend(config.reranker_backend)
        return backend != "none"

    def _resolve_reranker_candidate_top_k(
        self,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> int:
        if not self._reranker_enabled(config):
            return top_k
        numeric = int(config.reranker_candidate_top_k)
        if numeric <= 0:
            numeric = 50
        if numeric > 200:
            numeric = 200
        return max(top_k, numeric)

    @staticmethod
    def _normalize_context_similarity_threshold(value: float) -> float:
        numeric = float(value)
        if numeric < 0.6:
            return 0.6
        if numeric > 0.99:
            return 0.99
        return numeric

    @staticmethod
    def _normalize_context_sentence_limit(value: int) -> int:
        numeric = int(value)
        if numeric < 1:
            return 1
        if numeric > 5:
            return 5
        return numeric

    @staticmethod
    def _normalize_context_chars_per_chunk(value: int) -> int:
        numeric = int(value)
        if numeric < 120:
            return 120
        if numeric > 1200:
            return 1200
        return numeric

    @staticmethod
    def _normalize_context_total_chars(value: int) -> int:
        numeric = int(value)
        if numeric < 300:
            return 300
        if numeric > 12000:
            return 12000
        return numeric

    @classmethod
    def _resolve_context_total_chars(
        cls,
        *,
        max_chunks: int,
        per_chunk_chars: int,
        configured_total_chars: int,
    ) -> int:
        base = cls._normalize_context_total_chars(configured_total_chars)
        chunk_count = max(1, int(max_chunks))
        per_chunk = cls._normalize_context_chars_per_chunk(per_chunk_chars)
        scaled = chunk_count * min(per_chunk, 900)
        return min(12000, max(base, scaled))

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        normalized = " ".join((text or "").split()).strip()
        if len(normalized) <= max_chars:
            return normalized
        head = normalized[: max(24, max_chars - 3)].rstrip()
        return f"{head}..."

    def _extract_key_sentences(
        self,
        text: str,
        query_terms: Sequence[str],
        max_sentences: int,
        max_chars: int,
    ) -> str:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return ""
        if len(compact) <= max_chars and max_sentences >= 2:
            return compact

        sentences = self._split_context_sentences(text)
        if not sentences:
            return self._clip_text(compact, max_chars)

        query_set = set(query_terms)
        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            terms = [item.lower() for item in _WORD_RE.findall(sentence)]
            if not terms:
                continue
            term_set = set(terms)
            overlap = 0
            if query_set:
                overlap = sum(1 for term in query_set if term in term_set)
            overlap_ratio = overlap / float(max(1, len(query_set)))
            density = float(overlap) / float(max(1, len(terms)))
            digits_bonus = 0.0
            if any(ch.isdigit() for ch in sentence):
                digits_bonus = 0.08
            position_bonus = 0.15 / float(idx + 1)
            length_penalty = 0.0
            if len(sentence) > 220:
                length_penalty = -0.05
            score = overlap_ratio * 2.8 + density * 0.7 + digits_bonus + position_bonus + length_penalty
            if not query_set:
                score = position_bonus + digits_bonus - max(0, len(sentence) - 180) / 5000.0
            scored.append((score, idx, sentence))

        if not scored:
            return self._clip_text(compact, max_chars)

        best = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)
        selected = sorted(best[:max_sentences], key=lambda item: item[1])
        assembled = " ".join(item[2] for item in selected).strip()
        if not assembled:
            assembled = compact
        return self._clip_text(assembled, max_chars)

    def _split_context_sentences(self, text: str) -> list[str]:
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            return []

        blocks = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
        sentences: list[str] = []
        for block in blocks:
            if self._looks_code_like(block):
                compact = " ".join(block.split()).strip()
                if compact:
                    sentences.append(compact)
                continue

            current: list[str] = []
            for idx, ch in enumerate(block):
                if ch == "\n":
                    if current and current[-1] != " ":
                        current.append(" ")
                    continue
                current.append(ch)
                if ch in "。！？!?":
                    self._flush_context_sentence(current, sentences)
                    continue
                if ch == "." and self._should_split_context_period(block, idx):
                    self._flush_context_sentence(current, sentences)
            self._flush_context_sentence(current, sentences)
        return sentences

    @staticmethod
    def _flush_context_sentence(buffer: list[str], sentences: list[str]) -> None:
        if not buffer:
            return
        sentence = " ".join("".join(buffer).split()).strip()
        buffer.clear()
        if sentence:
            sentences.append(sentence)

    @staticmethod
    def _looks_code_like(text: str) -> bool:
        return bool(_CONTEXT_CODE_HINT_RE.search(text or ""))

    @staticmethod
    def _should_split_context_period(text: str, idx: int) -> bool:
        prev_char = text[idx - 1] if idx > 0 else ""
        next_idx = idx + 1
        while next_idx < len(text) and text[next_idx].isspace():
            next_idx += 1
        next_char = text[next_idx] if next_idx < len(text) else ""

        if prev_char.isdigit() and next_char.isdigit():
            return False

        prev_word_chars: list[str] = []
        cursor = idx - 1
        while cursor >= 0 and text[cursor].isalpha():
            prev_word_chars.append(text[cursor].lower())
            cursor -= 1
        prev_word = "".join(reversed(prev_word_chars))
        if prev_word in _COMMON_ABBREVIATIONS:
            return False
        if next_char and next_char.islower():
            return False
        return True

    def _is_semantic_duplicate(
        self,
        text: str,
        vector: Sequence[float],
        existing_texts: Sequence[str],
        existing_vectors: Sequence[Sequence[float]],
        similarity_threshold: float,
    ) -> bool:
        normalized = " ".join(text.split()).strip().lower()
        candidate_tokens = {
            token
            for token in _WORD_RE.findall(normalized)
            if self._is_query_keyword(token)
        }
        for existing_text, existing_vector in zip(existing_texts, existing_vectors):
            normalized_existing = " ".join(existing_text.split()).strip().lower()
            if normalized == normalized_existing:
                return True
            if normalized in normalized_existing or normalized_existing in normalized:
                shorter = min(len(normalized), len(normalized_existing))
                longer = max(len(normalized), len(normalized_existing))
                if shorter > 0 and shorter / float(longer) >= 0.82:
                    return True
            existing_tokens = {
                token
                for token in _WORD_RE.findall(normalized_existing)
                if self._is_query_keyword(token)
            }
            if candidate_tokens and existing_tokens:
                union = candidate_tokens | existing_tokens
                if union:
                    jaccard = len(candidate_tokens & existing_tokens) / float(len(union))
                    if jaccard >= 0.86:
                        return True
            cosine = self._cosine_similarity(vector, existing_vector)
            if cosine >= similarity_threshold:
                return True
        return False

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
        norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / float(norm_a * norm_b)

    @staticmethod
    def _token_hash_vector(text: str, dims: int) -> list[float]:
        if dims <= 0:
            return []
        vector = [0.0] * dims
        tokens = [token.lower() for token in _WORD_RE.findall(text)]
        if not tokens:
            return vector
        for token in tokens:
            idx = KnowledgeBaseManager._stable_token_hash(token, dims)
            vector[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _stable_token_hash(token: str, mod: int) -> int:
        acc = 0
        for idx, ch in enumerate(token):
            acc += (idx + 1) * ord(ch)
        return acc % mod

    def _apply_context_char_budget(
        self,
        hits: Sequence[KnowledgeBaseHit],
        max_total_chars: int,
    ) -> list[KnowledgeBaseHit]:
        if not hits:
            return []
        if max_total_chars <= 0:
            return list(hits)

        output: list[KnowledgeBaseHit] = []
        used = 0
        for hit in hits:
            text = " ".join(hit.text.split()).strip()
            if not text:
                continue
            remaining = max_total_chars - used
            if remaining <= 0:
                break
            if len(text) <= remaining:
                output.append(replace(hit, text=text))
                used += len(text)
                continue
            if remaining < 80 and output:
                break
            clipped = self._clip_text(text, remaining)
            output.append(replace(hit, text=clipped))
            used += len(clipped)
            break
        if not output:
            first = hits[0]
            output = [replace(first, text=self._clip_text(first.text, max_total_chars))]
        return output

    @staticmethod
    def _resolve_multi_query_count(config: KnowledgeBaseConfig) -> int:
        if not config.multi_query_enabled:
            return 1
        count = int(config.multi_query_count)
        if count < 3:
            return 3
        if count > 5:
            return 5
        return count

    def _resolve_multi_query_candidate_top_k(
        self,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> int:
        configured = self._normalize_candidate_top_k(config.multi_query_candidate_top_k, top_k)
        if configured is not None:
            return configured
        return self._scaled_candidate_top_k(top_k, factor=3.0, extra=2, cap=128)

    @staticmethod
    def _normalize_rrf_k(value: int) -> int:
        numeric = int(value)
        if numeric < 10:
            return 10
        if numeric > 200:
            return 200
        return numeric

    @staticmethod
    def _expand_query_variants(query_text: str, target_count: int) -> list[str]:
        text = query_text.strip()
        if not text:
            return []
        if target_count <= 1:
            return [text]

        variants: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            value = " ".join(candidate.split()).strip()
            if not value:
                return
            marker = value.lower()
            if marker in seen:
                return
            seen.add(marker)
            variants.append(value)

        tokens = _WORD_RE.findall(text)
        token_line = " ".join(tokens)
        keyword_tokens = [token for token in tokens if KnowledgeBaseManager._is_query_keyword(token)]

        _add(text)
        _add(token_line)
        if keyword_tokens:
            _add(" ".join(keyword_tokens[:10]))
            _add(" OR ".join(keyword_tokens[:8]))
            _add(" ".join(f'"{token}"' for token in keyword_tokens[:6]))

        if len(tokens) >= 4:
            _add(" ".join(tokens[: min(6, len(tokens))]))
            _add(" ".join(tokens[-min(6, len(tokens)) :]))

        for token in keyword_tokens[:6]:
            _add(token)
        for idx in range(0, max(0, len(keyword_tokens) - 1)):
            pair = f"{keyword_tokens[idx]} {keyword_tokens[idx + 1]}"
            _add(pair)

        if len(variants) < target_count:
            seed = keyword_tokens[0] if keyword_tokens else (tokens[0] if tokens else text)
            _add(f'"{seed}"')
            _add(f"{seed} OR {seed}")

        return variants[:target_count]

    @staticmethod
    def _is_query_keyword(token: str) -> bool:
        value = token.strip()
        if not value:
            return False
        if re.fullmatch(r"[A-Za-z0-9_]+", value):
            lowered = value.lower()
            if lowered in _EN_STOPWORDS:
                return False
            return len(value) > 1
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            if value in _CN_STOPWORDS:
                return False
            return len(value) >= 2
        return True

    def _build_query_plan(
        self,
        query_text: str,
        top_k: int,
        mode: str,
        config: KnowledgeBaseConfig,
    ) -> KnowledgeBaseQueryPlan:
        if mode == "auto":
            return self._build_auto_query_plan(query_text, top_k, config)
        if mode == "hybrid":
            return KnowledgeBaseQueryPlan(
                requested_mode=mode,
                resolved_mode=mode,
                query_type="manual",
                alpha=self._clamp(config.hybrid_alpha, 0.0, 1.0),
                vector_top_k=self._normalize_candidate_top_k(config.vector_top_k, top_k),
                fts_top_k=self._normalize_candidate_top_k(config.fts_top_k, top_k),
            )
        return KnowledgeBaseQueryPlan(
            requested_mode=mode,
            resolved_mode=mode,
            query_type="manual",
        )

    def _build_auto_query_plan(
        self,
        query_text: str,
        top_k: int,
        config: KnowledgeBaseConfig,
    ) -> KnowledgeBaseQueryPlan:
        if not self._fts_available(config):
            return KnowledgeBaseQueryPlan(
                requested_mode="auto",
                resolved_mode="vector",
                query_type="fts-unavailable",
            )

        keyword_score, semantic_score = self._query_signal_scores(query_text)
        if keyword_score >= 4 and semantic_score <= 1:
            return KnowledgeBaseQueryPlan(
                requested_mode="auto",
                resolved_mode="fts",
                query_type="keyword",
            )
        if semantic_score >= 4 and keyword_score <= 1:
            return KnowledgeBaseQueryPlan(
                requested_mode="auto",
                resolved_mode="vector",
                query_type="semantic",
            )

        bias = semantic_score - keyword_score
        base_alpha = self._clamp(config.hybrid_alpha, 0.0, 1.0)
        if bias >= 2:
            query_type = "semantic-heavy"
            alpha = self._clamp(base_alpha + 0.18, 0.55, 0.82)
            vector_top_k = self._scaled_candidate_top_k(top_k, factor=3.0, extra=2, cap=96)
            fts_top_k = self._scaled_candidate_top_k(top_k, factor=1.8, extra=1, cap=64)
        elif bias <= -2:
            query_type = "keyword-heavy"
            alpha = self._clamp(base_alpha - 0.18, 0.18, 0.45)
            vector_top_k = self._scaled_candidate_top_k(top_k, factor=1.8, extra=1, cap=64)
            fts_top_k = self._scaled_candidate_top_k(top_k, factor=3.0, extra=2, cap=96)
        else:
            query_type = "balanced"
            alpha = self._clamp(base_alpha, 0.35, 0.65)
            vector_top_k = self._scaled_candidate_top_k(top_k, factor=2.4, extra=2, cap=80)
            fts_top_k = self._scaled_candidate_top_k(top_k, factor=2.4, extra=2, cap=80)

        return KnowledgeBaseQueryPlan(
            requested_mode="auto",
            resolved_mode="hybrid",
            query_type=query_type,
            alpha=alpha,
            vector_top_k=vector_top_k,
            fts_top_k=fts_top_k,
        )

    @staticmethod
    def _fts_available(config: KnowledgeBaseConfig) -> bool:
        return bool(config.enable_fts) and config.store != "duckdb-vss"

    @staticmethod
    def _query_signal_scores(query_text: str) -> tuple[int, int]:
        text = query_text.strip()
        lower = text.lower()
        token_count = len(_WORD_RE.findall(text))
        char_count = len(text)

        has_quotes = any(char in text for char in ('"', "'", "`"))
        has_path_hint = bool(_PATH_HINT_RE.search(text))
        has_code_hint = bool(_CODE_HINT_RE.search(text))
        has_error_hint = bool(_ERROR_HINT_RE.search(lower))
        has_keyword_ops = bool(_KEYWORD_OP_RE.search(lower))
        has_semantic_hint = bool(_SEMANTIC_HINT_RE.search(lower))
        short_query = token_count <= 5 or char_count <= 24
        long_query = token_count >= 12 or char_count >= 64

        keyword_score = 0
        if has_quotes:
            keyword_score += 2
        if has_path_hint:
            keyword_score += 2
        if has_code_hint:
            keyword_score += 2
        if has_error_hint:
            keyword_score += 1
        if has_keyword_ops:
            keyword_score += 1
        if short_query:
            keyword_score += 1

        semantic_score = 0
        if has_semantic_hint:
            semantic_score += 2
        if long_query:
            semantic_score += 2
        if token_count >= 8 and not has_code_hint:
            semantic_score += 1
        if ("?" in text or "？" in text) and token_count >= 6 and not has_code_hint:
            semantic_score += 1

        return keyword_score, semantic_score

    @staticmethod
    def _normalize_candidate_top_k(value: int | None, top_k: int) -> int | None:
        if value is None:
            return None
        numeric = int(value)
        if numeric <= 0:
            return None
        return max(top_k, numeric)

    @staticmethod
    def _scaled_candidate_top_k(top_k: int, factor: float, extra: int, cap: int) -> int:
        scaled = int(round(top_k * factor)) + extra
        return min(cap, max(top_k, scaled))

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    def _build_store(
        self,
        config: KnowledgeBaseConfig,
        embedding_dim: int | None = None,
        *,
        use_config_default: bool = True,
    ) -> object:
        table = self._table_name(config)
        resolved_dim = embedding_dim
        if resolved_dim is None and use_config_default:
            resolved_dim = config.dims
        if config.store == "sqlite-vec1":
            return SqliteVec1Store(
                path=config.db_path,
                table=table,
                embedding_dim=resolved_dim,
                distance_metric=config.distance_metric,
                load_extension=not config.disable_sqlite_extension,
                extension_path=config.sqlite_extension_path,
                extension_whitelist=list(config.extension_whitelist or []),
            )
        if config.store == "sqlite-vec":
            return SqliteVecStore(
                path=config.db_path,
                table=table,
                embedding_dim=resolved_dim,
                distance_metric=config.distance_metric,
            )
        if config.store == "duckdb-vss":
            return DuckDbVssStore(
                path=config.db_path,
                table=table,
                embedding_dim=resolved_dim,
                distance_metric=config.distance_metric,
                enable_vss=not config.disable_vss_extension,
                persistent_index=config.vss_persistent_index,
                fail_if_no_vss=False,
            )
        raise ValueError(f"unsupported store backend: {config.store}")

    @staticmethod
    def _resolved_embedding_dims(config: KnowledgeBaseConfig, embedder: object) -> int:
        dims = embedder_dims(embedder)
        if dims is not None:
            return dims
        if config.dims <= 0:
            raise ValueError("embedding dims must be positive")
        return config.dims

    @staticmethod
    def _build_embedder(config: KnowledgeBaseConfig) -> object:
        provider = (config.embedding_provider or "auto").strip().lower()
        if provider in {"auto", "fastembed"}:
            try:
                importlib.import_module("fastembed")
                return FastEmbedder(
                    model_name=(config.embedding_model or DEFAULT_FASTEMBED_MODEL).strip()
                    or DEFAULT_FASTEMBED_MODEL,
                    batch_size=max(1, int(config.embedding_batch_size or 256)),
                )
            except ImportError:
                if provider == "fastembed":
                    raise RuntimeError(
                        "fastembed is not installed. Install with `pip install fastembed`."
                    )
        if provider in {"auto", "hashing", "local"}:
            return HashingEmbedder(dims=config.dims)
        raise ValueError(f"unsupported embedding provider: {config.embedding_provider}")

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
    def _build_chunker(config: KnowledgeBaseConfig) -> FixedChunker | RecursiveChunker | StructureAwareChunker:
        if config.chunker == "structured":
            return StructureAwareChunker(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
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

    @staticmethod
    def _fts_enabled_for_config(config: KnowledgeBaseConfig) -> bool:
        return bool(config.enable_fts) and config.store != "duckdb-vss"

    def _recover_pending_fts_consistency(self, config: KnowledgeBaseConfig) -> None:
        if not self._fts_enabled_for_config(config):
            return
        marker_path = self._pending_fts_recovery_path(config.db_path)
        recovery = self._load_pending_fts_recovery(marker_path)
        if recovery is None:
            return

        recovery_config = replace(
            config,
            store=recovery.store,
            table=recovery.table,
            fts_table=recovery.fts_table,
            enable_fts=True,
        )
        store = self._build_store(
            recovery_config,
            embedding_dim=None,
            use_config_default=False,
        )
        fts = SqliteFtsIndex(
            path=recovery_config.db_path,
            table=self._safe_identifier(recovery.fts_table, label="fts table"),
        )
        try:
            if not hasattr(store, "load_chunks_by_doc_ids"):
                raise RuntimeError(
                    f"store backend does not support FTS recovery: {recovery.store}"
                )
            chunks = list(store.load_chunks_by_doc_ids(recovery.doc_ids))
            fts.replace_by_doc_ids(recovery.doc_ids, chunks)
        finally:
            fts.close()
            self._close_store(store)
        self._clear_pending_fts_recovery(config)

    def _write_pending_fts_recovery(
        self,
        config: KnowledgeBaseConfig,
        doc_ids: Sequence[str],
    ) -> None:
        ids = self._normalize_doc_ids(doc_ids)
        if not ids:
            self._clear_pending_fts_recovery(config)
            return
        payload = {
            "version": _PENDING_FTS_RECOVERY_VERSION,
            "store": config.store,
            "table": self._table_name(config),
            "fts_table": self._safe_identifier(config.fts_table, label="fts table"),
            "doc_ids": ids,
        }
        write_text_atomic(
            self._pending_fts_recovery_path(config.db_path),
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _clear_pending_fts_recovery(self, config: KnowledgeBaseConfig) -> None:
        marker_path = self._pending_fts_recovery_path(config.db_path)
        try:
            marker_path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _pending_fts_recovery_path(db_path: str) -> Path:
        target = Path(db_path)
        return target.parent / f".{target.name}.fts-recovery.json"

    @classmethod
    def _load_pending_fts_recovery(cls, marker_path: Path) -> _PendingFtsRecovery | None:
        if not marker_path.exists():
            return None
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"pending KB recovery marker is corrupted: {marker_path}"
            ) from exc

        if payload.get("version") != _PENDING_FTS_RECOVERY_VERSION:
            raise RuntimeError(
                f"unsupported KB recovery marker version in {marker_path}"
            )
        store = str(payload.get("store", "")).strip()
        table = cls._safe_identifier(str(payload.get("table", "")).strip(), label="vector table")
        fts_table = cls._safe_identifier(
            str(payload.get("fts_table", "")).strip(),
            label="fts table",
        )
        doc_ids = cls._normalize_doc_ids(payload.get("doc_ids", []))
        if not store:
            raise RuntimeError(f"pending KB recovery marker is missing store: {marker_path}")
        if not doc_ids:
            raise RuntimeError(f"pending KB recovery marker has no doc_ids: {marker_path}")
        return _PendingFtsRecovery(
            store=store,
            table=table,
            fts_table=fts_table,
            doc_ids=tuple(doc_ids),
        )

    @staticmethod
    def _normalize_doc_ids(doc_ids: Sequence[str] | object) -> list[str]:
        if isinstance(doc_ids, (str, bytes, bytearray)):
            return []
        try:
            items = list(doc_ids)
        except TypeError:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            value = str(item).strip()
            if not value:
                continue
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

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
            except duckdb.Error:
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
            except duckdb.Error:
                return []
            return [str(row[0]) for row in rows if row and row[0]]
        finally:
            conn.close()
