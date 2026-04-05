"""Knowledge base service helpers for UI and tooling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    multi_query_enabled: bool = True
    multi_query_count: int = 4
    multi_query_rrf_k: int = 60
    multi_query_candidate_top_k: int | None = None
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
            self._last_query_plan = None
            return []
        if top_k <= 0:
            self._last_query_plan = None
            return []
        requested_mode = (mode or "").strip().lower()
        if requested_mode not in QUERY_MODE_CHOICES:
            raise ValueError(f"unsupported query mode: {mode}")

        plan = self._build_query_plan(text, top_k, requested_mode, config)
        variant_count = self._resolve_multi_query_count(config)
        variants = self._expand_query_variants(text, variant_count)
        if not variants:
            variants = [text]
        if len(variants) == 1:
            final_plan = replace(plan, query_variants=tuple(variants))
            self._last_query_plan = final_plan
            return self._execute_query_by_plan(variants[0], top_k, final_plan, config)

        candidate_top_k = self._resolve_multi_query_candidate_top_k(top_k, config)
        rrf_k = self._normalize_rrf_k(config.multi_query_rrf_k)
        runs: list[list[KnowledgeBaseHit]] = []
        for variant in variants:
            hits = self._execute_query_by_plan(variant, candidate_top_k, plan, config)
            if hits:
                runs.append(hits)
        final_plan = replace(
            plan,
            query_variants=tuple(variants),
            fusion="rrf",
            rrf_k=rrf_k,
            candidate_top_k=candidate_top_k,
        )
        self._last_query_plan = final_plan
        if not runs:
            return []
        return self._fuse_rrf_hits(
            runs=runs,
            top_k=top_k,
            rrf_k=rrf_k,
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
