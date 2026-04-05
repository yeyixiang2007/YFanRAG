"""Knowledge-base management window and helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence
import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, filedialog, ttk
from tkinter import scrolledtext

from ...knowledge_base import (
    QUERY_MODE_CHOICES,
    STORE_CHOICES,
    KnowledgeBaseConfig,
    KnowledgeBaseHit,
    KnowledgeBaseQueryPlan,
)
from ...loaders.text import DEFAULT_TEXT_EXTENSIONS


class AppKnowledgeBaseMixin:
    def _open_kb_window(self) -> None:
        if self.kb_window is not None and self.kb_window.winfo_exists():
            self.kb_window.deiconify()
            self.kb_window.lift()
            self.kb_window.focus_set()
            return

        window = tk.Toplevel(self.root)
        window.title("Knowledge Base Manager")
        window.geometry("1040x760")
        window.minsize(900, 620)
        window.configure(bg="#0B1020")
        window.protocol("WM_DELETE_WINDOW", self._close_kb_window)
        self.kb_window = window

        outer = ttk.Frame(window)
        outer.pack(fill=BOTH, expand=True, padx=10, pady=10)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        config_card = ttk.Frame(outer, style="Card.TFrame")
        config_card.grid(row=0, column=0, sticky="ew")
        config_card.columnconfigure(1, weight=1)
        config_card.columnconfigure(4, weight=1)
        config_card.columnconfigure(5, weight=0)

        ttk.Label(config_card, text="Knowledge Base", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 4),
        )
        ttk.Label(config_card, text="Database", style="Section.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=4,
        )
        ttk.Entry(config_card, textvariable=self.kb_db_var).grid(
            row=1,
            column=1,
            columnspan=4,
            sticky="ew",
            padx=6,
            pady=4,
        )
        ttk.Button(
            config_card,
            text="Browse",
            command=self._select_kb_db,
        ).grid(row=1, column=5, sticky="ew", padx=(6, 10), pady=4)

        ttk.Label(config_card, text="Store", style="Section.TLabel").grid(
            row=2,
            column=0,
            sticky="w",
            padx=10,
            pady=4,
        )
        ttk.Combobox(
            config_card,
            textvariable=self.kb_store_var,
            values=list(STORE_CHOICES),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(config_card, text="Embedding Dims", style="Section.TLabel").grid(
            row=2,
            column=2,
            sticky="w",
            padx=(10, 6),
            pady=4,
        )
        ttk.Entry(config_card, textvariable=self.kb_dims_var, width=8).grid(
            row=2,
            column=3,
            sticky="w",
            padx=6,
            pady=4,
        )
        ttk.Checkbutton(
            config_card,
            text="Enable FTS",
            variable=self.kb_enable_fts_var,
            style="Tool.TCheckbutton",
        ).grid(row=2, column=5, sticky="w", padx=(6, 10), pady=4)

        ttk.Label(config_card, text="Chunker", style="Section.TLabel").grid(
            row=3,
            column=0,
            sticky="w",
            padx=10,
            pady=4,
        )
        ttk.Combobox(
            config_card,
            textvariable=self.kb_chunker_var,
            values=["fixed", "recursive"],
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(config_card, text="Chunk Size", style="Section.TLabel").grid(
            row=3,
            column=2,
            sticky="w",
            padx=(10, 6),
            pady=4,
        )
        ttk.Entry(config_card, textvariable=self.kb_chunk_size_var, width=8).grid(
            row=3,
            column=3,
            sticky="w",
            padx=6,
            pady=4,
        )
        ttk.Label(config_card, text="Overlap", style="Section.TLabel").grid(
            row=3,
            column=4,
            sticky="w",
            padx=(6, 10),
            pady=4,
        )
        ttk.Entry(config_card, textvariable=self.kb_chunk_overlap_var, width=8).grid(
            row=3,
            column=5,
            sticky="w",
            padx=(0, 10),
            pady=4,
        )

        mode_row = ttk.Frame(config_card, style="Card.TFrame")
        mode_row.grid(row=4, column=0, columnspan=6, sticky="ew", padx=10, pady=(6, 6))
        mode_row.columnconfigure(7, weight=1)
        ttk.Label(mode_row, text="Query Mode", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Combobox(
            mode_row,
            textvariable=self.kb_query_mode_var,
            values=list(QUERY_MODE_CHOICES),
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=(8, 10))
        ttk.Checkbutton(
            mode_row,
            text="Use KB Context In Chat",
            variable=self.kb_use_context_var,
            style="Tool.TCheckbutton",
        ).grid(row=0, column=2, sticky="w")
        ttk.Label(mode_row, text="Context TopK", style="Section.TLabel").grid(
            row=0,
            column=3,
            sticky="w",
            padx=(14, 6),
        )
        ttk.Entry(mode_row, textvariable=self.kb_context_top_k_var, width=6).grid(
            row=0,
            column=4,
            sticky="w",
        )
        ttk.Button(mode_row, text="Refresh Stats", command=self._kb_refresh_stats).grid(
            row=0,
            column=5,
            sticky="w",
            padx=(14, 8),
        )
        ttk.Button(mode_row, text="List Doc IDs", command=self._kb_list_docs).grid(
            row=0,
            column=6,
            sticky="w",
        )

        ttk.Label(config_card, textvariable=self.kb_stats_var, style="Meta.TLabel").grid(
            row=5,
            column=0,
            columnspan=6,
            sticky="w",
            padx=10,
            pady=(0, 10),
        )

        ingest_card = ttk.Frame(outer, style="Card.TFrame")
        ingest_card.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ingest_card.columnconfigure(0, weight=1)
        ttk.Label(ingest_card, text="Ingest Paths (one per line)", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 4),
        )
        self.kb_paths_text = tk.Text(
            ingest_card,
            height=4,
            bg="#1A253D",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap="word",
            relief="flat",
            font=("Segoe UI", 10),
            padx=8,
            pady=8,
            takefocus=True,
        )
        self.kb_paths_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        ingest_actions = ttk.Frame(ingest_card, style="Card.TFrame")
        ingest_actions.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        ttk.Button(ingest_actions, text="Add Files", command=self._add_kb_files).pack(
            side=LEFT
        )
        ttk.Button(ingest_actions, text="Add Folder", command=self._add_kb_folder).pack(
            side=LEFT,
            padx=(8, 0),
        )
        ttk.Button(ingest_actions, text="Clear Paths", command=self._clear_kb_paths).pack(
            side=LEFT,
            padx=(8, 0),
        )
        ttk.Button(
            ingest_actions,
            text="Ingest / Upsert",
            style="Accent.TButton",
            command=self._kb_ingest_paths,
        ).pack(side=RIGHT)

        query_card = ttk.Frame(outer, style="Card.TFrame")
        query_card.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        query_card.columnconfigure(1, weight=1)
        ttk.Label(query_card, text="KB Query", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 6),
        )
        ttk.Entry(query_card, textvariable=self.kb_query_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6,
            pady=(10, 6),
        )
        ttk.Label(query_card, text="TopK", style="Section.TLabel").grid(
            row=0,
            column=2,
            sticky="w",
            padx=(8, 4),
            pady=(10, 6),
        )
        ttk.Entry(query_card, textvariable=self.kb_top_k_var, width=6).grid(
            row=0,
            column=3,
            sticky="w",
            padx=4,
            pady=(10, 6),
        )
        ttk.Button(query_card, text="Use Prompt", command=self._kb_use_current_prompt).grid(
            row=0,
            column=4,
            sticky="w",
            padx=(8, 4),
            pady=(10, 6),
        )
        ttk.Button(query_card, text="Run Query", command=self._kb_query_preview).grid(
            row=0,
            column=5,
            sticky="w",
            padx=(6, 10),
            pady=(10, 6),
        )

        ttk.Label(query_card, text="Delete Doc ID(s)", style="Section.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=10,
            pady=(0, 10),
        )
        ttk.Entry(query_card, textvariable=self.kb_doc_id_var).grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=6,
            pady=(0, 10),
        )
        ttk.Button(query_card, text="Delete", style="Danger.TButton", command=self._kb_delete_doc_ids).grid(
            row=1,
            column=5,
            sticky="w",
            padx=(6, 10),
            pady=(0, 10),
        )

        log_card = ttk.Frame(outer, style="Card.TFrame")
        log_card.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)
        ttk.Label(log_card, text="Knowledge Base Logs / Results", style="Section.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(10, 4),
        )
        self.kb_log_text = scrolledtext.ScrolledText(
            log_card,
            wrap="word",
            bg="#0E1628",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            relief="flat",
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        self.kb_log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.kb_log_text.tag_configure("info", foreground="#93C5FD")
        self.kb_log_text.tag_configure("warn", foreground="#FDE68A")
        self.kb_log_text.tag_configure("error", foreground="#FCA5A5")
        self.kb_log_text.tag_configure("result", foreground="#A7F3D0")
        self.kb_log_text.configure(state="disabled")

        self._kb_append_log("info", "Knowledge base manager ready.")
        self._kb_refresh_stats()

    def _close_kb_window(self) -> None:
        if self.kb_window is None:
            return
        try:
            self.kb_window.destroy()
        finally:
            self.kb_window = None
            self.kb_paths_text = None
            self.kb_log_text = None

    def _select_kb_db(self) -> None:
        current = self.kb_db_var.get().strip() or "yfanrag_kb.db"
        selected = filedialog.asksaveasfilename(
            title="Select knowledge base database file",
            defaultextension=".db",
            initialfile=Path(current).name,
            filetypes=[
                ("Database files", "*.db *.sqlite *.duckdb"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.kb_db_var.set(selected)
            self._kb_refresh_stats()

    def _add_kb_files(self) -> None:
        allowed_patterns = " ".join(
            f"*{suffix}" for suffix in sorted({ext.lower() for ext in DEFAULT_TEXT_EXTENSIONS})
        )
        selected = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[
                ("Text / Markdown / Code", allowed_patterns),
                ("All files", "*.*"),
            ],
        )
        self._append_kb_paths(selected)

    def _add_kb_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select document folder")
        if selected:
            self._append_kb_paths([selected])

    def _append_kb_paths(self, paths: Sequence[str]) -> None:
        if self.kb_paths_text is None:
            return
        existing = set(self._read_kb_paths())
        to_add = [item for item in paths if item and item not in existing]
        if not to_add:
            return
        current_text = self.kb_paths_text.get("1.0", END).strip()
        lines = [line for line in current_text.splitlines() if line.strip()]
        lines.extend(to_add)
        self.kb_paths_text.delete("1.0", END)
        self.kb_paths_text.insert("1.0", "\n".join(lines) + "\n")

    def _clear_kb_paths(self) -> None:
        if self.kb_paths_text is None:
            return
        self.kb_paths_text.delete("1.0", END)

    def _read_kb_paths(self) -> list[str]:
        if self.kb_paths_text is None:
            return []
        lines = self.kb_paths_text.get("1.0", END).splitlines()
        paths: list[str] = []
        for line in lines:
            value = line.strip()
            if value:
                paths.append(value)
        return paths

    def _collect_kb_config(self) -> KnowledgeBaseConfig:
        db_path = self.kb_db_var.get().strip()
        if not db_path:
            raise ValueError("database path cannot be empty")
        store = self.kb_store_var.get().strip()
        if store not in STORE_CHOICES:
            raise ValueError(f"unsupported store: {store}")

        dims = int(self.kb_dims_var.get().strip() or "8")
        chunk_size = int(self.kb_chunk_size_var.get().strip() or "800")
        chunk_overlap = int(self.kb_chunk_overlap_var.get().strip() or "120")
        chunker = self.kb_chunker_var.get().strip() or "recursive"
        if chunker not in {"fixed", "recursive"}:
            raise ValueError(f"unsupported chunker: {chunker}")

        return KnowledgeBaseConfig(
            db_path=db_path,
            store=store,
            dims=dims,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_fts=bool(self.kb_enable_fts_var.get()),
            disable_sqlite_extension=True,
        )

    def _kb_append_log(self, tone: str, text: str) -> None:
        if self.kb_log_text is None:
            return
        widget = self.kb_log_text
        widget.configure(state="normal")
        tag = tone if tone in {"info", "warn", "error", "result"} else "info"
        prefix = tag.upper()
        widget.insert(END, f"[{prefix}] {text}\n", (tag,))
        widget.configure(state="disabled")
        widget.see(END)

    def _kb_refresh_stats(self) -> None:
        try:
            config = self._collect_kb_config()
            stats = self.kb_manager.stats(config)
            db_name = Path(stats.db_path).name
            self.kb_stats_var.set(
                f"KB stats: db={db_name} store={stats.store} table={stats.table} "
                f"docs={stats.doc_count} chunks={stats.chunk_count}"
            )
        except Exception as exc:
            self.kb_stats_var.set(f"KB stats unavailable: {exc}")

    def _kb_list_docs(self) -> None:
        try:
            config = self._collect_kb_config()
            doc_ids = self.kb_manager.list_doc_ids(config, limit=200)
        except Exception as exc:
            self._kb_append_log("error", f"List docs failed: {exc}")
            return
        if not doc_ids:
            self._kb_append_log("warn", "No doc_id found in current knowledge base.")
            return
        self._kb_append_log("info", f"Found {len(doc_ids)} doc_id values:")
        for doc_id in doc_ids:
            self._kb_append_log("result", doc_id)

    def _kb_ingest_paths(self) -> None:
        paths = self._read_kb_paths()
        try:
            config = self._collect_kb_config()
            result = self.kb_manager.ingest_paths(paths, config)
        except Exception as exc:
            self._kb_append_log("error", f"Ingest failed: {exc}")
            return

        self._kb_append_log(
            "info",
            f"Ingested {result.document_count} documents, generated {result.chunk_count} chunks.",
        )
        preview_doc_ids = result.doc_ids[:8]
        for doc_id in preview_doc_ids:
            self._kb_append_log("result", f"upserted: {doc_id}")
        if len(result.doc_ids) > len(preview_doc_ids):
            self._kb_append_log(
                "info",
                f"... and {len(result.doc_ids) - len(preview_doc_ids)} more documents.",
            )
        self._kb_refresh_stats()

    def _kb_delete_doc_ids(self) -> None:
        raw = self.kb_doc_id_var.get().strip()
        doc_ids = [item for item in re.split(r"[,;\s]+", raw) if item]
        if not doc_ids:
            self._kb_append_log("warn", "Enter one or more doc_id values before delete.")
            return
        try:
            config = self._collect_kb_config()
            result = self.kb_manager.delete_doc_ids(doc_ids, config)
        except Exception as exc:
            self._kb_append_log("error", f"Delete failed: {exc}")
            return
        self._kb_append_log(
            "info",
            f"Deleted docs={len(result.doc_ids)} vector_rows={result.vector_deleted} "
            f"fts_rows={result.fts_deleted}",
        )
        self.kb_doc_id_var.set("")
        self._kb_refresh_stats()

    def _kb_query_preview(self) -> None:
        query_text = self.kb_query_var.get().strip()
        if not query_text:
            self._kb_append_log("warn", "Enter a KB query text.")
            return
        try:
            top_k = int(self.kb_top_k_var.get().strip() or "3")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            config = self._collect_kb_config()
            mode = self.kb_query_mode_var.get().strip() or "auto"
            if config.store == "duckdb-vss" and mode in {"hybrid", "fts"}:
                mode = "vector"
                self._kb_append_log(
                    "warn",
                    "duckdb-vss currently runs vector mode only in this manager.",
                )
            hits = self.kb_manager.query(
                query_text=query_text,
                top_k=top_k,
                mode=mode,
                config=config,
            )
        except Exception as exc:
            self._kb_append_log("error", f"Query failed: {exc}")
            return

        plan = self.kb_manager.last_query_plan
        if plan is not None and (plan.requested_mode == "auto" or plan.fusion is not None):
            self._kb_append_log("info", f"Query plan: {self._kb_plan_summary(plan)}")
        if not hits:
            self._kb_append_log("warn", "No retrieval results.")
            return
        self._kb_append_log("info", f"Returned {len(hits)} hits:")
        for hit in hits:
            snippet = " ".join(hit.text.split())
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            self._kb_append_log(
                "result",
                f"{hit.rank}. {hit.doc_id} | {self._kb_score_text(hit)}",
            )
            self._kb_append_log("result", f"    {snippet}")

    def _kb_use_current_prompt(self) -> None:
        text = self.input_text.get("1.0", END).strip()
        if not text:
            return
        self.kb_query_var.set(text)
        self._kb_append_log("info", "Loaded current prompt into KB query box.")

    @staticmethod
    def _kb_score_text(hit: KnowledgeBaseHit) -> str:
        prefix = ""
        if hit.rerank_score is not None:
            prefix += f"rerank={hit.rerank_score:.4f} "
        if hit.rrf_score is not None:
            prefix += f"rrf={hit.rrf_score:.4f} "
        if hit.source == "hybrid":
            fused = 0.0 if hit.score is None else hit.score
            vector_score = 0.0 if hit.vector_score is None else hit.vector_score
            fts_score = 0.0 if hit.fts_score is None else hit.fts_score
            body = (
                f"fused={fused:.4f} "
                f"vec={vector_score:.4f} fts={fts_score:.4f}"
            )
            return f"{prefix}{body}".strip()
        if hit.source == "vector":
            if hit.distance is None:
                body = "vector"
            else:
                body = f"distance={hit.distance:.4f}"
            return f"{prefix}{body}".strip()
        if hit.score is None:
            body = "fts"
        else:
            body = f"bm25={hit.score:.4f}"
        return f"{prefix}{body}".strip()

    @staticmethod
    def _kb_plan_summary(plan: KnowledgeBaseQueryPlan) -> str:
        parts = [f"{plan.requested_mode}->{plan.resolved_mode}", f"type={plan.query_type}"]
        if plan.alpha is not None:
            parts.append(f"alpha={plan.alpha:.2f}")
        if plan.vector_top_k is not None:
            parts.append(f"vec_k={plan.vector_top_k}")
        if plan.fts_top_k is not None:
            parts.append(f"fts_k={plan.fts_top_k}")
        if plan.fusion is not None:
            mq_count = len(plan.query_variants) if plan.query_variants else 1
            if plan.rrf_k is None:
                parts.append(f"fusion={plan.fusion}({mq_count})")
            else:
                parts.append(f"fusion={plan.fusion}({mq_count},k={plan.rrf_k})")
        if plan.candidate_top_k is not None:
            parts.append(f"cand_k={plan.candidate_top_k}")
        if plan.reranker_backend is not None:
            if plan.reranker_top_k is None:
                parts.append(f"rerank={plan.reranker_backend}")
            else:
                parts.append(f"rerank={plan.reranker_backend}(top={plan.reranker_top_k})")
        if plan.reranker_candidate_top_k is not None:
            parts.append(f"rerank_cand={plan.reranker_candidate_top_k}")
        return ", ".join(parts)

    def _build_kb_context_for_user_text(self, user_text: str) -> tuple[str, str]:
        if not self.kb_use_context_var.get():
            return user_text, ""

        mode = self.kb_query_mode_var.get().strip() or "auto"
        top_k = int(self.kb_context_top_k_var.get().strip() or "3")
        if top_k <= 0:
            raise ValueError("context top_k must be positive")
        config = self._collect_kb_config()
        if config.store == "duckdb-vss" and mode in {"hybrid", "fts"}:
            mode = "vector"

        hits = self.kb_manager.query(
            query_text=user_text,
            top_k=top_k,
            mode=mode,
            config=config,
        )
        if not hits:
            return user_text, "KB context enabled, but no matching chunks found."

        plan = self.kb_manager.last_query_plan
        mode_note = mode
        if plan is not None and (plan.requested_mode == "auto" or plan.fusion is not None):
            mode_note = self._kb_plan_summary(plan)
        context_block = self._format_kb_context(hits)
        payload = (
            "Knowledge Base Context:\n"
            "Use the excerpts below when relevant. If unrelated, ignore them.\n\n"
            f"{context_block}\n\n"
            f"User Question:\n{user_text}"
        )
        note = f"Attached {len(hits)} KB chunks ({mode_note}) from {Path(config.db_path).name}."
        return payload, note

    @staticmethod
    def _format_kb_context(hits: Sequence[KnowledgeBaseHit]) -> str:
        blocks: list[str] = []
        for hit in hits:
            snippet = " ".join(hit.text.split())
            if len(snippet) > 360:
                snippet = snippet[:357] + "..."
            blocks.append(
                f"[{hit.rank}] doc_id={hit.doc_id} chunk_id={hit.chunk_id}\n{snippet}"
            )
        return "\n\n".join(blocks)
