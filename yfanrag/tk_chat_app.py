"""Modern Tkinter chat UI with streaming and multi-provider API compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, filedialog, ttk
from tkinter import scrolledtext
import re
import threading
import tkinter as tk
from typing import Sequence

from .chat_providers import (
    ChatApiClient,
    ChatApiSettings,
    ChatMessage,
    PROVIDER_PRESETS,
    parse_json_dict,
)
from .knowledge_base import (
    QUERY_MODE_CHOICES,
    STORE_CHOICES,
    KnowledgeBaseConfig,
    KnowledgeBaseHit,
    KnowledgeBaseManager,
)
from .secure_config import SecureConfigStore


@dataclass(frozen=True)
class WorkerEvent:
    kind: str  # delta | done | error
    text: str = ""
    stopped: bool = False


class TkChatApp:
    def __init__(self) -> None:
        self._enable_hidpi()

        self.root = tk.Tk()
        self.root.title("YFanRAG Chat Studio")
        self.root.geometry("1280x820")
        self.root.minsize(980, 620)
        self.root.configure(bg="#0B1020")
        self._configure_tk_scaling()

        self.client = ChatApiClient()
        self.kb_manager = KnowledgeBaseManager()
        self.config_store = SecureConfigStore()
        self.queue: Queue[WorkerEvent] = Queue()
        self.messages: list[ChatMessage] = []
        self.transcript: list[dict[str, str]] = []
        self.pending = False
        self.stop_requested = False
        self.stream_message_index: int | None = None
        self.kb_window: tk.Toplevel | None = None
        self.kb_paths_text: tk.Text | None = None
        self.kb_log_text: scrolledtext.ScrolledText | None = None

        self.provider_display_to_key = self._build_provider_display_map()
        default_provider_key = "openai_compatible"
        self.provider_display_var = tk.StringVar(
            value=self._display_for_provider(default_provider_key)
        )
        self.endpoint_var = tk.StringVar(value=PROVIDER_PRESETS[default_provider_key]["endpoint"])
        self.model_var = tk.StringVar(value=PROVIDER_PRESETS[default_provider_key]["model"])
        self.api_key_var = tk.StringVar(value="")
        self.api_key_header_var = tk.StringVar(value="")
        self.temperature_var = tk.StringVar(value="0.2")
        self.max_tokens_var = tk.StringVar(value="1024")
        self.timeout_var = tk.StringVar(value="120")
        self.stream_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.provider_meta_var = tk.StringVar(value="")
        self.kb_db_var = tk.StringVar(value="yfanrag_kb.db")
        self.kb_store_var = tk.StringVar(value="sqlite-vec1")
        self.kb_enable_fts_var = tk.BooleanVar(value=True)
        self.kb_chunker_var = tk.StringVar(value="recursive")
        self.kb_chunk_size_var = tk.StringVar(value="800")
        self.kb_chunk_overlap_var = tk.StringVar(value="120")
        self.kb_dims_var = tk.StringVar(value="8")
        self.kb_query_mode_var = tk.StringVar(value="hybrid")
        self.kb_query_var = tk.StringVar(value="")
        self.kb_top_k_var = tk.StringVar(value="3")
        self.kb_doc_id_var = tk.StringVar(value="")
        self.kb_stats_var = tk.StringVar(value="KB stats: no data")
        self.kb_use_context_var = tk.BooleanVar(value=False)
        self.kb_context_top_k_var = tk.StringVar(value="3")

        self._configure_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_api_config(announce=False)
        self._refresh_provider_meta()
        self._append_log(
            "system",
            "Welcome to YFanRAG Chat Studio.\n"
            "Pick a provider preset, configure endpoint/model/key, then Send.\n"
            "Shortcut: Ctrl+Enter, Stop button supports stream interruption.",
        )
        self.root.after(180, self._focus_input)
        self.root.after(80, self._poll_queue)

    def _build_provider_display_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for key, preset in PROVIDER_PRESETS.items():
            display = f"{preset['label']} [{key}]"
            mapping[display] = key
        return mapping

    def _display_for_provider(self, provider_key: str) -> str:
        for display, key in self.provider_display_to_key.items():
            if key == provider_key:
                return display
        return provider_key

    def _provider_key(self) -> str:
        return self.provider_display_to_key.get(
            self.provider_display_var.get(),
            "openai_compatible",
        )

    @staticmethod
    def _enable_hidpi() -> None:
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes

                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _configure_tk_scaling(self) -> None:
        try:
            points_per_pixel = float(self.root.winfo_fpixels("1i")) / 72.0
            if points_per_pixel < 1.2:
                points_per_pixel = 1.2
            if points_per_pixel > 1.8:
                points_per_pixel = 1.8
            self.root.tk.call("tk", "scaling", points_per_pixel)
        except Exception:
            pass

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0B1020", foreground="#E5E7EB")
        style.configure("TFrame", background="#0B1020")
        style.configure("Topbar.TFrame", background="#111A2F")
        style.configure("Sidebar.TFrame", background="#0E1628")
        style.configure("Card.TFrame", background="#111A2F")
        style.configure("TLabel", background="#0B1020", foreground="#E5E7EB")
        style.configure("TopbarTitle.TLabel", background="#111A2F", font=("Segoe UI", 15, "bold"))
        style.configure("TopbarSub.TLabel", background="#111A2F", foreground="#9CA3AF", font=("Segoe UI", 10))
        style.configure("Meta.TLabel", background="#0B1020", foreground="#9CA3AF", font=("Segoe UI", 10))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", background="#0B1020", foreground="#9CA3AF", font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 10))
        style.configure(
            "Accent.TButton",
            background="#2563EB",
            foreground="#F8FAFC",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1D4ED8"), ("disabled", "#374151")],
            foreground=[("disabled", "#9CA3AF")],
        )
        style.configure(
            "Danger.TButton",
            background="#7F1D1D",
            foreground="#FEE2E2",
            padding=(12, 8),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#991B1B"), ("disabled", "#374151")],
            foreground=[("disabled", "#9CA3AF")],
        )
        style.configure("Tool.TCheckbutton", background="#111A2F", foreground="#D1D5DB", font=("Segoe UI", 10))
        style.configure(
            "TEntry",
            fieldbackground="#1A253D",
            foreground="#E5E7EB",
            borderwidth=1,
            insertcolor="#E5E7EB",
            padding=4,
        )
        style.configure(
            "TCombobox",
            fieldbackground="#1A253D",
            foreground="#E5E7EB",
            background="#1A253D",
            arrowcolor="#E5E7EB",
            borderwidth=1,
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", "#1A253D"),
                ("!readonly", "#1A253D"),
                ("disabled", "#374151"),
            ],
            foreground=[
                ("readonly", "#E5E7EB"),
                ("!readonly", "#E5E7EB"),
                ("disabled", "#9CA3AF"),
            ],
            background=[
                ("readonly", "#1A253D"),
                ("active", "#24324E"),
            ],
            selectbackground=[("readonly", "#2563EB"), ("!readonly", "#2563EB")],
            selectforeground=[("readonly", "#F8FAFC"), ("!readonly", "#F8FAFC")],
        )
        self.root.option_add("*TCombobox*Listbox.background", "#1A253D")
        self.root.option_add("*TCombobox*Listbox.foreground", "#E5E7EB")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#2563EB")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#F8FAFC")
        self.root.option_add("*TCombobox*Listbox.font", "Segoe UI 10")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self._build_topbar(outer)

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=BOTH, expand=True, pady=(10, 0))

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=430)
        main = ttk.Frame(body)
        body.add(sidebar, weight=0)
        body.add(main, weight=1)

        self._build_sidebar(sidebar)
        self._build_main_panel(main)

    def _build_topbar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent, style="Topbar.TFrame")
        bar.pack(fill=X)

        left = ttk.Frame(bar, style="Topbar.TFrame")
        left.pack(side=LEFT, fill=X, expand=True, padx=14, pady=10)
        ttk.Label(left, text="YFanRAG Chat Studio", style="TopbarTitle.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Desktop chat client for OpenAI-Compatible / DeepSeek / OpenAI Responses / Anthropic",
            style="TopbarSub.TLabel",
        ).pack(anchor="w")

        right = ttk.Frame(bar, style="Topbar.TFrame")
        right.pack(side=RIGHT, padx=14, pady=12)
        ttk.Button(
            right,
            text="Knowledge Base",
            command=self._open_kb_window,
        ).pack(side=LEFT, padx=(0, 10))
        ttk.Checkbutton(
            right,
            text="Stream",
            variable=self.stream_var,
            style="Tool.TCheckbutton",
        ).pack(side=LEFT, padx=(0, 10))
        self.status_badge = tk.Label(
            right,
            textvariable=self.status_var,
            bg="#1F2937",
            fg="#E5E7EB",
            padx=10,
            pady=5,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_badge.pack(side=RIGHT)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent, style="Sidebar.TFrame")
        container.pack(fill=BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(
            container,
            bg="#0E1628",
            highlightthickness=0,
            borderwidth=0,
        )
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        inner = ttk.Frame(canvas, style="Sidebar.TFrame")
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event: object | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(inner_window, width=canvas.winfo_width())

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_inner_configure)

        conn = self._new_section(inner, "Connection")
        provider_values = list(self.provider_display_to_key.keys())
        self._labeled_combobox(
            conn,
            "Provider",
            self.provider_display_var,
            provider_values,
            on_change=self._on_provider_changed,
        )
        self._labeled_entry(conn, "Endpoint", self.endpoint_var)
        self._labeled_entry(conn, "Model", self.model_var)
        self._labeled_entry(conn, "API Key", self.api_key_var, show="*")
        self._labeled_entry(conn, "API Key Header (optional)", self.api_key_header_var)

        gen = self._new_section(inner, "Generation")
        self._labeled_entry(gen, "Temperature", self.temperature_var)
        self._labeled_entry(gen, "Max Tokens", self.max_tokens_var)
        self._labeled_entry(gen, "Timeout Seconds", self.timeout_var)

        prompt = self._new_section(inner, "Prompt")
        ttk.Label(prompt, text="System Prompt", style="Section.TLabel").pack(anchor="w")
        self.system_text = tk.Text(
            prompt,
            height=5,
            bg="#1A253D",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap="word",
            relief="flat",
            font=("Segoe UI", 11),
            padx=8,
            pady=8,
            takefocus=True,
        )
        self.system_text.pack(fill=X, pady=(6, 0))

        advanced = self._new_section(inner, "Advanced JSON")
        ttk.Label(advanced, text="Extra Headers", style="Section.TLabel").pack(anchor="w")
        self.extra_headers_text = tk.Text(
            advanced,
            height=3,
            bg="#1A253D",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap="word",
            relief="flat",
            font=("Consolas", 10),
            padx=8,
            pady=8,
            takefocus=True,
        )
        self.extra_headers_text.pack(fill=X, pady=(6, 10))

        ttk.Label(advanced, text="Extra Body", style="Section.TLabel").pack(anchor="w")
        self.extra_body_text = tk.Text(
            advanced,
            height=4,
            bg="#1A253D",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap="word",
            relief="flat",
            font=("Consolas", 10),
            padx=8,
            pady=8,
            takefocus=True,
        )
        self.extra_body_text.pack(fill=X, pady=(6, 0))

        btn_row = ttk.Frame(inner, style="Sidebar.TFrame")
        btn_row.pack(fill=X, pady=(10, 6))
        ttk.Button(btn_row, text="Apply Preset", command=self._apply_provider_preset).pack(
            side=LEFT
        )
        ttk.Button(btn_row, text="Clear Chat", command=self._clear_chat).pack(side=RIGHT)

        cfg_row = ttk.Frame(inner, style="Sidebar.TFrame")
        cfg_row.pack(fill=X, pady=(0, 6))
        ttk.Button(
            cfg_row,
            text="Save API Config",
            command=lambda: self._save_api_config(verbose=True),
        ).pack(side=LEFT)
        ttk.Button(
            cfg_row,
            text="Reload API Config",
            command=lambda: self._load_api_config(announce=True),
        ).pack(side=RIGHT)

    def _build_main_panel(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill=BOTH, expand=True, padx=(10, 10), pady=10)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, minsize=172)

        header = ttk.Frame(outer, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="Conversation", style="Section.TLabel").pack(
            side=LEFT, padx=12, pady=10
        )
        ttk.Label(header, textvariable=self.provider_meta_var, style="Meta.TLabel").pack(
            side=RIGHT, padx=12, pady=10
        )

        chat_card = ttk.Frame(outer, style="Card.TFrame")
        chat_card.grid(row=1, column=0, sticky="nsew")
        chat_card.columnconfigure(0, weight=1)
        chat_card.rowconfigure(0, weight=1)
        self.chat_text = scrolledtext.ScrolledText(
            chat_card,
            wrap="word",
            bg="#0E1628",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            relief="flat",
            font=("Consolas", 12),
            padx=10,
            pady=10,
            spacing1=3,
            spacing2=2,
            spacing3=8,
            height=14,
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self.chat_text.configure(state="disabled")
        self.chat_text.tag_configure("user", foreground="#93C5FD")
        self.chat_text.tag_configure("assistant", foreground="#A7F3D0")
        self.chat_text.tag_configure("system", foreground="#C4B5FD")
        self.chat_text.tag_configure("error", foreground="#FCA5A5")
        self.chat_text.tag_configure("title", foreground="#F9FAFB", font=("Consolas", 12, "bold"))

        composer = ttk.Frame(outer, style="Card.TFrame")
        composer.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(composer, text="Message", style="Section.TLabel").pack(anchor="w", padx=12, pady=(10, 0))
        ttk.Label(
            composer,
            text="Type your prompt here (Ctrl+Enter to send)",
            style="Small.TLabel",
        ).pack(anchor="w", padx=12, pady=(2, 0))
        self.input_text = tk.Text(
            composer,
            height=5,
            bg="#1A253D",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            wrap="word",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 12),
            padx=10,
            pady=10,
            takefocus=True,
        )
        self.input_text.pack(fill=X, padx=12, pady=(6, 8))
        self.input_text.bind("<Control-Return>", self._on_send_shortcut)
        self.input_text.bind("<Button-1>", self._on_input_click)

        send_row = ttk.Frame(composer, style="Card.TFrame")
        send_row.pack(fill=X, padx=12, pady=(0, 10))
        ttk.Label(
            send_row,
            text="Ctrl+Enter to send",
            style="Small.TLabel",
        ).pack(side=LEFT)
        self.stop_button = ttk.Button(
            send_row,
            text="Stop",
            style="Danger.TButton",
            command=self._on_stop,
        )
        self.stop_button.pack(side=RIGHT, padx=(8, 0))
        self.send_button = ttk.Button(
            send_row,
            text="Send",
            style="Accent.TButton",
            command=self._on_send,
        )
        self.send_button.pack(side=RIGHT)
        self._set_pending(False)

    def _on_close(self) -> None:
        self._save_api_config(verbose=False)
        self._close_kb_window()
        self.root.destroy()

    def _collect_api_config_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "provider_key": self._provider_key(),
            "endpoint": self.endpoint_var.get(),
            "model": self.model_var.get(),
            "api_key": self.api_key_var.get(),
            "api_key_header": self.api_key_header_var.get(),
            "temperature": self.temperature_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "timeout_seconds": self.timeout_var.get(),
            "stream": bool(self.stream_var.get()),
            "system_prompt": self.system_text.get("1.0", END),
            "extra_headers_json": self.extra_headers_text.get("1.0", END),
            "extra_body_json": self.extra_body_text.get("1.0", END),
        }

    def _apply_api_config_payload(self, payload: dict[str, object]) -> None:
        provider_key = str(payload.get("provider_key", "")).strip() or "openai_compatible"
        if provider_key not in PROVIDER_PRESETS:
            provider_key = "openai_compatible"
        self.provider_display_var.set(self._display_for_provider(provider_key))
        endpoint = payload.get("endpoint")
        if isinstance(endpoint, str):
            self.endpoint_var.set(endpoint)
        model = payload.get("model")
        if isinstance(model, str):
            self.model_var.set(model)
        api_key = payload.get("api_key")
        if isinstance(api_key, str):
            self.api_key_var.set(api_key)
        api_key_header = payload.get("api_key_header")
        if isinstance(api_key_header, str):
            self.api_key_header_var.set(api_key_header)
        temperature = payload.get("temperature")
        if isinstance(temperature, str):
            self.temperature_var.set(temperature)
        max_tokens = payload.get("max_tokens")
        if isinstance(max_tokens, str):
            self.max_tokens_var.set(max_tokens)
        timeout_seconds = payload.get("timeout_seconds")
        if isinstance(timeout_seconds, str):
            self.timeout_var.set(timeout_seconds)
        self.stream_var.set(bool(payload.get("stream", self.stream_var.get())))

        system_prompt = payload.get("system_prompt")
        if isinstance(system_prompt, str):
            self.system_text.delete("1.0", END)
            self.system_text.insert("1.0", system_prompt)

        extra_headers_json = payload.get("extra_headers_json")
        if isinstance(extra_headers_json, str):
            self.extra_headers_text.delete("1.0", END)
            self.extra_headers_text.insert("1.0", extra_headers_json)

        extra_body_json = payload.get("extra_body_json")
        if isinstance(extra_body_json, str):
            self.extra_body_text.delete("1.0", END)
            self.extra_body_text.insert("1.0", extra_body_json)
        self._refresh_provider_meta()

    def _save_api_config(self, verbose: bool) -> None:
        try:
            payload = self._collect_api_config_payload()
            self.config_store.save(payload)
            if verbose:
                self._set_status("API config saved (encrypted)", tone="ok")
                self._append_log("system", "Saved encrypted API config to local disk.")
        except Exception as exc:
            if verbose:
                self._append_log("error", f"Save API config failed: {exc}")
            self._set_status("Save API config failed", tone="error")

    def _load_api_config(self, announce: bool) -> None:
        try:
            payload = self.config_store.load()
        except Exception as exc:
            if announce:
                self._append_log("error", f"Load API config failed: {exc}")
            self._set_status("Load API config failed", tone="warn")
            return

        if not payload:
            if announce:
                self._append_log("system", "No local API config found.")
            return
        try:
            self._apply_api_config_payload(payload)
        except Exception as exc:
            if announce:
                self._append_log("error", f"Apply API config failed: {exc}")
            self._set_status("Apply API config failed", tone="warn")
            return
        self._set_status("Encrypted API config loaded", tone="ok")
        if announce:
            self._append_log("system", "Loaded encrypted API config from local disk.")

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
        selected = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[
                ("Text and Markdown", "*.txt *.md"),
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
            mode = self.kb_query_mode_var.get().strip() or "hybrid"
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
        if hit.source == "hybrid":
            fused = 0.0 if hit.score is None else hit.score
            vector_score = 0.0 if hit.vector_score is None else hit.vector_score
            fts_score = 0.0 if hit.fts_score is None else hit.fts_score
            return (
                f"fused={fused:.4f} "
                f"vec={vector_score:.4f} fts={fts_score:.4f}"
            )
        if hit.source == "vector":
            if hit.distance is None:
                return "vector"
            return f"distance={hit.distance:.4f}"
        if hit.score is None:
            return "fts"
        return f"bm25={hit.score:.4f}"

    def _build_kb_context_for_user_text(self, user_text: str) -> tuple[str, str]:
        if not self.kb_use_context_var.get():
            return user_text, ""

        mode = self.kb_query_mode_var.get().strip() or "hybrid"
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

        context_block = self._format_kb_context(hits)
        payload = (
            "Knowledge Base Context:\n"
            "Use the excerpts below when relevant. If unrelated, ignore them.\n\n"
            f"{context_block}\n\n"
            f"User Question:\n{user_text}"
        )
        note = f"Attached {len(hits)} KB chunks ({mode}) from {Path(config.db_path).name}."
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

    def _new_section(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=X, pady=(0, 10))
        ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w", padx=10, pady=(10, 4))
        body = ttk.Frame(card, style="Card.TFrame")
        body.pack(fill=X, padx=10, pady=(0, 10))
        return body

    def _labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        show: str | None = None,
    ) -> None:
        ttk.Label(parent, text=label, style="Section.TLabel").pack(anchor="w")
        entry = ttk.Entry(parent, textvariable=var, show=show)
        entry.pack(fill=X, pady=(4, 9))

    def _labeled_combobox(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        values: list[str],
        on_change: object | None = None,
    ) -> None:
        ttk.Label(parent, text=label, style="Section.TLabel").pack(anchor="w")
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        box.pack(fill=X, pady=(4, 9))
        if on_change is not None:
            box.bind("<<ComboboxSelected>>", on_change)

    def _on_provider_changed(self, _event: object | None = None) -> None:
        self._apply_provider_preset()

    def _refresh_provider_meta(self) -> None:
        provider = self._provider_key()
        model = self.model_var.get().strip() or "(no model)"
        mode = "stream" if self.stream_var.get() else "non-stream"
        self.provider_meta_var.set(f"{provider} | model: {model} | {mode}")

    def _set_status(self, text: str, tone: str = "normal") -> None:
        self.status_var.set(text)
        bg = "#1F2937"
        if tone == "ok":
            bg = "#14532D"
        elif tone == "warn":
            bg = "#78350F"
        elif tone == "error":
            bg = "#7F1D1D"
        self.status_badge.configure(bg=bg)

    def _set_pending(self, value: bool) -> None:
        self.pending = value
        if value:
            self.send_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
        else:
            self.send_button.state(["!disabled"])
            self.stop_button.state(["disabled"])
        self._ensure_editable_text_fields()

    def _apply_provider_preset(self) -> None:
        provider = self._provider_key()
        preset = PROVIDER_PRESETS[provider]
        self.endpoint_var.set(preset["endpoint"])
        self.model_var.set(preset["model"])
        self.api_key_header_var.set(preset["api_key_header"])
        self._refresh_provider_meta()
        self._set_status(f"Preset applied: {preset['label']}", tone="ok")
        self._save_api_config(verbose=False)

    def _clear_chat(self) -> None:
        self.messages.clear()
        self.transcript.clear()
        self.stream_message_index = None
        self._render_chat()
        self._append_log("system", "Conversation cleared.")
        self._set_status("Chat cleared", tone="normal")
        self._focus_input()

    def _on_send_shortcut(self, _event: object | None = None) -> str:
        self._on_send()
        return "break"

    def _on_input_click(self, _event: object | None = None) -> None:
        self._focus_input()

    def _on_stop(self) -> None:
        if not self.pending:
            return
        self.stop_requested = True
        self._set_status("Stopping...", tone="warn")

    def _on_send(self) -> None:
        if self.pending:
            return
        user_text = self.input_text.get("1.0", END).strip()
        if not user_text:
            return

        outbound_text = user_text
        kb_note = ""
        try:
            outbound_text, kb_note = self._build_kb_context_for_user_text(user_text)
        except Exception as exc:
            self._append_log("error", f"Knowledge base context failed: {exc}")
            self._set_status("KB context failed", tone="warn")
            outbound_text = user_text
            kb_note = ""

        self.input_text.delete("1.0", END)
        self._append_log("user", user_text)
        if kb_note:
            self._append_log("system", kb_note)
        self.messages.append(ChatMessage(role="user", content=outbound_text))

        try:
            settings = self._collect_settings()
        except Exception as exc:
            self._append_log("error", f"Invalid settings: {exc}")
            self._set_status("Invalid settings", tone="error")
            return
        self._save_api_config(verbose=False)

        payload_messages = list(self.messages)
        self.stop_requested = False
        self._start_stream_placeholder()
        self._set_pending(True)
        self._set_status("Request in flight...", tone="warn")

        stream_enabled = bool(self.stream_var.get())
        thread = threading.Thread(
            target=self._request_worker,
            args=(settings, payload_messages, stream_enabled),
            daemon=True,
        )
        thread.start()

    def _collect_settings(self) -> ChatApiSettings:
        provider = self._provider_key()
        endpoint = self.endpoint_var.get().strip()
        model = self.model_var.get().strip()
        api_key = self.api_key_var.get().strip()
        api_key_header = self.api_key_header_var.get().strip()

        temperature_raw = self.temperature_var.get().strip()
        temperature = float(temperature_raw) if temperature_raw else None

        max_tokens_raw = self.max_tokens_var.get().strip()
        max_tokens = int(max_tokens_raw) if max_tokens_raw else None

        timeout_raw = self.timeout_var.get().strip()
        timeout_seconds = int(timeout_raw) if timeout_raw else 120

        system_prompt = self.system_text.get("1.0", END).strip()
        extra_headers = parse_json_dict(
            self.extra_headers_text.get("1.0", END),
            "Extra Headers",
        )
        extra_body = parse_json_dict(
            self.extra_body_text.get("1.0", END),
            "Extra Body",
        )

        self._refresh_provider_meta()
        return ChatApiSettings(
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            api_key_header=api_key_header,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            extra_headers={str(k): str(v) for k, v in extra_headers.items()},
            extra_body=extra_body,
        )

    def _request_worker(
        self,
        settings: ChatApiSettings,
        messages: list[ChatMessage],
        stream_enabled: bool,
    ) -> None:
        try:
            if stream_enabled:
                text, stopped = self.client.chat_stream(
                    settings=settings,
                    messages=messages,
                    on_delta=lambda chunk: self.queue.put(
                        WorkerEvent(kind="delta", text=chunk)
                    ),
                    should_stop=lambda: self.stop_requested,
                )
                self.queue.put(WorkerEvent(kind="done", text=text, stopped=stopped))
            else:
                text = self.client.chat(settings=settings, messages=messages)
                self.queue.put(WorkerEvent(kind="done", text=text, stopped=False))
        except Exception as exc:
            self.queue.put(WorkerEvent(kind="error", text=str(exc), stopped=False))

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self.queue.get_nowait()
                if event.kind == "delta":
                    self._append_stream_delta(event.text)
                    continue

                self._set_pending(False)
                if event.kind == "done":
                    self._finalize_stream_message(event.text)
                    if event.stopped:
                        self._set_status("Stopped", tone="warn")
                    else:
                        self._set_status("Response received", tone="ok")
                    self._focus_input()
                    continue

                if event.kind == "error":
                    self._finalize_stream_message("")
                    self._append_log("error", event.text)
                    self._set_status("Request failed", tone="error")
                    self._focus_input()
        except Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _start_stream_placeholder(self) -> None:
        self.transcript.append({"role": "assistant", "text": ""})
        self.stream_message_index = len(self.transcript) - 1
        self._render_chat()

    def _append_stream_delta(self, delta: str) -> None:
        if not delta:
            return
        if self.stream_message_index is None:
            self._start_stream_placeholder()
        if self.stream_message_index is None:
            return
        self.transcript[self.stream_message_index]["text"] += delta
        self._render_chat()

    def _finalize_stream_message(self, final_text: str) -> None:
        if self.stream_message_index is None:
            return
        idx = self.stream_message_index
        existing = self.transcript[idx]["text"].strip()
        fallback = final_text.strip()
        text = existing or fallback
        self.transcript[idx]["text"] = text
        self.stream_message_index = None
        self._render_chat()
        if text:
            self.messages.append(ChatMessage(role="assistant", content=text))

    def _append_log(self, role: str, text: str) -> None:
        self.transcript.append({"role": role, "text": text})
        self._render_chat()

    def _render_chat(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", END)
        for item in self.transcript:
            role = item["role"]
            text = item["text"]
            role_display = role.upper()
            self.chat_text.insert(END, f"{role_display}\n", ("title",))
            tag = role if role in {"user", "assistant", "system", "error"} else "system"
            body = text if text.strip() else "…"
            self.chat_text.insert(END, f"{body}\n\n", (tag,))
        self.chat_text.configure(state="disabled")
        self.chat_text.see(END)

    def _ensure_editable_text_fields(self) -> None:
        for widget in (
            getattr(self, "input_text", None),
            getattr(self, "system_text", None),
            getattr(self, "extra_headers_text", None),
            getattr(self, "extra_body_text", None),
            getattr(self, "kb_paths_text", None),
        ):
            if widget is None:
                continue
            try:
                if str(widget.cget("state")) != "normal":
                    widget.configure(state="normal")
            except Exception:
                pass

    def _focus_input(self) -> None:
        self._ensure_editable_text_fields()
        try:
            self.input_text.focus_set()
            self.input_text.mark_set("insert", END)
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()


def launch_tk_chat_app() -> None:
    app = TkChatApp()
    app.run()


def main() -> None:
    launch_tk_chat_app()


if __name__ == "__main__":
    main()
