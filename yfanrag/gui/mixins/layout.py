"""Layout and styling methods for the Tk chat app."""

from __future__ import annotations

import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, ttk
from tkinter import scrolledtext

from ...chat_providers import PROVIDER_PRESETS
from ...knowledge_base import QUERY_MODE_CHOICES, STORE_CHOICES
from ...loaders.text import DEFAULT_TEXT_EXTENSIONS


class AppLayoutMixin:
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
        self.chat_text.tag_configure("md_h1", font=("Segoe UI", 16, "bold"), spacing1=8, spacing3=3)
        self.chat_text.tag_configure("md_h2", font=("Segoe UI", 14, "bold"), spacing1=7, spacing3=3)
        self.chat_text.tag_configure("md_h3", font=("Segoe UI", 13, "bold"), spacing1=6, spacing3=2)
        self.chat_text.tag_configure("md_h4", font=("Segoe UI", 12, "bold"), spacing1=5, spacing3=2)
        self.chat_text.tag_configure("md_bold", font=("Consolas", 12, "bold"))
        self.chat_text.tag_configure("md_italic", font=("Consolas", 12, "italic"))
        self.chat_text.tag_configure("md_bold_italic", font=("Consolas", 12, "bold italic"))
        self.chat_text.tag_configure("md_strike", overstrike=1)
        self.chat_text.tag_configure(
            "md_code_inline",
            font=("Consolas", 11),
            foreground="#FDE68A",
            background="#1F2937",
        )
        self.chat_text.tag_configure(
            "md_code_block",
            font=("Consolas", 11),
            foreground="#E5E7EB",
            background="#111827",
            lmargin1=14,
            lmargin2=14,
            spacing1=2,
            spacing3=2,
        )
        self.chat_text.tag_configure(
            "md_code_meta",
            font=("Consolas", 10, "italic"),
            foreground="#9CA3AF",
            lmargin1=14,
            lmargin2=14,
        )
        self.chat_text.tag_configure(
            "md_quote",
            foreground="#CBD5E1",
            lmargin1=16,
            lmargin2=16,
        )
        self.chat_text.tag_configure("md_quote_marker", foreground="#64748B")
        self.chat_text.tag_configure("md_list", lmargin1=12, lmargin2=24)
        self.chat_text.tag_configure("md_list_marker", foreground="#D1D5DB")
        self.chat_text.tag_configure("md_link", foreground="#60A5FA", underline=1)
        self.chat_text.tag_configure("md_link_url", foreground="#93C5FD", font=("Consolas", 10))
        self.chat_text.tag_configure("md_hr", foreground="#334155")

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
