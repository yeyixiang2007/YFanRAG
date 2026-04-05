"""Modern Tkinter chat UI with streaming and multi-provider API compatibility."""

from __future__ import annotations

from queue import Queue
import tkinter as tk
from tkinter import scrolledtext, ttk

from ..chat_providers import ChatApiClient, ChatMessage, PROVIDER_PRESETS
from ..feedback_loop import FeedbackLoopStore
from ..knowledge_base import KnowledgeBaseManager
from ..secure_config import SecureConfigStore
from .events import WorkerEvent
from .mixins import (
    AppChatMixin,
    AppConfigMixin,
    AppCoreMixin,
    AppKnowledgeBaseMixin,
    AppLayoutMixin,
)


class TkChatApp(
    AppLayoutMixin,
    AppConfigMixin,
    AppKnowledgeBaseMixin,
    AppChatMixin,
    AppCoreMixin,
):
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
        self.feedback_store = FeedbackLoopStore()
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
        self.kb_query_mode_var = tk.StringVar(value="auto")
        self.kb_query_var = tk.StringVar(value="")
        self.kb_top_k_var = tk.StringVar(value="3")
        self.kb_doc_id_var = tk.StringVar(value="")
        self.kb_stats_var = tk.StringVar(value="KB stats: no data")
        self.kb_use_context_var = tk.BooleanVar(value=False)
        self.kb_context_top_k_var = tk.StringVar(value="3")
        self.kb_traceability_required = False
        self.kb_traceability_refs: list[str] = []
        self.kb_feedback_refs: list[dict[str, object]] = []
        self.kb_feedback_plan_summary = ""
        self.kb_feedback_requested_mode = ""
        self.kb_feedback_resolved_mode = ""
        self.kb_feedback_query_type = ""
        self.kb_feedback_db_path = ""
        self.pending_feedback_context: dict[str, object] | None = None
        self.feedback_target: dict[str, object] | None = None
        self.feedback_helpful_button: ttk.Button | None = None
        self.feedback_unhelpful_button: ttk.Button | None = None

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


def launch_tk_chat_app() -> None:
    app = TkChatApp()
    app.run()


def main() -> None:
    launch_tk_chat_app()


if __name__ == "__main__":
    main()
