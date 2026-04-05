"""Core behaviors for the Tk chat app."""

from __future__ import annotations

import tkinter as tk
from tkinter import END, X, ttk

from ...chat_providers import PROVIDER_PRESETS


class AppCoreMixin:
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

    def _on_close(self) -> None:
        self._save_api_config(verbose=False)
        self._close_kb_window()
        self.root.destroy()

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
