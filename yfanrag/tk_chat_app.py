"""Backward-compatible entrypoint for the Tk chat app.

Implementation now lives under :mod:`yfanrag.gui`.
"""

from __future__ import annotations

from .gui.app import TkChatApp, launch_tk_chat_app, main

__all__ = ["TkChatApp", "launch_tk_chat_app", "main"]


if __name__ == "__main__":
    main()
