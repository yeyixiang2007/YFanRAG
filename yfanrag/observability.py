"""Unified logging and slow-query hint helpers."""

from __future__ import annotations

import logging
import os

LOGGER_NAME = "yfanrag"
DEFAULT_SLOW_QUERY_MS = 200.0


def configure_logging(level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)

    level_name = level or os.getenv("YFANRAG_LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def slow_query_threshold_ms() -> float:
    raw = os.getenv("YFANRAG_SLOW_QUERY_MS", "")
    if not raw:
        return DEFAULT_SLOW_QUERY_MS
    try:
        value = float(raw)
        if value <= 0:
            return DEFAULT_SLOW_QUERY_MS
        return value
    except ValueError:
        return DEFAULT_SLOW_QUERY_MS


def log_slow_query(operation: str, elapsed_ms: float, extra: str = "") -> None:
    logger = get_logger()
    threshold = slow_query_threshold_ms()
    if elapsed_ms >= threshold:
        msg = (
            f"slow query detected: op={operation} elapsed_ms={elapsed_ms:.2f} "
            f"threshold_ms={threshold:.2f}"
        )
        if extra:
            msg += f" {extra}"
        logger.warning(msg)
    else:
        logger.debug("query op=%s elapsed_ms=%.2f", operation, elapsed_ms)
