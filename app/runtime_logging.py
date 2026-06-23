from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_level=True,
        show_path=False,
        omit_repeated_times=False,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
