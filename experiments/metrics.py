"""Thin wrapper re-exporting metrics utilities from pipelines.metrics (single source of truth)."""

from pipelines.metrics import *  # noqa: F401,F403
from pipelines import metrics as _metrics  # type: ignore

__all__ = getattr(_metrics, "__all__", [])
