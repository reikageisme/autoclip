"""Small helpers to construct shared contract objects defensively.

The media modules only *consume* ``autoclip.contracts``.  Because the contract
module is owned by the lead and may be a dataclass or a pydantic model, object
construction and counter updates go through these helpers instead of assuming a
specific implementation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

T = TypeVar("T")

# Canonical counter name -> names a contract implementation may use instead.
REPORT_COUNTERS: dict[str, tuple[str, ...]] = {
    "clips_seen": ("clips_seen", "clips_found", "found", "total_clips", "clips_total", "total"),
    "clips_added": ("clips_added", "added", "new_clips"),
    "clips_skipped": ("clips_skipped", "skipped", "duplicates"),
    "segments_added": ("segments_added", "segments", "total_segments"),
    "segments_deduped": ("segments_deduped", "deduped", "duplicates_removed"),
}


def field_names(model: type[Any]) -> set[str]:
    """Return declared field names for a dataclass, pydantic v2 or v1 model."""
    if dataclasses.is_dataclass(model):
        return {field.name for field in dataclasses.fields(model)}
    model_fields = getattr(model, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return set(model_fields)
    legacy_fields = getattr(model, "__fields__", None)
    if isinstance(legacy_fields, Mapping):
        return set(legacy_fields)
    return set()


def construct(model: type[T], **values: Any) -> T:
    """Build a contract object, ignoring keywords the model does not declare."""
    fields = field_names(model)
    if not fields:
        return model(**values)  # type: ignore[call-arg]
    filtered = {key: value for key, value in values.items() if key in fields}
    return model(**filtered)  # type: ignore[call-arg]


def bump(report: Any, canonical: str, amount: int = 1) -> None:
    """Increment one report counter, tolerating alternative field names."""
    names = REPORT_COUNTERS.get(canonical, (canonical,))
    fields = field_names(type(report))
    for name in names:
        if name not in fields and not hasattr(report, name):
            continue
        try:
            setattr(report, name, int(getattr(report, name, 0) or 0) + amount)
        except (AttributeError, TypeError, ValueError):
            continue


def add_error(report: Any, path: str, message: object) -> None:
    """Record a failure on the report (``IngestError`` when the contract has it)."""
    errors = getattr(report, "errors", None)
    if not isinstance(errors, list):
        return
    item: Any
    try:
        from autoclip.contracts import IngestError

        item = IngestError(path=str(path), message=str(message))
    except (ImportError, AttributeError, TypeError, ValueError):
        item = f"{path}: {message}"
    try:
        errors.append(item)
    except (AttributeError, TypeError, ValueError):
        return
