"""Optional-dependency helpers."""

from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(name: str, extra: str) -> ModuleType:
    """Import ``name`` or raise an ImportError that names the extra to install."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"'{name}' is required for this feature. Install it with "
            f"`pip install 'bna-toolkit[{extra}]'`."
        ) from exc


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
