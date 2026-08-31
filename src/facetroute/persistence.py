"""Small, crash-resistant JSON persistence primitives."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import PersistenceError


class AtomicJsonStore:
    """Persist one JSON object using same-directory atomic replacement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load(self, default: Any = None) -> Any:
        with self._lock:
            if not self.path.exists():
                return default
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise PersistenceError(f"Cannot read JSON state {self.path}: {exc}") from exc

    def save(self, value: Any) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_name, self.path)
                except Exception:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name)
                    raise
            except (OSError, TypeError, ValueError) as exc:
                raise PersistenceError(f"Cannot write JSON state {self.path}: {exc}") from exc
