"""Small, crash-resistant JSON persistence primitives."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import PersistenceError


def _json_bytes(value: Any, *, max_bytes: int | None = None) -> bytes:
    """Encode one portable JSON document and enforce an optional byte limit."""

    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise ValueError("max_bytes must be a positive integer")
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ValueError(f"encoded JSON state exceeds {max_bytes} bytes ({len(encoded)} bytes)")
    return encoded


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _destinations_collide(left: Path, right: Path) -> bool:
    try:
        resolved_left = left.resolve()
        resolved_right = right.resolve()
        if (
            resolved_left == resolved_right
            or resolved_left in resolved_right.parents
            or resolved_right in resolved_left.parents
        ):
            return True
        return _lexists(left) and _lexists(right) and left.samefile(right)
    except (OSError, RuntimeError):
        return False


def _stage_bytes(destination: Path, content: bytes, *, suffix: str = ".tmp") -> Path:
    """Create, flush, and sync one same-directory staging file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    temporary = Path(temporary_name)
    stream = None
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1  # Ownership transferred to ``stream``.
        with stream:
            written = stream.write(content)
            if written != len(content):
                raise OSError(f"short write: wrote {written} of {len(content)} bytes")
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except BaseException:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.unlink(temporary)
        raise


def _atomic_write_bytes_bundle(
    writes: Mapping[Path, bytes],
    *,
    validators: Mapping[Path, Callable[[Path], None]] | None = None,
) -> None:
    """Commit several files as one recoverable operation.

    All payloads are staged and synced before an existing destination is moved.
    If any pre-commit replacement fails, prior destinations are restored and
    newly-created destinations are removed.  Atomicity across directories is not
    available from the filesystem, so this helper provides rollback rather than
    pretending that the group has one kernel-level linearization point.
    """

    entries = tuple(writes.items())
    if not entries:
        return
    if any(not isinstance(content, bytes) for _, content in entries):
        raise TypeError("atomic byte payloads must be bytes")
    for index, (left, _) in enumerate(entries):
        for right, _ in entries[index + 1 :]:
            if _destinations_collide(left, right):
                raise ValueError(f"atomic output destinations collide: {left} and {right}")
    for destination, _ in entries:
        if _lexists(destination) and destination.is_dir():
            raise IsADirectoryError(f"output destination is a directory: {destination}")
    validation_by_destination = (
        {Path(path): validator for path, validator in validators.items()} if validators else {}
    )
    destinations = {destination for destination, _ in entries}
    if any(destination not in destinations for destination in validation_by_destination):
        raise ValueError("atomic output validator has no corresponding destination")

    staged: dict[Path, Path] = {}
    backup_placeholders: dict[Path, Path] = {}
    backups_moved: set[Path] = set()
    originally_present = tuple(destination for destination, _ in entries if _lexists(destination))
    originally_present_set = set(originally_present)
    try:
        for destination, content in entries:
            staged[destination] = _stage_bytes(destination, content)
        for destination, validator in validation_by_destination.items():
            validator(staged[destination])
        for destination in originally_present:
            backup = _stage_bytes(destination, b"", suffix=".bak")
            backup_placeholders[destination] = backup
            try:
                os.replace(destination, backup)
            except BaseException:
                # A fault injector may raise after performing the replacement.
                if not _lexists(destination) and _lexists(backup):
                    backups_moved.add(destination)
                raise
            backups_moved.add(destination)
        for destination, _ in entries:
            temporary = staged[destination]
            os.replace(temporary, destination)
            staged.pop(destination)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for destination, _ in reversed(entries):
            backup_path = backup_placeholders.get(destination)
            if destination in backups_moved and backup_path is not None and _lexists(backup_path):
                try:
                    os.replace(backup_path, destination)
                    backups_moved.remove(destination)
                except BaseException as rollback_exc:
                    rollback_errors.append(f"restore {destination}: {rollback_exc}")
            elif destination not in originally_present_set and _lexists(destination):
                try:
                    os.unlink(destination)
                except BaseException as rollback_exc:
                    rollback_errors.append(f"remove {destination}: {rollback_exc}")
        for temporary in (*staged.values(), *backup_placeholders.values()):
            if _lexists(temporary):
                try:
                    os.unlink(temporary)
                except BaseException as cleanup_exc:
                    rollback_errors.append(f"remove staging file {temporary}: {cleanup_exc}")
        if rollback_errors and hasattr(exc, "add_note"):
            exc.add_note("atomic output rollback errors: " + "; ".join(rollback_errors))
        raise
    else:
        for backup in backup_placeholders.values():
            with suppress(OSError):
                os.unlink(backup)


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
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise PersistenceError(f"Cannot read JSON state {self.path}: {exc}") from exc

    def save(self, value: Any, *, max_bytes: int | None = None) -> None:
        with self._lock:
            try:
                encoded = _json_bytes(value, max_bytes=max_bytes)
                _atomic_write_bytes_bundle({self.path: encoded})
            except (OSError, TypeError, ValueError, RecursionError) as exc:
                raise PersistenceError(f"Cannot write JSON state {self.path}: {exc}") from exc
