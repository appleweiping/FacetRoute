"""Strict JSON helpers shared by public input surfaces.

The standard :mod:`json` decoder accepts duplicate object keys and non-finite
numbers.  Both are dangerous for experiment manifests and HTTP requests: the
same bytes can be interpreted differently by another implementation.  These
helpers intentionally accept the portable JSON subset instead.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any

_MAX_JSON_NESTING = 256


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains the same key more than once."""


def _object_from_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_float(value: str) -> float:
    """Decode a JSON float without permitting exponent overflow to infinity."""

    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:  # Defensive across Python float parsers.
        raise ValueError(f"non-finite JSON number is not allowed: {value}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return number


def _validate_nesting(value: str) -> None:
    """Apply one platform-independent nesting limit before the recursive decoder."""

    depth = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise ValueError("JSON nesting exceeds the supported depth")
        elif character in "]}":
            depth -= 1


def _validate_portable_values(value: Any) -> None:
    """Reject non-Unicode strings and non-finite values anywhere in the result."""

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8", "strict")
            except UnicodeEncodeError as exc:
                raise ValueError("JSON strings must contain valid Unicode scalars") from exc
        elif isinstance(current, float) and not math.isfinite(current):
            raise ValueError("non-finite JSON number is not allowed")
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())


def loads_strict(value: str | bytes | bytearray) -> Any:
    """Decode portable UTF-8 JSON with bounded depth and unambiguous values."""

    try:
        if isinstance(value, str):
            try:
                # Snapshot a str subclass through the built-in implementation and
                # reject invalid scalar values before invoking any parser logic.
                text = str.encode(value, "utf-8", "strict").decode("utf-8", "strict")
            except UnicodeError as exc:
                raise ValueError("JSON strings must contain valid Unicode scalars") from exc
        elif isinstance(value, (bytes, bytearray)):
            text = bytes(value).decode("utf-8", "strict")
        else:
            raise TypeError("JSON input must be str, bytes, or bytearray")
        _validate_nesting(text)
        decoded = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
        _validate_portable_values(decoded)
        return decoded
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported depth") from exc
    except UnicodeError as exc:
        raise ValueError("JSON input is not valid UTF-8") from exc
