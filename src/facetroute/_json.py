"""Strict JSON helpers shared by public input surfaces.

The standard :mod:`json` decoder accepts duplicate object keys and non-finite
numbers.  Both are dangerous for experiment manifests and HTTP requests: the
same bytes can be interpreted differently by another implementation.  These
helpers intentionally accept the portable JSON subset instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


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


def loads_strict(value: str | bytes | bytearray) -> Any:
    """Decode standards-compliant JSON, rejecting ambiguity and NaN values."""

    return json.loads(
        value,
        object_pairs_hook=_object_from_pairs,
        parse_constant=_reject_constant,
    )
