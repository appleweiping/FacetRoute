"""FacetRoute exception hierarchy."""

from __future__ import annotations


class FacetRouteError(Exception):
    """Base class for domain errors raised by FacetRoute."""


class ConfigurationError(FacetRouteError, ValueError):
    """Raised when a model, profile, rule, or request is invalid."""


class NoEligibleModelError(FacetRouteError):
    """Raised when every candidate violates at least one hard constraint."""

    def __init__(self, reasons: dict[str, tuple[str, ...]]) -> None:
        self.reasons = reasons
        detail = "; ".join(
            f"{model_id}: {', '.join(items)}" for model_id, items in sorted(reasons.items())
        )
        super().__init__(f"No model satisfies the request constraints. {detail}")


class PersistenceError(FacetRouteError, OSError):
    """Raised when durable local state cannot be read or written safely."""
