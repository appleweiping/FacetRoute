"""FacetRoute: offline-first, explainable personalized LLM routing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bandit import LinUCBPolicy, LinUCBRouter
from .constraints import ConstraintEngine, ConstraintResult
from .errors import ConfigurationError, FacetRouteError, NoEligibleModelError, PersistenceError
from .features import CONTEXT_DIMENSION, QueryFeatureExtractor
from .feedback import FeedbackEvent, FeedbackLog, ModelFeedbackSummary
from .pareto import dominates, pareto_front
from .profiles import PreferenceStore
from .routers import BatchRouter, BatchRouteResult, ParetoRouter, RuleRouter
from .rules import RoutingRule
from .scoring import MultiObjectiveScorer
from .simulator import EvaluationReport, OfflineSimulator, SimulationObservation
from .types import (
    ModelCandidate,
    QueryFeatures,
    RouteDecision,
    RouteRequest,
    ScoreBreakdown,
    UserPreferences,
)

try:
    __version__ = version("facetroute")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.1.0"

__all__ = [
    "CONTEXT_DIMENSION",
    "BatchRouteResult",
    "BatchRouter",
    "ConfigurationError",
    "ConstraintEngine",
    "ConstraintResult",
    "EvaluationReport",
    "FacetRouteError",
    "FeedbackEvent",
    "FeedbackLog",
    "LinUCBPolicy",
    "LinUCBRouter",
    "ModelCandidate",
    "ModelFeedbackSummary",
    "MultiObjectiveScorer",
    "NoEligibleModelError",
    "OfflineSimulator",
    "ParetoRouter",
    "PersistenceError",
    "PreferenceStore",
    "QueryFeatureExtractor",
    "QueryFeatures",
    "RouteDecision",
    "RouteRequest",
    "RoutingRule",
    "RuleRouter",
    "ScoreBreakdown",
    "SimulationObservation",
    "UserPreferences",
    "dominates",
    "pareto_front",
]
