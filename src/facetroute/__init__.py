"""FacetRoute: offline-first, explainable personalized LLM routing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bandit import LinUCBPolicy, LinUCBRouter
from .benchmark import (
    BenchmarkManifest,
    BenchmarkMetrics,
    BenchmarkReport,
    BenchmarkRunner,
    IntervalEstimate,
    PolicySpec,
)
from .calibration import CalibrationPoint, CalibrationReport, ThresholdCalibrator
from .constraints import ConstraintEngine, ConstraintResult
from .errors import ConfigurationError, FacetRouteError, NoEligibleModelError, PersistenceError
from .features import CONTEXT_DIMENSION, QueryFeatureExtractor
from .feedback import FeedbackEvent, FeedbackLog, ModelFeedbackSummary
from .pareto import dominates, pareto_front
from .profiles import PreferenceStore
from .reporting import (
    benchmark_rows,
    write_benchmark_csv,
    write_benchmark_html,
    write_calibration_csv,
    write_json,
)
from .routers import BatchRouter, BatchRouteResult, ParetoRouter, RuleRouter
from .rules import RoutingRule
from .scoring import MultiObjectiveScorer
from .server import FacetRouteHTTPServer, create_server, route_request_from_http
from .simulator import EvaluationReport, OfflineSimulator, SimulationObservation
from .traces import RouteTrace, TraceOutcome, file_sha256, iter_traces, load_traces
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
    __version__ = "0.2.0"

__all__ = [
    "CONTEXT_DIMENSION",
    "BatchRouteResult",
    "BatchRouter",
    "BenchmarkManifest",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "BenchmarkRunner",
    "CalibrationPoint",
    "CalibrationReport",
    "ConfigurationError",
    "ConstraintEngine",
    "ConstraintResult",
    "EvaluationReport",
    "FacetRouteError",
    "FacetRouteHTTPServer",
    "FeedbackEvent",
    "FeedbackLog",
    "IntervalEstimate",
    "LinUCBPolicy",
    "LinUCBRouter",
    "ModelCandidate",
    "ModelFeedbackSummary",
    "MultiObjectiveScorer",
    "NoEligibleModelError",
    "OfflineSimulator",
    "ParetoRouter",
    "PersistenceError",
    "PolicySpec",
    "PreferenceStore",
    "QueryFeatureExtractor",
    "QueryFeatures",
    "RouteDecision",
    "RouteRequest",
    "RouteTrace",
    "RoutingRule",
    "RuleRouter",
    "ScoreBreakdown",
    "SimulationObservation",
    "ThresholdCalibrator",
    "TraceOutcome",
    "UserPreferences",
    "benchmark_rows",
    "create_server",
    "dominates",
    "file_sha256",
    "iter_traces",
    "load_traces",
    "pareto_front",
    "route_request_from_http",
    "write_benchmark_csv",
    "write_benchmark_html",
    "write_calibration_csv",
    "write_json",
]
