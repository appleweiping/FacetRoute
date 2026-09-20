"""FacetRoute: offline-first, explainable personalized LLM routing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .async_client import (
    AsyncChatCompletionProvider,
    AsyncProviderRegistry,
    AsyncProviderTarget,
    AsyncRoutingController,
    RoutedCompletion,
    RoutedStream,
)
from .bandit import LinUCBPolicy, LinUCBRouter
from .benchmark import (
    BenchmarkManifest,
    BenchmarkMetrics,
    BenchmarkReport,
    BenchmarkRunner,
    IntervalEstimate,
    PolicySpec,
)
from .benchmark_formats import (
    BenchmarkExample,
    BenchmarkFormat,
    load_benchmark_examples,
    write_benchmark_examples,
)
from .benchmark_sweep import (
    CachedSweep,
    SweepConfig,
    SweepPoint,
    SweepReport,
    cached_threshold_sweep,
    load_sweep_traces,
    run_threshold_sweep,
)
from .calibration import CalibrationPoint, CalibrationReport, ThresholdCalibrator
from .constraints import ConstraintEngine, ConstraintResult
from .errors import ConfigurationError, FacetRouteError, NoEligibleModelError, PersistenceError
from .factorization import (
    FACTOR_FORMAT,
    FACTOR_SCHEMA_VERSION,
    FactorizationConfig,
    FactorizationRouter,
    PairwiseFactorModel,
    evaluate_held_out,
)
from .features import CONTEXT_DIMENSION, QueryFeatureExtractor
from .feedback import FeedbackEvent, FeedbackLog, ModelFeedbackSummary
from .pareto import dominates, pareto_front
from .profiles import PreferenceStore
from .providers import (
    ChatCompletionProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderFailure,
    ProviderRegistry,
    ProviderTarget,
    load_provider_registry,
)
from .reporting import (
    benchmark_rows,
    write_benchmark_csv,
    write_benchmark_html,
    write_calibration_csv,
    write_json,
)
from .resilience import ProviderResiliencePolicy, ResilientProviderRegistry
from .routers import BatchRouter, BatchRouteResult, ParetoRouter, RuleRouter
from .rules import RoutingRule
from .scoring import MultiObjectiveScorer
from .server import (
    FacetRouteHTTPServer,
    chat_completion_from_http,
    create_server,
    route_request_from_http,
)
from .similarity import (
    SIMILARITY_FEATURE_SCHEMA_VERSION,
    SIMILARITY_FORMAT,
    SIMILARITY_SCHEMA_VERSION,
    SimilarityCalibrationPoint,
    SimilarityEvaluation,
    SimilarityExperimentReport,
    SimilarityFeatureConfig,
    SimilarityMatch,
    SimilarityModel,
    SimilarityRouter,
    fit_calibrate_evaluate,
)
from .simulator import EvaluationReport, OfflineSimulator, SimulationObservation
from .splitting import TracePartitions, split_traces, write_trace_partitions
from .traces import (
    RouteTrace,
    TraceOutcome,
    file_sha256,
    iter_traces,
    load_traces,
    traces_sha256,
    write_traces,
)
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
    __version__ = "0.10.0"

__all__ = [
    "CONTEXT_DIMENSION",
    "FACTOR_FORMAT",
    "FACTOR_SCHEMA_VERSION",
    "SIMILARITY_FEATURE_SCHEMA_VERSION",
    "SIMILARITY_FORMAT",
    "SIMILARITY_SCHEMA_VERSION",
    "AsyncChatCompletionProvider",
    "AsyncProviderRegistry",
    "AsyncProviderTarget",
    "AsyncRoutingController",
    "BatchRouteResult",
    "BatchRouter",
    "BenchmarkExample",
    "BenchmarkFormat",
    "BenchmarkManifest",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "BenchmarkRunner",
    "CachedSweep",
    "CalibrationPoint",
    "CalibrationReport",
    "ChatCompletionProvider",
    "ConfigurationError",
    "ConstraintEngine",
    "ConstraintResult",
    "EvaluationReport",
    "FacetRouteError",
    "FacetRouteHTTPServer",
    "FactorizationConfig",
    "FactorizationRouter",
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
    "OpenAICompatibleProvider",
    "PairwiseFactorModel",
    "ParetoRouter",
    "PersistenceError",
    "PolicySpec",
    "PreferenceStore",
    "ProviderError",
    "ProviderFailure",
    "ProviderRegistry",
    "ProviderResiliencePolicy",
    "ProviderTarget",
    "QueryFeatureExtractor",
    "QueryFeatures",
    "ResilientProviderRegistry",
    "RouteDecision",
    "RouteRequest",
    "RouteTrace",
    "RoutedCompletion",
    "RoutedStream",
    "RoutingRule",
    "RuleRouter",
    "ScoreBreakdown",
    "SimilarityCalibrationPoint",
    "SimilarityEvaluation",
    "SimilarityExperimentReport",
    "SimilarityFeatureConfig",
    "SimilarityMatch",
    "SimilarityModel",
    "SimilarityRouter",
    "SimulationObservation",
    "SweepConfig",
    "SweepPoint",
    "SweepReport",
    "ThresholdCalibrator",
    "TraceOutcome",
    "TracePartitions",
    "UserPreferences",
    "benchmark_rows",
    "cached_threshold_sweep",
    "chat_completion_from_http",
    "create_server",
    "dominates",
    "evaluate_held_out",
    "file_sha256",
    "fit_calibrate_evaluate",
    "iter_traces",
    "load_benchmark_examples",
    "load_provider_registry",
    "load_sweep_traces",
    "load_traces",
    "pareto_front",
    "route_request_from_http",
    "run_threshold_sweep",
    "split_traces",
    "traces_sha256",
    "write_benchmark_csv",
    "write_benchmark_examples",
    "write_benchmark_html",
    "write_calibration_csv",
    "write_json",
    "write_trace_partitions",
    "write_traces",
]
