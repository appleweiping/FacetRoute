"""Deterministic query feature extraction with no model downloads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .types import QueryFeatures, RouteRequest, UserPreferences

_CODE_MARKERS = re.compile(
    r"(```|\b(def|class|function|SELECT|INSERT|async|await|import|const|let|var)\b|[{};])",
    re.IGNORECASE,
)
_MATH_MARKERS = re.compile(r"([=+*/^]|\b(integral|derivative|equation|matrix|probability)\b)", re.IGNORECASE)
_MULTISTEP = re.compile(
    r"\b(step[- ]by[- ]step|first.+then|compare.+and|analy[sz]e|prove|derive|plan)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class QueryFeatureExtractor:
    """Extract inexpensive lexical features suitable for offline routing."""

    long_query_chars: int = 1200

    def extract(self, request: RouteRequest) -> QueryFeatures:
        text = request.query.strip()
        character_count = len(text)
        token_estimate = (
            request.context_tokens
            if request.context_tokens is not None
            else max(1, math.ceil(character_count / 4))
        )
        code_hits = len(_CODE_MARKERS.findall(text))
        math_hits = len(_MATH_MARKERS.findall(text))
        scale = max(1, len(text.split()))
        code_fraction = min(1.0, code_hits / scale)
        math_fraction = min(1.0, math_hits / scale)
        question_count = text.count("?") + text.count("\N{FULLWIDTH QUESTION MARK}")
        has_multistep = bool(_MULTISTEP.search(text))
        task = request.task_hint or self._infer_task(text, code_fraction, math_fraction)

        length_signal = min(1.0, character_count / max(1, self.long_query_chars))
        difficulty = 0.2 + 0.35 * length_signal
        difficulty += 0.2 if has_multistep else 0.0
        difficulty += min(0.15, 0.5 * code_fraction)
        difficulty += min(0.10, 0.4 * math_fraction)
        difficulty = round(min(1.0, difficulty), 6)

        capabilities = set(request.required_capabilities)
        capabilities.add("text")
        if task == "code":
            capabilities.add("code")
        elif task == "math":
            capabilities.add("math")
        elif task == "reasoning":
            capabilities.add("reasoning")
        if request.needs_tools:
            capabilities.add("tools")
        if request.needs_json:
            capabilities.add("json")

        return QueryFeatures(
            task=task,
            token_estimate=token_estimate,
            character_count=character_count,
            difficulty=difficulty,
            code_fraction=round(code_fraction, 6),
            math_fraction=round(math_fraction, 6),
            question_count=question_count,
            has_multistep_language=has_multistep,
            required_capabilities=frozenset(capabilities),
        )

    @staticmethod
    def _infer_task(text: str, code_fraction: float, math_fraction: float) -> str:
        lowered = text.lower()
        if code_fraction >= 0.025 or any(word in lowered for word in ("debug", "compile", "sql", "api")):
            return "code"
        if math_fraction >= 0.04 or any(word in lowered for word in ("calculate", "theorem", "algebra")):
            return "math"
        if any(word in lowered for word in ("translate", "translation", "翻译")):
            return "translation"
        if any(word in lowered for word in ("summarize", "summary", "摘要", "总结")):
            return "summarization"
        if _MULTISTEP.search(text):
            return "reasoning"
        return "general"

    @staticmethod
    def context_vector(features: QueryFeatures, preferences: UserPreferences) -> tuple[float, ...]:
        """Return a stable, bounded vector used by the contextual bandit."""

        task_order = ("general", "code", "math", "reasoning", "translation", "summarization")
        task_flags = tuple(1.0 if features.task == task else 0.0 for task in task_order)
        quality, cost, latency = preferences.objective_weights(features.task)
        return (
            1.0,
            features.difficulty,
            min(1.0, features.token_estimate / 32_000.0),
            features.code_fraction,
            features.math_fraction,
            min(1.0, features.question_count / 5.0),
            1.0 if features.has_multistep_language else 0.0,
            *task_flags,
            quality,
            cost,
            latency,
        )


CONTEXT_DIMENSION = 16
