# Opt-in provider resilience and asynchronous client

The existing `/v1/chat/completions` proxy remains synchronous and does not
retry by default. `serve --provider-resilience` enables a per-catalog-model
circuit breaker. `--provider-retry-attempts 2` (maximum 5) additionally enables
exponential backoff, but **only** for a transport error certified to have
occurred before any request bytes were sent. There is no failover to another
model: the normal router applies all hard constraints once, then the selected
model either succeeds or returns a redacted provider error.

```bash
facetroute serve --models examples/models.json \
  --providers examples/providers.example.json \
  --provider-resilience --provider-retry-attempts 2 \
  --provider-retry-backoff 0.1 --provider-retry-max-backoff 2 \
  --provider-circuit-failures 3 --provider-circuit-recovery 30
```

An initial DNS/TCP/TLS connection failure is the only built-in provider error
marked retry-safe. A timeout during request send, while waiting for response
headers, while reading a response, or after an SSE chunk may already have
generated output and is **never retried**. HTTP 429/5xx responses are counted
as provider failures for circuit health but are not automatically retried. A
retry is skipped if backoff would exhaust the original call deadline. The
breaker has one bounded state cell per configured model (maximum 1,024),
counts consecutive operational failures, rejects while open, and admits only
one half-open probe after recovery. A failed or abandoned probe reopens it.
No prompt, body, URL credentials, or API key appears in breaker errors.

Python applications can inject a genuinely asynchronous provider transport:

```python
from facetroute import (
    AsyncProviderRegistry, AsyncProviderTarget, AsyncRoutingController,
    ProviderResiliencePolicy, RouteRequest, RuleRouter,
)

# `client` implements async complete(...) and an async-iterator stream(...).
registry = AsyncProviderRegistry(
    (AsyncProviderTarget("catalog-id", "upstream-id", client),),
    policy=ProviderResiliencePolicy(max_attempts=2),
)
controller = AsyncRoutingController(router, registry)
result = await controller.complete(
    RouteRequest(query="hello"), {"messages": [{"role": "user", "content": "hello"}]},
    timeout_seconds=20,
)
print(result.decision.selected_model, result.response)
```

The async controller awaits the injected transport and never runs the blocking
stdlib HTTP provider on the event loop. The injected transport is responsible
for honestly marking only proven pre-send failures as retry-safe. The async
registry passes a copy of the request with the selected upstream `model` and
the correct `stream` boolean, overriding conflicting caller fields without
changing the caller's mapping. It validates completions/chunks and preserves the same no-partial-stream
retry rule. Fake-clock, concurrency, deadline, and local-server tests cover
the shipped boundary without outbound credentials.

### Optional native asynchronous HTTP transport

Install `facetroute[async]`, then explicitly import
`AsyncOpenAICompatibleProvider` from `facetroute.async_http` and pass it as the
`client` above. The default package import and CLI do not require HTTPX; the
synchronous provider and `serve` behavior are unchanged.

The native client uses a fixed, validated OpenAI-compatible endpoint. Plain
HTTP is accepted only for loopback development; HTTPS verifies certificates.
Environment proxies and redirects are disabled. A separate client is owned
and closed for each call, including cancellation or partial stream closure;
call `await stream.aclose()` if you stop consuming an SSE stream early. Request
JSON is capped at 16 MiB by default, response JSON at 16 MiB, raw SSE at
64 MiB, and each SSE event at 1 MiB. Compressed or otherwise encoded responses
are rejected. The specified timeout is a total wall-clock deadline, not just
an idle-read timeout: elapsed time while the caller pauses an SSE stream also
counts, but the caller is not cancelled during that pause. Only a direct
connection failure or connect timeout is
certified pre-send and retry-safe; write/read/pool timeouts, response errors,
and interrupted streams are not. The streaming HTTP server and external
provider CI profile remain roadmap work.
