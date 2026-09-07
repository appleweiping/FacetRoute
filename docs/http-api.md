# HTTP routing and chat-completions contract

FacetRoute exposes a bounded standard-library HTTP service. Routing remains
provider independent; chat execution exists only when the operator supplies a
provider registry at startup.

## Endpoints

- `GET /health` returns `{"status":"ok"}`.
- `GET /v1/models` lists routing-visible capabilities, regions, context limits,
  and enabled state. Arbitrary catalog metadata and prices are omitted.
- `POST /v1/route` returns a complete `routing.decision` and selected catalog
  model ID without calling a provider.
- `POST /v1/chat/completions` routes and executes a text chat request through a
  fixed OpenAI-compatible provider binding. Without `--providers`, it returns a
  structured `503 provider_not_configured` error.

## Provider registry

The registry is local operator configuration, never caller input:

```json
{
  "models": [
    {
      "model_id": "marble-reasoner",
      "upstream_model": "vendor/reasoner-v1",
      "base_url": "https://api.vendor.example/v1",
      "api_key_env": "VENDOR_API_KEY"
    }
  ]
}
```

Every enabled routing-catalog model must have exactly one `model_id` binding;
unknown or missing enabled IDs fail startup. `upstream_model` is the only model
name sent upstream. `base_url` must end in `/v1`; it cannot contain credentials,
a query, or a fragment. Plain HTTP is loopback-only unless the operator passes
`--allow-insecure-provider-http`. `api_key_env` is optional for unauthenticated
local servers; when present, the variable must exist and its value is sent as a
Bearer token. Literal keys are not part of the schema.

Provider response, stream, event, and timeout limits are configurable with:

- `--provider-timeout` (default 60 seconds);
- `--max-provider-response-bytes` (default 16 MiB);
- `--max-provider-stream-bytes` (default 64 MiB);
- `--max-provider-event-bytes` (default 1 MiB).

## Routing requests

The native `/v1/route` body uses `RouteRequest` fields. As an integration
convenience, a body with `messages` accepts text strings or text parts, a
completion token limit, tools presence, and `response_format` of `text`,
`json_object`, or `json_schema`. Images, audio, non-text tool results,
completion-generation fields, and unknown keys are rejected. Its response is a
`routing.decision`, never a completion.

## Routed chat completions

The chat endpoint accepts the following bounded OpenAI request subset:

- required virtual model `facetroute` and a non-empty text-only `messages`
  array of at most 128 items;
- `max_tokens` or `max_completion_tokens`, but not both;
- `frequency_penalty`, `presence_penalty`, `temperature`, `top_p`, `seed`,
  `stop`, `n`, `logprobs`, `top_logprobs`, and `user`;
- `tools`, `tool_choice`, `parallel_tool_calls`, and `response_format`;
- `stream` and `stream_options.include_usage`.

The optional `facetroute` extension accepts only:

```json
{
  "required_capabilities": ["reasoning"],
  "max_cost_usd": 0.05,
  "max_latency_ms": 1500,
  "region": "us",
  "sensitivity": "normal",
  "task_hint": "math",
  "context_tokens": 4096,
  "metadata": {"tenant": "research"}
}
```

This extension influences routing and is removed before provider execution.
Requests cannot override provider URLs, credentials, or upstream model names.
The selected catalog ID and policy are returned as `X-FacetRoute-Model` and
`X-FacetRoute-Policy` response headers.

Non-streaming upstream responses must be strict JSON with a
`chat.completion` object, non-empty ID and model, and choices array. They are
then returned without inventing or modifying generated content.

Streaming upstream responses must use `text/event-stream` and consist of
bounded `data:` events containing strict `chat.completion.chunk` JSON, followed
by `[DONE]`. FacetRoute validates each chunk and emits a chunked SSE response.
If a provider fails before the first event, the caller receives a normal
structured HTTP error. If it fails after streaming began, FacetRoute emits one
redacted SSE error object, omits `[DONE]`, and closes the connection.

## Operational bounds

- loopback bind by default;
- non-loopback bind requires the named inbound bearer-token environment
  variable unless an explicit unsafe override is passed;
- constant-time inbound bearer comparison;
- configurable request bytes, active concurrency, socket read timeout, and
  provider response limits;
- `Content-Length` and JSON content type required; transfer encoding rejected;
- duplicate keys, non-finite numbers, malformed UTF-8, and unknown fields
  rejected;
- `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, and an opaque
  request ID on structured responses;
- access logging disabled so prompts and authorization headers are not emitted;
- upstream error bodies are discarded and never exposed to callers.

The socket timeout limits request I/O; it cannot safely interrupt arbitrary
custom Python router or provider code. Use a process supervisor plus a TLS and
authentication reverse proxy for an untrusted or internet-facing deployment.
The built-in service does not implement retries: automatic retries can duplicate
non-idempotent generation and should be an explicit application policy.

## Status codes

| Status | Meaning |
|---:|---|
| 200 | Successful health, catalog, decision, completion, or started stream. |
| 400 | Malformed/ambiguous JSON, length, or transfer encoding. |
| 401 | Incorrect configured inbound bearer token. |
| 404 | Unknown path. |
| 411 | Missing `Content-Length`. |
| 413 | Request body exceeds the configured limit. |
| 415 | Request content type is not JSON. |
| 422 | JSON cannot form a request or no model is eligible. |
| 502 | Provider rejected the request or returned a malformed/failed response. |
| 503 | Routing capacity, provider binding, rate limit, or provider availability failure. |
| 504 | Provider timeout. |

Every error message is stable and redacted. Mid-stream failures are represented
inside SSE because HTTP status and headers have already been sent.
