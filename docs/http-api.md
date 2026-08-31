# HTTP routing contract

The optional server makes local routing decisions. It does not authenticate to
providers, proxy traffic, execute a model, stream tokens, or return a chat
completion.

## Endpoints

- `GET /health` returns `{"status":"ok"}`.
- `GET /v1/models` lists routing-visible capabilities, regions, context limits,
  and enabled state. Arbitrary catalog metadata and prices are omitted.
- `POST /v1/route` accepts strict JSON and returns a complete `decision` plus
  the selected ID under `model`.

The native body uses `RouteRequest` fields. As an integration convenience, a
body with `messages` accepts text strings or text parts, a completion token
limit, tools presence, and `response_format` of `text`, `json_object`, or
`json_schema`. Images, audio, non-text tool results, completion-generation
fields, and unknown keys are rejected. The result object is
`routing.decision` and never contains `choices`.

## Operational bounds

- loopback bind by default;
- non-loopback bind requires the named bearer-token environment variable
  unless an explicit unsafe override is passed;
- constant-time bearer comparison;
- configurable body bytes, active concurrency, and socket read timeout;
- `Content-Length` and JSON content type required; transfer encoding rejected;
- duplicate keys, non-finite numbers, malformed UTF-8, and unknown fields
  rejected;
- `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, and an opaque
  request ID on structured responses;
- access logging disabled so prompts and authorization headers are not emitted.

The socket timeout limits request I/O; it cannot safely interrupt arbitrary
custom Python router code. Use a process supervisor and TLS-authenticating
reverse proxy for an untrusted or internet-facing deployment.

## Status codes

| Status | Meaning |
|---:|---|
| 200 | Successful health, catalog, or decision. |
| 400 | Malformed/ambiguous JSON, length, or transfer encoding. |
| 401 | Incorrect configured bearer token. |
| 404 | Unknown path. |
| 411 | Missing `Content-Length`. |
| 413 | Body exceeds the configured limit. |
| 415 | Content type is not JSON. |
| 422 | JSON cannot form a request or no model is eligible. |
| 500 | Unexpected router failure; details are redacted. |
| 503 | Concurrent routing capacity is full. |
