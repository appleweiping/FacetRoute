# Security policy

FacetRoute's offline workflows and decision endpoint do not call providers.
Catalogs, provider bindings, profiles, traces, feedback logs, bandit state, and
inbound HTTP requests still cross a trust boundary. Validate their provenance,
keep raw prompts and personal data out of public fixtures, and apply normal
filesystem access controls to local state.

The optional HTTP service binds to loopback by default, reads an optional
bearer token only from the named environment variable, caps body size and
concurrency, and emits no access log. For non-loopback production use, place it
behind TLS and stronger authentication. The socket timeout bounds I/O, not
arbitrary custom-router execution time.

The optional chat-completions proxy is disabled unless `--providers` is set.
Provider URLs and upstream model names come only from local startup
configuration, keys come only from named environment variables, plain HTTP is
loopback-only by default, and upstream errors are redacted. Operators remain
responsible for provider trust, prompt/data policy, credential rotation, egress
controls, TLS termination, rate limits, and retention. FacetRoute deliberately
does not retry generation requests automatically because doing so can duplicate
non-idempotent work.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
