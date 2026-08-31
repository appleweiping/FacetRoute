# Security policy

FacetRoute does not call providers. Catalogs, profiles, traces, feedback logs,
bandit state, and inbound HTTP requests still cross a trust boundary. Validate
their provenance, keep raw prompts and personal data out of public fixtures,
and apply normal filesystem access controls to local state.

The optional HTTP service binds to loopback by default, reads an optional
bearer token only from the named environment variable, caps body size and
concurrency, and emits no access log. For non-loopback production use, place it
behind TLS and stronger authentication. The socket timeout bounds I/O, not
arbitrary custom-router execution time.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
