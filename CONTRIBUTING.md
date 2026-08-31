# Contributing to FacetRoute

FacetRoute welcomes focused fixes, tests, documentation, and new policies that
preserve its offline-first and explainable design.

## Development setup

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check src tests examples
mypy -p facetroute
pytest --cov=facetroute --cov-branch
python -m build
```

## Change guidelines

- Keep the required runtime dependency list empty. Proposals for a runtime
  dependency need an issue explaining why the standard library is insufficient.
- Apply hard constraints before any learned or heuristic score.
- Preserve deterministic tie-breaking and seeded simulations.
- Every new decision factor must appear in `RouteDecision` or its explanation.
- Do not add provider network calls, API keys, telemetry, or silent persistence.
- Persist versioned JSON that can be inspected and replayed without FacetRoute.
- Add tests for success, invalid input, deterministic behavior, and state
  round-trips. Tests must not require network access.
- Public functions and classes require type annotations and concise docstrings.

## Pull request scope

Prefer one behavior change per pull request. Include:

1. the user-visible problem;
2. the chosen behavior and alternatives considered;
3. exact commands run and their results;
4. compatibility or persistence implications;
5. disclosure of material automated or AI assistance.

Generated output, secrets, local feedback logs, and bandit state do not belong
in commits. Review the complete diff and remain able to explain every line.

## Security and privacy

Model catalogs and feedback can reveal infrastructure or user preferences.
Use synthetic fixtures in tests and examples. Report vulnerabilities privately
to the repository maintainers instead of publishing exploit details first.
