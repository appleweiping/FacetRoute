# Security policy

FacetRoute does not call providers or read API keys, but catalogs, profiles, feedback logs, and bandit state still cross a trust boundary. Validate their provenance, keep raw prompts and personal data out of public fixtures, and apply normal filesystem access controls to local state.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
