# Changelog

## Unreleased

- Process‑parallel evolution uses lightweight per‑iteration snapshots (parent + curated context + artifacts) for better scalability.
- Meta‑prompting and RL‑based parent selection now learn online in process‑parallel runs and persist via checkpoints.
- Prompt context includes truly diverse MAP‑Elites elites, improving quality‑diversity search.
- Diff‑based evolution is more robust to whitespace/mismatch, reducing no‑op mutations.
- Novelty/embedding edge cases fixed; population capping preserves MAP‑Elites coverage before pruning elites.
