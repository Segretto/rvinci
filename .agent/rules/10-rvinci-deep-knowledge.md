---
trigger: model_decision
description: Apply this rule when detailed repository conventions or canonical templates are required.
---

# .agent/rules/10-rvinci-deep-knowledge.md

Use this rule to pull in deep reference docs when you need detailed patterns (directory templates, run contracts, model registry semantics, uv workflows). Use the `view_file` tool on the explicitly provided file paths below.

## References (use `view_file` on these paths)

- **Architecture and layout templates:** `docs/architecture.md`
- **Runs & artifacts canonical structure:** `docs/runs_and_artifacts.md`
- **Model registry layout and referencing rules:** `docs/model_registry.md`
- **Development environment and uv usage:** `docs/dev_env.md`
- **Testing guidelines:** `docs/testing.md`
- **Porting existing scripts:** `docs/porting.md`
- **Rules overview:** `docs/rules.md`

## How to use this rule
1. Identify the exact question (file placement, CLI pattern, run format, model storage).
2. Use `view_file` to read ONLY the relevant document(s) listed above.
3. Apply the documented pattern exactly, keeping changes consistent with `00-rvinci-constitution.md`.