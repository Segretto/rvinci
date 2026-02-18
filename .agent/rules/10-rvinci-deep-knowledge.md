---
trigger: model_decision
description: Apply this rule when detailed repository conventions or canonical templates are required.
---

# .agent/rules/10-rvinci-deep-knowledge.md
# Activation: Manual (recommended) or Model Decision
#
# Purpose: Pull in deep reference docs only when needed.
# This rule intentionally references repository docs via @mentions.

# rvinci Deep Knowledge (Reference)

Use this rule when you need detailed patterns or exact specifications
(directory templates, run contracts, model registry semantics, uv workflows).

## References (read as needed)

### Architecture and layout templates
@../../docs/architecture.md

### Runs & artifacts canonical structure and required files
@../../docs/runs_and_artifacts.md

### Model registry layout, manifests, and referencing rules
@../../docs/model_registry.md

### Development environment and uv usage
@../../docs/dev_env.md

### Testing
@../../docs/testing.md

## How to use this rule
1. Identify the exact question (file placement, CLI pattern, run format, model storage, env setup).
2. Read only the relevant referenced doc section(s).
3. Apply the documented pattern exactly.
4. Keep changes consistent with `00-rvinci-constitution.md`.