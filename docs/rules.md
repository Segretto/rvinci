> **AGENT DIRECTIVE:** You are maintaining the `rvinci` monorepo.
> 1. **Namespace Rigor:** Import via `rvinci.<module>`. No relative imports.
> 2. **Config Safety:** Wrap Hydra configs in Pydantic schemas immediately.
> 3. **Context Strategy:** `rules.md` is your constitution. If you need implementation details (code snippets, directory trees), YOU MUST REQUEST the relevant documentation file listed below.

# rvinci Repository Rules

This file defines the non-negotiable architectural rules of rvinci.

If a change violates this document, the change is incorrect.

For procedural details see:
- docs/dev_env.md
- docs/runs_and_artifacts.md
- docs/model_registry.md
- docs/architecture.md


----------------------------------------------------------------------
1. Namespace Is Law
----------------------------------------------------------------------

All code lives under:

    src/rvinci/

All imports must use the installed namespace:

    from rvinci.core.logging import get_logger

Never import using:
- src.*
- relative ../../ paths
- assumptions about repo-root execution

The package must work after:

    pip install -e .

If it only works from the repository root, it is incorrect.

Every folder under rvinci/ must contain an __init__.py.


----------------------------------------------------------------------
2. Architectural Layers
----------------------------------------------------------------------

Dependency flow must always go downward:

    core → libs → skills → projects

Layer definitions:

core
    - Lightweight primitives only
    - No torch, no OpenCV, no SDKs
    - Geometry, logging, config bridge, artifact helpers

libs
    - Reusable utilities
    - May depend on core
    - May include heavier deps

skills
    - Reusable deep-learning capabilities
    - May depend on libs + core
    - May depend on other skills via public API only
    - Must NOT depend on projects
    - Must NOT require Hydra

projects
    - Applications / experiments
    - May depend on skills + libs + core
    - Own CLI entrypoints
    - Own Hydra configs
    - Must NOT import other projects

Forbidden:

- projects → projects
- skills → projects
- core → libs / skills / projects


----------------------------------------------------------------------
3. Public API Rule (Skills)
----------------------------------------------------------------------

If a skill is reused by another skill or project,
it must expose a stable public API:

    rvinci.skills.<domain>.<skill>.api

Never import internal modules across skill boundaries.

If internal code must be reused:
    promote it to libs.


----------------------------------------------------------------------
4. Configuration Gate
----------------------------------------------------------------------

Hydra is allowed ONLY in projects.

Skills must remain framework-agnostic.

In project CLI:

1. Hydra composes config
2. OmegaConf resolves interpolations
3. Convert to Pydantic schema immediately
4. Raw DictConfig must NOT leave cli.py

Example pattern:

    config = validate_config(cfg, ProjectConfig)

Typed config flows downward.
Hydra never leaks into skills.


----------------------------------------------------------------------
5. Run Ownership Rule
----------------------------------------------------------------------

Only projects create run directories.

Skills may write artifacts only within a project-provided run path.

See:
    docs/runs_and_artifacts.md


----------------------------------------------------------------------
6. Model Ownership Rule
----------------------------------------------------------------------

Model weights are artifacts, not source code.

Reusable models live under:

    repo_root/models/

They are not committed to git.

Projects must not hardcode absolute weight paths.

See:
    docs/model_registry.md


----------------------------------------------------------------------
7. Logging & Error Discipline
----------------------------------------------------------------------

Use:

    rvinci.core.logging.get_logger

Library code:
    - Do not catch broad exceptions.

CLI boundary:
    - One broad catch allowed.
    - Must re-raise after logging.


----------------------------------------------------------------------
8. Reproducibility Contract
----------------------------------------------------------------------

Every experiment must be reproducible via:

- config_resolved.yaml
- git_hash
- uv.lock
- correct model version

If an experiment cannot be reconstructed,
the implementation is incomplete.

See:
    docs/dev_env.md


----------------------------------------------------------------------
9. Pre-Commit Checklist
----------------------------------------------------------------------

Before committing:

- No src.* imports
- No project-to-project imports
- No skill-to-project imports
- Raw DictConfig does not escape CLI
- Heavy dependencies isolated
- Artifacts follow canonical structure
- No model weights committed
- Tests pass


----------------------------------------------------------------------
10. Guiding Principle
----------------------------------------------------------------------

rvinci is not a collection of scripts.

It is a modular robotics capability framework.

If adding something:
- Ask which layer it belongs to.
- Respect dependency gravity.
- Keep skills reusable.
- Keep projects isolated.
- Preserve reproducibility.

When in doubt: simplify.


For full architectural layout, templates, and extension guidelines see:
    docs/architecture.md