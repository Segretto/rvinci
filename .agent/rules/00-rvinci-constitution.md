---
trigger: always_on
---

These rules apply to all work in this workspace.

----------------------------------------------------------------------
1) Namespace Is Law
----------------------------------------------------------------------

- All code lives under `src/rvinci/`.
- Import only via `rvinci.*` (never `src.*` or relative `../../` imports).
- The package must work after `pip install -e .`.
- Never modify `sys.path` to fix imports. If imports fail, fix packaging or execution.

----------------------------------------------------------------------
2) Dependency Gravity (Non-Negotiable)
----------------------------------------------------------------------

Dependencies must always flow downward:

    core → libs → skills → projects

Forbidden:
- projects → projects
- skills  → projects
- core    → libs/skills/projects

----------------------------------------------------------------------
3) Skills vs Projects
----------------------------------------------------------------------

- Skills are reusable deep-learning capabilities (library-like).
- Projects are orchestration/experiments (Hydra + runs/ outputs).
- Skills must expose a stable public API at:
      `rvinci.skills.<domain>.<skill>.api`
- Never import internal modules across skill boundaries. If code must be shared, promote it to `libs/`.
- Projects must not be imported by other projects.

----------------------------------------------------------------------
4) Configuration Gate
----------------------------------------------------------------------

- Hydra is allowed ONLY in `projects`.
- Raw `DictConfig` must NOT leave `projects/<name>/cli.py`.
- Convert config immediately to typed Pydantic models and pass typed configs downward.
- Skills must remain Hydra-agnostic.

----------------------------------------------------------------------
5) Runs and Models
----------------------------------------------------------------------

- Only projects create run directories under `runs/`.
- Skills may write artifacts only within a project-provided run path.
- Model weights are artifacts (not source). Store locally under `repo_root/models/` (not in git).
- Do not hardcode absolute paths to weights.

----------------------------------------------------------------------
6) Canonical Execution and Tests
----------------------------------------------------------------------

- Run project entrypoints via module execution:
      `uv run python -m rvinci.projects.<project_name>.cli`
- Never run internal files directly by path for normal operation.
- Default checks before committing:
      `uv run ruff format .`
      `uv run ruff check .`
      `uv run pytest`

----------------------------------------------------------------------
7) When Unsure
----------------------------------------------------------------------

If unsure where code belongs, how to structure outputs, or how to format artifacts:
- Use the `rvinci-docs` skill, or
- Activate the deep reference rule: `10-rvinci-deep-knowledge.md`.