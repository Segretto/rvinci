# R-VINCI — Robotics Vision & Skill Framework

R-VINCI is a modular deep-learning framework for robotics vision and skill development.

It is designed to:

- Build reusable deep-learning capabilities ("skills")
- Compose them into reproducible experiments ("projects")
- Enforce strict architectural layering
- Prevent research code rot
- Maintain clean dependency boundaries
- Support robotics deployment environments

This repository is not just a collection of scripts — it is a structured, layered system for developing vision-driven robotic capabilities.

----------------------------------------------------------------------
Architecture Overview
----------------------------------------------------------------------

The repository follows a strict dependency flow:

    core → libs → skills → projects

- **core/**  
  Lightweight, stable primitives (logging, paths, config bridges, geometry).  
  No heavy dependencies.

- **libs/**  
  Reusable domain utilities (vision IO, mask tools, visualization, bridges).  
  May depend on core.

- **skills/**  
  Deep-learning capabilities (segmentation, depth, mapping, manipulation components).  
  Skills are Hydra-agnostic and expose stable public APIs.

- **projects/**  
  Experiment pipelines and application stacks.  
  Projects orchestrate skills using Hydra and generate structured run artifacts.

This separation ensures:

- Reusability
- Isolation of heavy dependencies
- Clear API contracts
- Clean reproducibility boundaries

See `docs/architecture.md` for full details.

----------------------------------------------------------------------
Installation
----------------------------------------------------------------------

This project uses `uv` for dependency management.

1) Install uv (if needed):

    curl -LsSf https://astral.sh/uv/install.sh | sh

2) Sync dependencies:

    uv sync

The package is installed in editable mode and must be imported via:

    import rvinci

Do not modify sys.path to fix imports.

----------------------------------------------------------------------
Running Projects
----------------------------------------------------------------------

Projects must be executed as modules.

Example:

    uv run python -m rvinci.projects.<project_name>.cli

Never run internal files directly by path.

Example (legacy vision suite):

    uv run python -m rvinci.projects.vision_suite_legacy.cli analyze \
        --gt path/to/gt.json \
        --pred path/to/pred.json

Projects are responsible for:
- Hydra configuration
- Run directory creation
- Artifact generation
- Skill composition

Skills must remain orchestration-agnostic.

----------------------------------------------------------------------
Testing
----------------------------------------------------------------------

Testing is based on pytest.

Run default (fast) tests:

    uv run pytest

Run integration tests:

    uv run pytest -m integration

Run GPU tests:

    uv run pytest -m gpu

See `docs/testing.md` for full testing policy.

----------------------------------------------------------------------
Model Artifacts
----------------------------------------------------------------------

Model weights are artifacts, not source code.

- Stored locally under `models/`
- Not committed to git
- Referenced by projects
- Managed according to `docs/model_registry.md`

----------------------------------------------------------------------
Development Guidelines
----------------------------------------------------------------------

Before committing:

    uv run ruff format .
    uv run ruff check .
    uv run pytest

Follow architectural rules in:

    docs/architecture.md
    docs/runs_and_artifacts.md
    docs/dev_env.md

----------------------------------------------------------------------
Contributing
----------------------------------------------------------------------

1) Respect dependency layering.
2) Keep skills reusable and Hydra-agnostic.
3) Add unit tests for core, libs, and skills.
4) Add smoke tests for projects.
5) Never patch sys.path.

If unsure about structure or conventions, consult:

    docs/architecture.md
    docs/testing.md

R-VINCI is designed to scale from research prototypes to real robotic systems while maintaining clarity, modularity, and reproducibility.
