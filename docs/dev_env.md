# Development Environment Guide

This document describes how to set up and manage development
environments for rvinci.

For architectural rules and dependency gravity, see:
    rules.md


----------------------------------------------------------------------
1. Dependency Management Philosophy
----------------------------------------------------------------------

rvinci uses uv for:

- Deterministic dependency resolution
- Lockfile-based reproducibility
- Optional dependency groups
- Virtual environment isolation

We use a single installable package with optional groups.

This allows:
- Clean separation of perception, mapping, manipulation stacks
- GPU vs CPU builds
- Robot-specific SDK isolation
- Never use absolute paths (e.g. `/home/user/...`) in `pyproject.toml` for local or editable dependencies. Use relative paths for purely local dependencies, and prefer direct `git` URLs (e.g. `{ git = "https://github.com/..." }`) for open source projects.


----------------------------------------------------------------------
2. Installation (Standard Workflow)
----------------------------------------------------------------------

Install development tools:

    uv sync --group dev

Install optional capability groups:

    uv pip install -e ".[dinov3]"
    uv pip install -e ".[stereo]"
    uv pip install -e ".[spot]"
    uv pip install -e ".[viz]"

Combine groups if needed:

    uv pip install -e ".[dinov3,viz]"


----------------------------------------------------------------------
3. Multi-Environment Strategy (Recommended for Robotics)
----------------------------------------------------------------------

Robotics stacks often require incompatible dependencies.

Create separate virtual environments per domain:

    uv venv .venv-dinov3
    source .venv-dinov3/bin/activate
    uv pip install -e ".[dinov3]"
    deactivate

    uv venv .venv-spot
    source .venv-spot/bin/activate
    uv pip install -e ".[spot]"
    deactivate

    uv venv .venv-stereo
    source .venv-stereo/bin/activate
    uv pip install -e ".[stereo]"
    deactivate

This allows:

- CUDA-specific environments
- Robot-specific SDK builds
- Clean dependency isolation


----------------------------------------------------------------------
4. Lockfile Discipline
----------------------------------------------------------------------

uv.lock must be committed.

Rules:

- Do not manually edit uv.lock
- Update dependencies via:
      uv add <package>
      uv remove <package>
- Always re-run:
      uv sync

This ensures reproducibility across machines.


----------------------------------------------------------------------
5. GPU / CUDA Notes
----------------------------------------------------------------------

For GPU-based skills:

- Ensure correct CUDA toolkit compatibility
- Prefer pinned torch versions
- Test environment with:

      python -c "import torch; print(torch.cuda.is_available())"

If deployment differs from training machine,
use separate environments.


----------------------------------------------------------------------
6. Pre-Commit and Testing
----------------------------------------------------------------------

Before pushing changes:

    uv run pytest
    uv run ruff check .
    uv run ruff format .

All code must pass tests and lint checks.


----------------------------------------------------------------------
7. Docker (Optional)
----------------------------------------------------------------------

Dockerfiles exist under:

    docker/

Use Docker for:

- Reproducible GPU training
- Robot runtime isolation
- CI consistency

Docker is recommended for deployment,
not required for development.


----------------------------------------------------------------------
8. Principles
----------------------------------------------------------------------

- Never rely on system Python
- Never mix robot SDKs with training environments
- Keep environments focused and minimal
- Use optional groups rather than adding everything globally
