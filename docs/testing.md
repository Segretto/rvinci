# Testing Guidelines

This document defines how testing works in the `rvinci` repository.

Testing must reinforce architectural boundaries, not bypass them.

----------------------------------------------------------------------
1. Testing Philosophy
----------------------------------------------------------------------

The repository follows a layered architecture:

    core → libs → skills → projects

Testing must respect this layering.

Expectations:

- Core, libs, and skills require unit tests.
- Projects require at least smoke tests.
- Heavy or hardware-dependent tests must be explicitly marked.
- Tests must work when the package is installed (pip install -e .).
- Never modify sys.path in tests.

----------------------------------------------------------------------
2. Directory Structure
----------------------------------------------------------------------

Tests mirror the repository structure:

tests/
├── core/
├── libs/
├── skills/
├── integration/
├── e2e/
└── conftest.py

Meaning:

- core/         → tests for src/rvinci/core
- libs/         → tests for src/rvinci/libs
- skills/       → tests for src/rvinci/skills
- integration/  → component-level integration tests
- e2e/          → end-to-end CLI or pipeline tests

Do not mix test types.

----------------------------------------------------------------------
3. Test Types
----------------------------------------------------------------------

Unit Tests

- Fast (<1s ideally)
- No network
- No large files
- No GPU required
- No external services

Required for:
- core
- libs
- skills public APIs


Integration Tests

- Test interaction between components
- May use small sample data
- Must not require GPU unless explicitly marked

Example:

    import pytest

    @pytest.mark.integration
    def test_something():
        ...


End-to-End (E2E) Tests

- Run full CLI pipelines
- May use subprocess
- Must be explicitly marked

    @pytest.mark.e2e
    def test_cli_smoke():
        ...

----------------------------------------------------------------------
4. Hardware and Heavy Dependency Markers
----------------------------------------------------------------------

Heavy tests must be marked.

Available markers:

- integration
- e2e
- gpu
- robot

Examples:

    @pytest.mark.gpu
    def test_gpu_inference():
        ...

    @pytest.mark.robot
    def test_spot_runtime():
        ...

These tests do NOT run by default.

----------------------------------------------------------------------
5. Running Tests
----------------------------------------------------------------------

Default (safe):

    uv run pytest

Runs only fast tests.

Run integration tests:

    uv run pytest -m integration

Run GPU tests:

    uv run pytest -m gpu

Run robot tests:

    uv run pytest -m robot

Run everything:

    uv run pytest -m "integration or e2e or gpu or robot"

----------------------------------------------------------------------
6. CLI Testing Pattern
----------------------------------------------------------------------

When testing a project CLI:

    import subprocess

    def test_project_cli_smoke():
        result = subprocess.run(
            ["uv", "run", "python", "-m", "rvinci.projects.example.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

Never call internal scripts directly by path.

Always use:

    python -m rvinci.projects.<project>.cli

----------------------------------------------------------------------
7. What Tests Must NOT Do
----------------------------------------------------------------------

Tests must not:

- Modify sys.path
- Assume current working directory
- Download datasets
- Load full pretrained weights
- Require Hydra inside skills
- Depend on external robots unless marked robot
- Depend on CUDA unless marked gpu

If a test requires assets, use small samples from:

    data/samples/

----------------------------------------------------------------------
8. Sample Data Policy
----------------------------------------------------------------------

All test data must be:

- Small
- Version-controlled
- Located in data/samples/

Large datasets must not be used in tests.

----------------------------------------------------------------------
9. Pytest Configuration
----------------------------------------------------------------------

Markers must be declared in pyproject.toml:

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    addopts = "-ra"
    markers = [
      "integration: component-level integration tests",
      "e2e: end-to-end CLI tests",
      "gpu: requires CUDA/GPU",
      "robot: requires robot SDK/hardware",
    ]

----------------------------------------------------------------------
10. Minimum Quality Expectations
----------------------------------------------------------------------

- Core must have strong unit coverage.
- Skills must test public APIs.
- Projects must include at least one smoke test.
- CI must pass:

    uv run pytest

before merging.

Testing exists to protect architectural boundaries and ensure reproducibility.

If unsure where a test belongs, consult:

    docs/architecture.md

----------------------------------------------------------------------
11. Test Fixtures and Generated Artifacts
----------------------------------------------------------------------

To ensure tests are deterministic, reproducible, and clean, this repository
distinguishes between:

1) Static test fixtures (mock inputs)
2) Generated artifacts (outputs created during tests)

----------------------------------------------------------------------
Static Test Fixtures
----------------------------------------------------------------------

All mock JSON files, small images, toy annotations, and other static inputs
used for testing must live under:

    tests/fixtures/

Recommended structure:

tests/
  fixtures/
    core/
    libs/
    skills/
    integration/
    e2e/

Example:

    tests/fixtures/libs/utils/coco_toy.json
    tests/fixtures/skills/perception/segmentation/rgb_0001.png
    tests/fixtures/skills/perception/segmentation/mask_0001.png
    tests/fixtures/skills/perception/segmentation/intrinsics.json

Guidelines:

- Fixtures must be small and version-controlled.
- Prefer minimal “toy” examples.
- Do not include large datasets.
- Use clear naming:
    *_toy.json
    *_small.json
    *_edgecase_*.json
- Fixtures are read-only. Tests must never modify files inside tests/fixtures/.

If a file is only used for tests, it belongs in tests/fixtures/,
not in data/samples/.

----------------------------------------------------------------------
Accessing Fixtures
----------------------------------------------------------------------

Tests must not assume the current working directory.

Use a path relative to the test file location or a helper in tests/conftest.py.

Example pattern:

    from pathlib import Path

    FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

    def test_example():
        path = FIXTURES / "libs" / "utils" / "coco_toy.json"
        ...

Avoid hardcoded absolute paths.

----------------------------------------------------------------------
Generated Artifacts (Temporary Outputs)
----------------------------------------------------------------------

Artifacts created during tests (predictions, logs, run directories, etc.)
must NOT be written to:

- runs/
- models/
- repo root
- tests/fixtures/

Instead, use pytest temporary directories:

    def test_writer(tmp_path):
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        ...

Rules:

- Always use tmp_path or tmp_path_factory.
- Do not persist generated files.
- Tests must leave the repository unchanged after execution.

----------------------------------------------------------------------
Large Data Policy
----------------------------------------------------------------------

Test fixtures must remain lightweight.

If a test requires large data, it should:

- Be marked appropriately (e.g., integration, gpu)
- Use externally managed datasets
- Not commit large assets to the repository

Testing must remain fast, deterministic, and self-contained.
