# Vision Suite Legacy

## Purpose
This project houses legacy scripts migrated from the original `vision_suite` package. It provides a canonical entrypoint for running these scripts within the `rvinci` architecture.

## Composition
This project composes the functionality previously found in `scripts/`, using libraries from:
- `rvinci.libs.vision_data`
- `rvinci.libs.inference`
- `rvinci.libs.visualization`
- `rvinci.libs.utils`

## Usage

Run via the unified CLI:
```bash
uv run python -m rvinci.projects.vision_suite_legacy.cli project.name=vision_suite_legacy
```

Or using the installed script alias:
```bash
uv run rvinci-legacy
```

## Outputs
Outputs are stored in `runs/vision_suite_legacy/<date>_<git_hash>/` by default.
