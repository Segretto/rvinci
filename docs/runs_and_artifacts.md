# Runs & Artifacts Specification

This document defines how experiments store outputs.

For architectural layering and dependency rules, see:
    rules.md


----------------------------------------------------------------------
1. Ownership Rules
----------------------------------------------------------------------

Only projects create run directories.

Skills:
- Must not create standalone run folders
- May write artifacts only within a project-provided run directory

This prevents uncontrolled artifact sprawl.


----------------------------------------------------------------------
2. Canonical Run Directory Structure
----------------------------------------------------------------------

All experiments must follow this structure:

runs/
  <project_name>/
    <YYYY-MM-DD_HH-MM-SS>_<git_hash>/
      ├── config.yaml
      ├── config_resolved.yaml
      ├── run_manifest.json
      ├── metrics.json
      ├── artifacts/
      │   ├── predictions/
      │   │   ├── frame_0000.npz
      │   │   └── metadata.json
      │   └── checkpoints/
      └── logs/


----------------------------------------------------------------------
3. Required Files
----------------------------------------------------------------------

config.yaml
    The composed Hydra configuration (may contain interpolations).

config_resolved.yaml
    Fully resolved configuration with no ${...} fields.

run_manifest.json must include:

- schema_version
- project_name
- git_hash
- package_versions
- optional:
    - cuda version
    - torch version
    - device info

metrics.json
    Scalar metrics or time-series values.
    Must be JSON serializable.


----------------------------------------------------------------------
4. Predictions Contract
----------------------------------------------------------------------

All saved predictions must include:

artifacts/predictions/
    frame_<id>.npz
    metadata.json

metadata.json must include:

- frame_id
- timestamp
- intrinsics (or reference)
- contract_version


----------------------------------------------------------------------
5. Serialization Rules
----------------------------------------------------------------------

Allowed formats:

- .npz for numpy arrays
- .json for metadata
- .yaml for configs

Forbidden:

- Pickle of custom classes
- Torch serialized Python objects outside checkpoints


----------------------------------------------------------------------
6. Checkpoints
----------------------------------------------------------------------

Training checkpoints must live under:

    artifacts/checkpoints/

Recommended:

- best.pt
- last.pt

Projects may optionally promote a model to:

    repo_root/models/

See:
    docs/model_registry.md


----------------------------------------------------------------------
7. Reproducibility Requirements
----------------------------------------------------------------------

Every run must be reproducible by:

1) Loading config_resolved.yaml
2) Using the recorded git_hash
3) Installing dependencies from uv.lock
4) Loading the correct model version


----------------------------------------------------------------------
8. Principles
----------------------------------------------------------------------

- runs/ is experimental output
- models/ is curated reusable artifacts
- No hardcoded absolute paths
- Every artifact must be traceable
- Every experiment must be reconstructable
