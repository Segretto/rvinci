# rvinci Architecture Guide

This document explains how rvinci is structured and how to extend it correctly.

For non-negotiable rules see:
    rules.md


----------------------------------------------------------------------
1. Architectural Philosophy
----------------------------------------------------------------------

rvinci separates reusable capabilities from applications.

We distinguish four layers:

    core → libs → skills → projects

Dependencies must always flow downward.

This prevents:
- Circular dependencies
- Experiment glue leaking into reusable code
- Architecture collapse over time


----------------------------------------------------------------------
2. Repository Layout (Template)
----------------------------------------------------------------------

src/rvinci/
├── core/        # Strict, lightweight, stable
├── libs/        # Shared utilities
├── skills/      # Reusable deep-learning capabilities
└── projects/    # Isolated applications / experiments


Expanded example:

src/rvinci/
    core/
    libs/
    skills/
        perception/
            segmentation/
            depth/
            multitask_dino/
        navigation/
            elevation_mapping/
            traversability/
        manipulation/
            grasping/
            visuomotor/
        locomotion/
            terrain_perception/
    projects/
        spot_pick_place/
        field_nav_stack/
        stereo_elevation_demo/
        dinov3_benchmark/

[Projects]  (Application Glue, Hydra, CLI)
        │
        ▼
    [Skills]    (Deep Learning, Pydantic, No-Hydra)
        │
        ▼
     [Libs]     (Shared Utils, Robot Wrappers)
        │
        ▼
     [Core]     (Primitives, Logging, Config Bridge)

----------------------------------------------------------------------
3. Layer Responsibilities
----------------------------------------------------------------------

core
-----

Contains:
- Geometry primitives
- Logging
- Path helpers
- Config validation bridge
- Artifact utilities

Must remain lightweight.
No torch, no SDKs, no heavy deps.


libs
-----

Reusable domain utilities.

Examples:
- dataset loaders
- visualization helpers
- robot SDK wrappers

May depend on core.


skills
-------

Reusable deep learning capabilities.

Each skill should contain:

    skill_name/
        api.py
        schemas/
        models/
        training/
        inference/
        outputs/

Rules:
- Expose stable public API via api.py
- No Hydra usage
- Accept typed configs (Pydantic)
- May depend on other skills via public API only


projects
---------

Application-level orchestration.

Each project should contain:

    project_name/
        cli.py
        configs/
        schemas/
        pipeline/
        outputs/

Projects:
- Own Hydra
- Own experiment runs
- Compose skills
- May not be imported by other projects


----------------------------------------------------------------------
4. Canonical CLI Pattern
----------------------------------------------------------------------

Hydra must only exist inside projects.

Example:

# src/rvinci/projects/<project_name>/cli.py
```python
import hydra
from omegaconf import DictConfig
from rvinci.core.config import validate_config
from rvinci.core.logging import get_logger
from rvinci.projects.<project_name>.schemas.config import ProjectConfig

log = get_logger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        config: ProjectConfig = validate_config(cfg, ProjectConfig)
        run_pipeline(config)
    except Exception:
        log.exception("Run failed")
        raise

```

Rules:

- Raw DictConfig must NOT leave cli.py
- Convert immediately to Pydantic
- Skills remain Hydra-agnostic


----------------------------------------------------------------------
5. Hydra Working Directory Rule
----------------------------------------------------------------------

Hydra may change the working directory.

Therefore:

- Resolve paths relative to:
    hydra.utils.get_original_cwd()
  OR
    rvinci.core.paths.get_repo_root()

- Prefer rvinci.core.paths over calling hydra.utils.get_original_cwd() directly.

- Prefer disabling:
    hydra.job.chdir = false

Never assume current working directory is repo root.


----------------------------------------------------------------------
6. How to Add a New Skill
----------------------------------------------------------------------

1. Create folder under:

    src/rvinci/skills/<domain>/<skill_name>/

2. Add:

    api.py
    schemas/
    models/
    training/ (optional)
    inference/ (optional)

3. Ensure:
    - No Hydra usage
    - Public API defined
    - No project imports


----------------------------------------------------------------------
7. How to Add a New Project
----------------------------------------------------------------------

1. Create folder under:

    src/rvinci/projects/<project_name>/

2. Add:

    cli.py
    configs/
    schemas/
    pipeline/

3. Ensure:
    - Hydra lives here
    - Config validated immediately
    - Runs follow canonical structure
    - No project-to-project imports


----------------------------------------------------------------------
8. Where to Put Code
----------------------------------------------------------------------

If code is:

Reusable across multiple domains → libs

Reusable DL capability → skills

Experiment-specific glue → projects

Foundational primitive → core

If unsure:
    Place higher in the stack only if reuse is proven.
    Otherwise, keep local to project.
