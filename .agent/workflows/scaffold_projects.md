---
description: Scaffold a new rvinci project (Hydra-enabled) with canonical CLI gate, configs layout, schema stubs, and pipeline entrypoint.
---

# Workflow: scaffold-project

**User Input:**  
"Scaffold a new project named {{project_name}} that composes skills {{skills_list}}"

Example:  
"Scaffold a new project named spot_pick_place that composes skills perception.segmentation, manipulation.visuomotor"

## Steps

1. **Validate inputs**
   - Ensure `{{project_name}}` is `snake_case` and filesystem-safe.
   - Ensure each entry in `{{skills_list}}` uses dotted form: `<domain>.<skill>` (e.g., `perception.segmentation`).
   - Confirm the project does NOT duplicate an existing folder under `src/rvinci/projects/`.

2. **Read canonical structure**
   - Read `docs/architecture.md` and follow the "How to Add a New Project" and "Canonical CLI Pattern" sections.
   - Read `docs/runs_and_artifacts.md` and ensure run ownership/structure is respected.

3. **Create project skeleton**
   - Run the following bash snippet to scaffold the directory structure and empty files.
   
   // turbo
   ```bash
   mkdir -p src/rvinci/projects/{{project_name}}/{configs/hydra,schemas,pipeline,outputs}
   touch src/rvinci/projects/{{project_name}}/__init__.py
   touch src/rvinci/projects/{{project_name}}/schemas/__init__.py
   touch src/rvinci/projects/{{project_name}}/pipeline/__init__.py
   touch src/rvinci/projects/{{project_name}}/outputs/__init__.py
   ```

5. **Create `cli.py` (Hydra entrypoint + Config Gate)**
   - Create:
     - `src/rvinci/projects/{{project_name}}/cli.py`
   - Requirements:
     - Hydra is allowed ONLY here (project scope).
     - Accept `DictConfig`, validate immediately into Pydantic schema.
     - Raw `DictConfig` must NOT leave `cli.py`.
     - Use `rvinci.core.logging.get_logger`.
     - Top-level broad exception catch allowed ONLY here; re-raise.
     - Call `run_pipeline(config)` from `pipeline/run.py`.

   - Canonical skeleton:

     ```python
     import hydra
     from omegaconf import DictConfig
     from rvinci.core.config import validate_config
     from rvinci.core.logging import get_logger
     from rvinci.projects.{{project_name}}.schemas.config import ProjectConfig
     from rvinci.projects.{{project_name}}.pipeline.run import run_pipeline

     log = get_logger(__name__)

     @hydra.main(version_base=None, config_path="configs", config_name="config")
     def main(cfg: DictConfig) -> None:
         try:
             config: ProjectConfig = validate_config(cfg, ProjectConfig)
             run_pipeline(config)
         except Exception:
             log.exception("Run failed")
             raise

     if __name__ == "__main__":
         main()
     ```

6. **Create Hydra config structure**
   - Create:
     - `src/rvinci/projects/{{project_name}}/configs/config.yaml`
     - `src/rvinci/projects/{{project_name}}/configs/hydra/default.yaml`

   - Minimal `config.yaml` should include:
     - a defaults list
     - a `project` section (name, run_dir, etc.)
     - a `skills` section listing required skill configs
     - a `hydra` include for safe cwd behavior

   - Example `configs/config.yaml`:

     ```yaml
     defaults:
       - hydra: default
       - _self_

     project:
       name: {{project_name}}
       runs_root: runs
       notes: ""

     skills:
       enabled:
         - {{skills_list}}
     ```

   - Example `configs/hydra/default.yaml` (recommended):

     ```yaml
     hydra:
       job:
         chdir: false
     ```

7. **Create project schema (Pydantic)**
   - Create:
     - `src/rvinci/projects/{{project_name}}/schemas/config.py`

   - Requirements:
     - Define `ProjectConfig` as Pydantic model
     - Include:
       - `project` section (name, runs_root, notes)
       - `skills` section (enabled list + optional per-skill config references)
     - Use `extra="forbid"` if consistent with repo style.

8. **Create pipeline entrypoint**
   - Create:
     - `src/rvinci/projects/{{project_name}}/pipeline/run.py`

   - Requirements:
     - Must only accept typed `ProjectConfig`
     - Must compose skills by importing ONLY their public `api.py`
     - Must create/resolve run directory via project-owned logic
     - Must write outputs following `docs/runs_and_artifacts.md`

   - Minimal skeleton should:
     - resolve run directory
     - log config
     - call placeholder functions for skill composition

9. **Create outputs helpers (optional stub)**
   - Create:
     - `src/rvinci/projects/{{project_name}}/outputs/writers.py`

   - Purpose:
     - centralize writing `config_resolved.yaml`, `run_manifest.json`, `metrics.json`
     - enforce canonical run structure

10. **Create minimal project README**
    - Create:
      - `src/rvinci/projects/{{project_name}}/README.md`

    Must include:
    - purpose
    - what skills it composes
    - how to run via `uv run python -m rvinci.projects.{{project_name}}.cli`
    - example Hydra overrides
    - where outputs go (`runs/{{project_name}}/...`)

11. **Do NOT add dependencies automatically**
    - If new deps are needed:
      - propose them and ask the user before editing `pyproject.toml`.

12. **Optional: basic sanity checks (safe)**
    - If available:
      - `uv run ruff format .`
      - `uv run ruff check .`
      - `uv run pytest`

## Output
A new project skeleton created under `src/rvinci/projects/{{project_name}}/` with:
- Hydra-enabled `cli.py` (Config Gate enforced)
- `configs/` layout
- Pydantic schema stub
- pipeline entrypoint stub
- README and outputs stubs
