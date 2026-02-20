---
trigger: always_on
---

# rvinci Constitution

These rules apply to all work in this workspace.

1) **Namespace Is Law:** All code lives under `src/rvinci/`. Import only via `rvinci.*` (never relative). The package must work after `pip install -e .`. Never modify `sys.path`.
2) **Dependency Gravity:** core → libs → skills → projects. Forbidden: projects→projects, skills→projects, core→libs/skills/projects.
3) **Skills vs Projects:** 
   - **Skills:** Reusable deep-learning capabilities (no Hydra, Pydantic configs, stable API at `rvinci.skills.<domain>.<skill>.api`).
   - **Projects:** Orchestration/experiments (owns Hydra configs, creates run directories).
4) **Configuration Gate:** Hydra ONLY in `projects`. Convert `DictConfig` immediately to typed Pydantic models in `cli.py` and pass downward.
5) **Runs and Models:** Only projects create run directories under `runs/`. Model weights are artifacts stored locally under `models/` (not committed). Do not hardcode absolute paths.
6) **Canonical Execution:** Run projects via `uv run python -m rvinci.projects.<project_name>.cli`. Check code via `uv run ruff format .`, `uv run ruff check .`, `uv run pytest`.
7) **Agent Workflows:** 
   - If user asks to create a new project, execute the `/scaffold_projects` workflow (use `view_file` on `.agent/workflows/scaffold_projects.md`).
   - If user asks to create a new skill, execute the `/scaffold_skill` workflow (use `view_file` on `.agent/workflows/scaffold_skill.md`).
8) **Deep Knowledge:** If unsure about architecture, runs, dev env, or porting, read `.agent/rules/10-rvinci-deep-knowledge.md` or use the `rvinci-docs` skill.