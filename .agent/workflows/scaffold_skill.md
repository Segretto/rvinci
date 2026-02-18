---
description: Scaffold a new rvinci skill with the canonical structure (no Hydra in skills), including api.py and schema stubs.
---

# Workflow: scaffold-skill

**User Input:**  
"Scaffold a new skill named {{skill_name}} in domain {{domain}}"

## Steps

1. **Validate inputs**
   - Ensure `{{domain}}` is one of: `perception`, `navigation`, `manipulation`, `locomotion` (or an existing folder under `src/rvinci/skills/`).
   - Ensure `{{skill_name}}` is `snake_case` and filesystem-safe.

2. **Read canonical structure**
   - Read `docs/architecture.md` and follow the "How to Add a New Skill" structure exactly.

3. **Create directories**
   - Create:
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/`
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/schemas/`

4. **Create required package files**
   - Create empty:
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/__init__.py`
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/schemas/__init__.py`

5. **Create `api.py` (public interface)**
   - Create:
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/api.py`
   - Requirements:
     - Provide a minimal, stable public interface (functions/classes) that projects and other skills import.
     - Do **not** import Hydra or OmegaConf.
     - Accept typed configs (Pydantic models) or explicit arguments.
     - Use clear docstrings and placeholder implementations (`NotImplementedError`) where appropriate.

6. **Create schema stub(s)**
   - Create:
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/schemas/config.py`
   - Requirements:
     - Define a minimal Pydantic config model (e.g., `SkillConfig`) with `extra="forbid"` if consistent with repo style.
     - Keep it small: only fields needed to define the interface contract.

7. **Create a minimal skill README**
   - Create:
     - `src/rvinci/skills/{{domain}}/{{skill_name}}/README.md`
   - Must include:
     - Purpose (1–2 sentences)
     - Public API usage example (`from rvinci.skills... import ...`)
     - Expected inputs/outputs (conceptual, not implementation details)
     - Note: "This skill is Hydra-agnostic; orchestration belongs in projects."

8. **Do NOT add dependencies automatically**
   - If new dependencies are required:
     - Propose them and ask the user before editing `pyproject.toml`.

9. **Optional: basic sanity checks (safe)**
   - If available in the workspace:
     - `uv run ruff format .`
     - `uv run ruff check .`
     - `uv run pytest`

## Output
- A new skill skeleton created under `src/rvinci/skills/{{domain}}/{{skill_name}}/`
- Includes: `__init__.py`, `api.py`, `schemas/config.py`, and `README.md`
