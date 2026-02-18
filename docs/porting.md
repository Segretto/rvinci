# docs/porting_guide.md
# Porting Working Scripts into rvinci (Agent Instructions)

This document instructs the agent how to port unstructured but working research scripts
(e.g., custom DINO architectures) into the rvinci repository while respecting rvinci rules.

Goal:
- Preserve correctness
- Remove path hacks and ad-hoc assumptions
- Avoid redundancy
- Place code in the correct layer (skill vs project)
- Add minimal, high-value documentation and tests


----------------------------------------------------------------------
0) Mandatory Inputs (from the user)
----------------------------------------------------------------------

Before porting, ensure the user provides:

- Path(s) to the script folder(s) to port
- The primary entrypoint script(s) (what is executed to run it)
- A short description of what the scripts do and expected outputs
- Any required checkpoints / weights location (do not commit weights)
- Example command(s) currently used to run the scripts

If any of these are missing, infer from repository structure and script content,
but do not invent hidden dependencies or magical files.


----------------------------------------------------------------------
1) Read the rvinci Constitution First
----------------------------------------------------------------------

The agent must follow:

- .agent/rules/00-rvinci-constitution.md (Always On)
- If needed, activate: .agent/rules/10-rvinci-deep-knowledge.md
- Consult docs/architecture.md for the canonical structure
- Consult docs/runs_and_artifacts.md for outputs/run directories
- Consult docs/model_registry.md for weights handling
- Consult docs/testing.md for tests and fixtures


----------------------------------------------------------------------
2) Porting Strategy Overview
----------------------------------------------------------------------

Porting must be done as a controlled refactor, not a direct copy.

High-level phases:
1) Inventory and classify the existing scripts
2) Detect existing functionality in rvinci (avoid redundancy)
3) Choose correct destination: skill vs project
4) Refactor into rvinci structure with stable APIs and typed configs
5) Replace ad-hoc IO with canonical artifacts/run structure
6) Add minimal docs + tests + fixtures
7) Verify behavior matches original scripts


----------------------------------------------------------------------
3) Inventory: Understand What the Scripts Do
----------------------------------------------------------------------

The agent must create an inventory including:

- Entry points: what is executed
- Major modules/classes/functions
- Training/inference/eval/export responsibilities
- Inputs: dataset formats, config files, CLI args
- Outputs: prediction formats, metrics, logs, checkpoints
- Dependencies: torch, opencv, transformers, custom libs, etc.

Produce a short map like:

- train.py -> training loop, writes checkpoints
- infer.py -> inference, writes masks/depth npz
- model.py -> DINO backbone + heads
- datasets.py -> COCO loader

Do not start moving code before this inventory is clear.


----------------------------------------------------------------------
4) Redundancy Check (Avoid Re-Implementing What Exists)
----------------------------------------------------------------------

Before porting any component, search in rvinci for equivalents.

Rules:
- If logic already exists in rvinci core/libs/skills, reuse it.
- If the script duplicates existing functionality, do not copy it.
- If the script improves or generalizes an existing module, refactor upstream
  (prefer moving common pieces to libs or skills).

Common redundancy hotspots:
- COCO IO / parsing
- mask conversion / drawing / visualization
- metrics computation / plotting
- run directory handling
- config parsing


----------------------------------------------------------------------
5) Decide: Skill vs Project
----------------------------------------------------------------------

Default decision rule:

A) Add as a SKILL if:
- Reusable DL capability (model, encoder-head architecture, inference service)
- Can be called by multiple projects
- Has a clean API boundary and stable inputs/outputs
- Does not require Hydra orchestration
- Encapsulates a self-contained deep-learning capability 
  with clear inputs and outputs (e.g., segmentation, depth, detection),
  independent of experiment orchestration.

B) Add as a PROJECT if:
- It is an experiment pipeline, benchmark, or application
- It orchestrates multiple skills
- It owns Hydra configs and run outputs
- It produces canonical runs/ artifacts

If unclear:
- Put the architecture/model code into a skill
- Put training/evaluation scripts into a project that uses that skill


----------------------------------------------------------------------
6) Documentation Requirements (Minimal but Sufficient)
----------------------------------------------------------------------

For each ported module:

A) Skill README.md must include:
- what it does
- public API usage example
- expected inputs/outputs
- dependency extras required (if applicable)

B) Project README.md must include:
- purpose
- how to run via uv run + -m
- example Hydra overrides
- where outputs go under runs/


----------------------------------------------------------------------
7) Verification Checklist (Behavioral Parity)
----------------------------------------------------------------------

After porting, verify:

- Ported code reproduces the original script’s outputs on a small fixture dataset
- CLI works via module execution (-m)
- No sys.path modifications exist in src/ or tests/
- Artifacts follow canonical structure
- No weights were committed
- lint + tests run successfully:

    uv run ruff format .
    uv run ruff check .
    uv run pytest

----------------------------------------------------------------------
End State Definition (What Success Looks Like)
----------------------------------------------------------------------

Porting is considered complete when:

- The capability is placed in the correct layer (skill/project)
- Code is importable via rvinci.* without hacks
- A project (if applicable) runs via uv run python -m ...
- Outputs follow canonical runs/ artifacts layout
- Minimal docs and tests exist
- Redundant code has been removed or upstreamed to libs/skills
