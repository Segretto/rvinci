# Dataset & Benchmark Registry Specification

Just as model weights are treated as curated artifacts, datasets must be tracked immutably to ensure reproducibility—especially when benchmarking different architectures for research.

----------------------------------------------------------------------
1. Canonical Directory Structure
----------------------------------------------------------------------
All reusable datasets must live under `repo_root/data/`. This directory is tracked, but its heavy contents are ignored via `.gitignore`.

data/
  perception/
    segmentation/
      agro_benchmark/
        v1_0/
          images/
          annotations.json
          manifest.json
          splits/

----------------------------------------------------------------------
2. manifest.json Specification
----------------------------------------------------------------------
Every dataset version must contain a `manifest.json` ensuring traceability:

{
  "schema_version": 1,
  "dataset_name": "agro_benchmark",
  "domain": "perception.segmentation",
  "version": "v1.0",
  "class_map": {
    "0": "background",
    "1": "target_object"
  },
  "source_hash": "a1b2c3d4",
  "splits": ["train", "val", "test"]
}

----------------------------------------------------------------------
3. Data Referencing Rule
----------------------------------------------------------------------
Projects must NOT hardcode absolute paths to data. Data must be referenced via a structured URI in the Hydra config:

    dataset_ref: "data://perception/segmentation/agro_benchmark/v1_0"

This guarantees that a benchmark experiment can be perfectly reconstructed without local filesystem assumptions.