----------------------------------------------------------------------
 Model Registry (Local, Not Committed)
----------------------------------------------------------------------

rvinci maintains a canonical local model registry for:

- Pre-trained models
- Fine-tuned models
- Exported ONNX / deployment artifacts
- Curated checkpoints intended for reuse

Model weights are NOT committed to git.

This section defines the standard layout and usage contract.


----------------------------------------------------------------------
1 Location
----------------------------------------------------------------------

All reusable models must live under:

    repo_root/models/

This directory is tracked, but its contents are ignored.

Add to .gitignore:

    models/**/*
    !models/README.md
    !models/.gitkeep

The repository should include:

    models/
      README.md
      .gitkeep

Weights and large artifacts must never be committed.


----------------------------------------------------------------------
2 Canonical Directory Structure
----------------------------------------------------------------------

Models must be organized by skill domain and version:

    models/
      perception/
        segmentation/
          dinov3_multitask/
            pretrained/
              dinov3_base/
                model.pth
                manifest.json
            trained/
              agro_data_v1/
                2026-02-17_1a2b3c4d/
                  model.pt
                  model.onnx
                  manifest.json
                  metrics.json
                  config_resolved.yaml

      navigation/
        mapping/
          stereo_depth/
            pretrained/
              raft_stereo/
                model.pth
                manifest.json


Directory semantics:

- pretrained/  → externally sourced weights
- trained/     → models produced by rvinci projects
- <dataset_or_variant>/ → semantic grouping
- <timestamp>_<git_hash>/ → reproducible version identifier


----------------------------------------------------------------------
3 Required Files Per Model
----------------------------------------------------------------------

Every stored model must include:

1) Weight file
   - .pt
   - .pth
   - .onnx
   - (or deployment-specific format)

2) manifest.json

Recommended additions for trained models:

- config_resolved.yaml
- metrics.json


----------------------------------------------------------------------
4 manifest.json Specification
----------------------------------------------------------------------

Each model directory must contain a manifest.json file with at least:

{
  "schema_version": 1,
  "model_name": "dinov3_multitask",
  "skill": "perception.segmentation",
  "format": "onnx",
  "git_hash": "1a2b3c4d",
  "input_spec": {
    "resolution": [1024, 1024],
    "normalization": "imagenet"
  },
  "class_map": ["soybean", "cotton"],
  "export": {
    "opset": 17
  }
}

The manifest ensures:

- Traceability
- Reproducibility
- Compatibility validation
- Deployment sanity checks


----------------------------------------------------------------------
5 Model Referencing Rule
----------------------------------------------------------------------

Projects must NOT hardcode filesystem paths like:

    /home/user/models/my_model.pth

Instead, projects must reference models via a structured model reference:

Example (in config):

    model_ref:
      uri: "local://perception/segmentation/dinov3_multitask/trained/agro_data_v1/2026-02-17_1a2b3c4d"
      format: "onnx"

The URI must be resolved through a registry helper in core.

Skills and projects must resolve model paths programmatically via:

    registry.resolve(model_ref)

This prevents:
- Hardcoded absolute paths
- Environment-dependent failures
- Silent mismatches


----------------------------------------------------------------------
6 Relationship with Runs
----------------------------------------------------------------------

runs/ contains ephemeral experiment outputs.

models/ contains curated, reusable artifacts.

Workflow recommendation:

1. Training project writes checkpoints to:

       runs/<project>/<timestamp>/artifacts/checkpoints/

2. When validated, the best model is promoted into:

       models/<skill>/.../trained/<dataset>/<timestamp>_<git_hash>/

Promotion may be done via a script:

       scripts/promote_model.py

Skills must not write directly into models/.
Only projects or explicit promotion scripts may do so.


----------------------------------------------------------------------
7 Future Extensions (Optional)
----------------------------------------------------------------------

The registry design allows future extensions such as:

- hf:// model URIs
- s3:// model URIs
- remote artifact stores
- checksum validation
- automatic compatibility checks

The local layout must remain stable even if remote backends are added later.


----------------------------------------------------------------------
11.8 Design Principles
----------------------------------------------------------------------

- Models are artifacts, not code.
- Code never depends on absolute paths.
- All reusable weights must have a manifest.
- models/ is curated.
- runs/ is experimental.

This separation preserves clarity, reproducibility, and long-term maintainability.
