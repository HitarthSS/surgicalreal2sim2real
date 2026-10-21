# Contributing

Use the public modules in `scene_pipeline/` for orchestration and the documented CLI for execution. The `runtime_template/` directory is a pinned compatibility snapshot; changes there need reconstruction regression checks, not only import checks.

Before a change:

1. Preserve source images, calibration, human annotations and previous outputs.
2. Use a fresh output directory for a deliberate scientific rerun.
3. Keep credentials, model weights and generated results outside Git.

For a pull request:

- Run the CPU tests and repository-content check.
- Describe the input conditions, behavior change, failure handling and validation.
- Separate control-layer changes from reconstruction/parameter changes.
- Do not add rules keyed to individual scene IDs to improve reported results.
- Preserve heuristic baselines, failed candidates and charged-attempt accounting.
- Report processing completion, geometry consistency and independent accuracy separately.

Do not introduce hidden calibration changes, arbitrary coordinate clipping, uncounted API calls or a new optimizer without documenting the change. GPU tests must respect the target server's scheduler rules. Changes to exports should retain all available fragments, the original observations and the joint-angle/coordinate contracts.
