# Configuration and input schema

Use `examples/config.json` and `examples/manifest.json` as templates. Keep your edited copies named `*.local.json` so they remain ignored by Git.

## External assets

| Configuration field | Required value |
| --- | --- |
| `sam3_checkpoint` | Fine-tuned SAM3 inference checkpoint path |
| `sam3_sha256` | SHA256 of that checkpoint |
| `native_tracker` | Compatible `online_dvrk_tracking_clean` checkout, native meshes, TuRBO and Tip2DNet weights |
| `legacy_assets` | Existing compatibility-assets directory; explicit scene inputs are taken from the manifest |
| `ffs_root` | Fast-FoundationStereo checkout |
| `ffs_checkpoint` | The compatible `23-36-37` stereo checkpoint |
| `container_image` | Compatible Apptainer SIF image |
| `bind_paths` | Additional directories that the container must be able to read |
| `path_overrides` | Explicit additional relocations for a different legacy source layout |

`doctor` checks existence, SAM3 checkpoint identity and the pinned native files in `scene_pipeline/resources/native_reference_hashes.json`. Set `verify_native_reference` to `false` only when deliberately using a different native revision; that configuration is outside the pinned reference. Do not replace the native tracker with the similarly named `Surgical_Instrument_Pose_Estimation` repository: its landmark semantics differ.

## Scenes

Each scene needs:

- `case_id`: unique `snapshot_YYYYMMDD_HHMMSS` identifier.
- `difficulty`: `easy` or `hard`, supplied by your dataset protocol.
- `partition`: your dataset partition label.
- `image_sha256.left` and `.right`: hashes of the complete rectified PNG images.
- A standalone `case` object with `inputs.left`, `inputs.right`, `inputs.K` and `baseline_m`.

`K` is a comma-separated 3x3 camera matrix. Both images must share the declared rectified calibration. Supply the actual baseline in meters, not the illustrative example value.

Optional fields:

- `inputs.depth_npy`: an existing depth map in meters. If unavailable, depth is generated using the configured stereo model.
- `calibration_path`: original camera-calibration JSON, even when it is stored outside the image directory.
- `robot_state_path`: original recorded robot-state JSON.
- `source_case`: a path to an existing case JSON instead of an inline `case` object. This is a convenience for an existing dataset, not a required laboratory folder structure.

Explicit metadata paths are checked and hashed during preparation. A missing explicit path raises an error rather than silently omitting the observation. Original calibration metadata is required for the portable collection export.

On Linux, compute input hashes with:

```bash
sha256sum /data/youruser/data/scene01/left_rect.png
sha256sum /data/youruser/data/scene01/right_rect.png
sha256sum /data/youruser/models/sam3_epoch2.pt
```

Preparation creates a new output directory, verifies image identity, copies the small calibration matrix and binds source metadata by hash. It does not run models or modify the source images. Existing output directories are rejected to protect previous work.
