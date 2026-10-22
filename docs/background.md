# Optional background surfaces

This stage uses stereo depth and foreground masks to create a textured background in the rectified left-camera frame. It runs on CPU after foreground reconstruction and is disabled by default.

Install the extra dependencies when running outside the supplied container:

```bash
python3 -m pip install '.[background]'
```

## Add it to a new pipeline run

Add this block to your asset configuration before `prepare`:

```json
"background": {
  "enabled": true,
  "vlm": true,
  "stride": 8,
  "foreground_margin_px": 3,
  "plane_tolerance_m": 0.002,
  "max_edge_m": 0.01
}
```

The VLM inspects both complete images and selects a surface model and material preset. It can select appearance roughness and metallic values in `[0,1]`; it cannot change cameras, metric scale, geometry checks or invent physical measurements. One physical attempt is allowed within the existing 23-attempt scene cap. If that budget is exhausted or selection fails, stereo reconstruction still runs with a deterministic choice and the failure is recorded. Set `vlm` to `false` to use no API calls.

## Add backgrounds to completed scenes

Enable the same block in the existing run's `config.json`, then run inside the container:

```bash
python3 -m scene_pipeline background /data/youruser/runs/run01
```

With `vlm: true`, supply the API key through stdin; do not put it in the command or save it on the server. Use the same Slurm/Apptainer rules as other stages. This command reads completed foreground results and never calls foreground reconstruction. Existing background attempts are preserved and skipped.

For one scene without API calls:

```bash
python3 -m scene_pipeline background-case /data/youruser/runs/run01/gpt/dayYYYYMMDD/SCENE/case.json /data/youruser/runs/run01/gpt/dayYYYYMMDD/SCENE/background --config examples/background.json
```

Add `--vlm` to request one visual selection with the credential read from stdin. For a saved decision instead, set `decision_file` in the background config to a JSON file containing exactly:

```json
{
  "surface_model": "auto",
  "material_preset": "cloth",
  "roughness": 0.95,
  "metallic": 0,
  "reason": "Visible fabric support surface"
}
```

Surface choices are `auto`, `plane`, `depth_mesh`, and `sam3d`. Material presets are `cloth`, `silicone`, `rigid_plastic`, `metal`, `tissue_like`, and `unknown`. Physics coefficients are assumed presets; this stage does not create a deformable cloth/tissue simulator.

## Geometry and outputs

- Flat surfaces use robust plane fitting and camera-ray intersections.
- Curved surfaces use a smoothed stereo-depth mesh. Triangles crossing foreground masks, missing depth or excessive edge lengths are removed.
- Tools, needle and thread are excluded using existing masks with a configurable margin. Missing masks cause an explicit failure rather than treating the foreground as background.
- Depth agreement and right-camera background support gate acceptance. Stereo colour disagreement is a diagnostic only.
- The texture preserves visible background pixels; foreground areas use nearest-background infill and are marked as inferred geometry where depth was unavailable.

Each `SCENE/background/` contains `background.json`, `background_geometry.npz`, a textured camera-frame OBJ/MTL, `texture.png`, `material.json`, background masks and a full stereo preview. A separate status receipt records failures. Rejected candidates are retained but excluded from combined geometry.

The normal collector includes accepted backgrounds under both heuristic and agent scene folders. It adds a Newton-frame textured OBJ and transformed NPZ; `scene.json` points to them. The combined inspection OBJ also includes a neutral-colour background. Collection still requires a fresh output directory; preserve an old collection before collecting again.

## SAM 3D extension

SAM 3D inference is not bundled or automatically launched. An externally generated mesh can be supplied as `sam3d_candidate`, an NPZ with:

- `vertices`: finite `(N,3)` local mesh coordinates.
- `faces`: integer `(M,3)` triangle indices.
- `anchor_points_local`: at least six local 3D anchors, not all collinear.
- `anchor_pixels_left`: matching `(A,2)` full-frame left-image pixels with valid background stereo depth.

The stage fits positive uniform scale, rotation and translation from those anchors to stereo depth. Alignment and depth/right-view checks must pass before export. A missing or rejected candidate falls back to the stereo surface and keeps the reason in the manifest. This interface does not claim that SAM 3D was executed or that generated hidden surfaces were observed.
