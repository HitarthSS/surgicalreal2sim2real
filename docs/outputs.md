# Output contract

Heuristic and agent reconstructions are collected into separate folders. Availability of a file does not imply that its geometry passed every check.

| Artifact | Contents |
| --- | --- |
| `scene.json` | Component availability, selected geometry, coordinate conventions and validation flags |
| `thread_fragments.npz` | Available fragments, including disconnected geometry |
| `thread_curve.npz` | A single thread curve when the export's checks permit it |
| `needle_geometry.npz` | Selected rigid needle hypothesis and inferred portions |
| `tools/tool_A/` and `tools/tool_B/` | Fitted visual tool geometry, pose and distal joint parameters |
| `pose_and_joint_angles.npz` | Distal joint angles in radians/degrees and native-frame transforms |
| `joint_angles.json` | Joint names, estimated values and separation from recorded robot states |
| `observations/` | Original stereo images, camera metadata, calibration and recorded state when available |
| `geometry/scene_newton_m.obj` | Inspection mesh including the available thread, needle and tools |
| `review/overlay.jpg` | Complete left/right fields of view with selected geometry |
| `package_verified.json` | Exported-file hashes and source-geometry preservation checks |

The four fitted native distal joints are `Joint5`, `Joint6`, `Joint7` and `Joint8`. They are not a calibrated full-arm estimate. `tool_A` and `tool_B` describe visual instances; no PSM assignment is implied.

The attachment bundle, when available, records the thread/needle anchor contract separately. Other fragments remain disconnected unless an actual connection was selected. Inferred connectors and geometry are marked as hypotheses.

## Read status correctly

- `processing_completed` means a scene pipeline returned, including results with component failures.
- `queue_complete.json` means all requested scenes reached a terminal processing state.
- Numerical checks assess the declared geometry constraints.
- Collection `complete` means the expected controller entries were packaged, not that every reconstruction is scientifically successful.
- Independent accuracy and Newton dynamics require separate evaluation.

`packaging_failures.json` records individual export errors. Original attempts and partial outputs remain available in the run directory. Re-collecting into an existing collection directory is rejected rather than replacing it.
