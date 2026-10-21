# Architecture

The public control layer is separate from the preserved reconstruction runtime.

| Module | Responsibility |
| --- | --- |
| `prepare.py` | Validate assets and manifest; create date-scoped runtime copies; bind immutable inputs |
| `submit.py` | Generate and submit the Slurm/Apptainer script; prevent duplicate submission |
| `credential_relay.py` / `credential_entry.py` | Transfer a hidden credential through a private server RAM socket |
| `runner.py` | Process scenes sequentially, enforce watchdogs, preserve interrupted attempts and verify source hashes |
| `review.py` | Render selected geometry and every available thread fragment in complete stereo views |
| `collect.py` | Export separate controller packages, joint angles, NPZs, OBJ scenes and verified archives |
| `runtime_template/` | Pinned segmentation, reconstruction, tool-fitting, repair and verification compatibility modules |

## Baseline-preserving repair

The heuristic reconstruction is computed first. The agent receives stereo evidence, the baseline and failure diagnostics. Its proposed segmentation or geometry repair is a separate candidate. Verification and visual review decide whether it replaces the selected result. A failed repair or exhausted budget retains the best preserved result.

The native tool stage uses two distal jaw-tip observations for each tool. The VLM identifies them from the full images; it is not given the GPT coordinates from a previous experiment. The original tracker performs Bayesian initialization followed by CMA-ES. The jaw midpoint is derived internally. There is no extra custom tool-pose optimizer after that native fit in this revision.

## Budget and failure boundaries

The current scene cap is 23 physical model attempts: up to 5 perception, 12 scene-agent, 5 tool-agent and 1 needle-size attempt. Charged failures remain counted. Logs distinguish unavailable geometry, provider failures, malformed decisions, exhausted budgets and packaging errors.

The runner does not promise that every snapshot converts successfully. It provides bounded execution and explicit terminal states. A process watchdog terminates the affected process group, retains its artifacts and continues to the next scene. Interrupted charged work is not automatically replayed after restart.

## Compatibility boundary

Historical scripts inside `runtime_template/` are source dependencies, not supported standalone launchers. Some retain original date prefixes or default paths; `prepare` relocates the supported execution paths into the new run. Do not reorganize those internal modules without reconstruction regression tests. Shared scientific assets and weights remain outside the repository.
