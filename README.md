# Surgical Scene Pipeline

Reconstruct thread, needle and tool geometry from stereo images using SAM3, Fast-FoundationStereo, native tool fitting and GPT-6.1. Exports separate heuristic and agent reconstructions with NPZs, OBJ meshes, joint angles and stereo overlays.

## Requirements

- Python 3.10+
- Linux server with Slurm and Apptainer
- Compatible GPU container; CUDA 12.8+ for Blackwell GPUs
- SAM3 checkpoint, Fast-FoundationStereo source/checkpoint, and an `online_dvrk_tracking_clean` checkout with meshes and Tip2DNet weights
- OpenAI API key

Model weights, datasets and container images are supplied separately. See [deployment](docs/deployment.md) for runtime setup. `environment/Dockerfile.cuda128` extends the external `med3d:cu128` base; `environment/validated-freeze.txt` records the installed dependencies.

## Setup

```bash
git clone https://github.com/HitarthSS/surgicalreal2sim2real.git
cd surgicalreal2sim2real
cp examples/config.json config.local.json
cp examples/manifest.json manifest.local.json
```

Edit `config.local.json` with your checkpoint, tracker, stereo-model and container paths.

Edit `manifest.local.json` with:

- Rectified left/right PNG paths and SHA256 hashes
- Camera intrinsics CSV and measured stereo baseline in meters
- Original calibration JSON and optional recorded robot-state JSON
- Existing depth map, if available; otherwise the pipeline generates depth

Replace all example paths, hashes and calibration values. Add directories outside `/data` to `bind_paths`. See [configuration](docs/configuration.md) for the complete schema.

## Run

Run from the repository root on the server:

```bash
python3 -m scene_pipeline doctor config.local.json
python3 -m scene_pipeline prepare manifest.local.json config.local.json /data/youruser/runs/run01
python3 -m scene_pipeline submit /data/youruser/runs/run01
python3 -m scene_pipeline credential /data/youruser/runs/run01
```

The final command prompts for the API key with hidden input. The submitted job runs independently of your SSH connection.

Default allocation: one GPU shard, eight CPUs, 40 GiB RAM and 12 hours. Adjust `scene_pipeline/submit.py` before submission if your cluster requires different resources.

Check the job with:

```bash
squeue --me
```

Use a new output directory for each run. Existing run directories are preserved.

## Outputs

```text
run01/
├── progress.json
├── queue_complete.json
├── gpt/dayYYYYMMDD/SCENE/       Logs, baselines and repair candidates
├── review/gallery.html         Heuristic and agent stereo overlays
└── reconstructions/GPT61_Reconstructions/
    ├── heuristic/SCENE/
    └── agent/SCENE/
```

Scene folders contain geometry NPZs, inspection OBJ meshes, observations, calibration and fitted joint angles. See [output files](docs/outputs.md).

To regenerate overlays or collect existing results without running inference:

```bash
python3 -m scene_pipeline review /data/youruser/runs/run01
python3 -m scene_pipeline collect /data/youruser/runs/run01
```

Collection requires a new `reconstructions/` directory. See [deployment and recovery](docs/deployment.md) for handling interrupted jobs.

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_repository.py
```

## More information

- [Architecture](docs/architecture.md)
- [Optional background surfaces and materials](docs/background.md)
- [Contributing](CONTRIBUTING.md)
- [Third-party components](THIRD_PARTY.md)
- [Licensing notice](LICENSE_NOTICE.md)
