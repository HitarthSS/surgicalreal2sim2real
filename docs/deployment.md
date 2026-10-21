# Deployment and recovery

## GPU runtime

The validated target is a Linux Slurm server using Apptainer. All GPU work must run through Slurm, inside the container. Docker may be used to build an image on the CPU and export it as a SIF; it is not the GPU execution path.

`environment/Dockerfile.cuda128` extends the laboratory's external `med3d:cu128` base. That base is not included or implicitly downloaded. `environment/validated-freeze.txt` records the tested environment and includes build-system paths; do not use it directly as a pip lockfile on another machine.

Supply a compatible SIF and external model assets in your local configuration. The native tracker checkout must include its meshes, FK, optimizer sources and `ContourTipNet/models/cnn_model.pth`. SAM3 weights and any fine-tuned checkpoint must be obtained separately. Fast-FoundationStereo is required when depth is not supplied.

## Submission

Use the public CLI from the source checkout on the server. `submit` records the job ID in `submission.json` before another invocation is allowed. Review resource defaults in `scene_pipeline/submit.py` before submitting on a different cluster.

```bash
squeue --me
sacct -X -j JOB_ID --format=JobID,State,Elapsed
```

A queued job is not a running job. Account GPU quotas, resource availability and queue priority can delay its start. The RAM credential relay waits independently of SSH, but a server reboot destroys its in-memory credential. The relay stores only nonsensitive status metadata on disk.

Do not paste API credentials into configuration files or command arguments. Run the `credential` command in a terminal; its input is hidden. An existing relay marker prevents an accidental second relay. If a relay fails, inspect its recorded PID/status before replacing it.

## Failure and resumption

The runner preserves scene start markers, terminal receipts, raw failed attempts and selected baselines. It continues to the next scene after a failure. Watchdogs bound preprocessing and scene subprocess groups.

If a job is interrupted, inspect the stage logs and existing results first. Restarting the same prepared runner preserves completed scenes and marks previously started, nonterminal scenes as interrupted; it does not replay their charged scientific work. A deliberate scientific rerun requires a fresh output directory.

A failed submission is also recorded. Inspect whether Slurm accepted the job before clearing any submission marker; deleting a marker blindly can create duplicate jobs.

## Export-only commands

These commands perform no new model inference:

```bash
python3 -m scene_pipeline review /data/youruser/runs/run01
python3 -m scene_pipeline collect /data/youruser/runs/run01
```

`collect` rejects an existing `reconstructions/` directory to protect previously exported work. Inspect partial packages and their failure receipts before attempting a separate collection revision. Never delete or replace source annotations to repair an export.
