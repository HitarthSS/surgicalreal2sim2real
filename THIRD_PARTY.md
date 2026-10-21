# External components

The release is a reproducible research source snapshot, not a replacement license for its dependencies.

- `online_dvrk_tracking_clean`: obtain the original checkout, native mesh assets, FK, TuRBO and Tip2DNet checkpoint from the laboratory's existing distribution. Check its component licenses before redistribution. This release contains adapters, not its large learned weights.
- SAM3: https://github.com/facebookresearch/sam3 ; runtime recipe pins660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7. The separate fine-tuned checkpoint is an external asset.
- Fast-FoundationStereo: source and23-36-37 checkpoint are external assets under their upstream terms.
- PyTorch3D0.7.9, nvdiffrast, FastGeodis, evotorch, gpytorch, Newton1.3.0 and other dependencies retain their upstream licenses. See environment/Dockerfile.cuda128 and validated-freeze.txt.

The newer Surgical_Instrument_Pose_Estimation optimizer is not executed by this release. Do not replace the configured native tracker with that repository: the keypoint semantics differ.

The laboratory's med3d:cu128 base image is not included. The Dockerfile records extensions to that existing base, and the validated Apptainer environment is documented separately. Model downloads or access tokens are never performed implicitly by the full-pipeline launcher.
