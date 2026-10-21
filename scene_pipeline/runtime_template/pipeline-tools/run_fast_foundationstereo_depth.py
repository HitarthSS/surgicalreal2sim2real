#!/usr/bin/env python3
"""Run NVlabs Fast-FoundationStereo and save a depth map for this pipeline.

This is a thin headless wrapper around a checked-out Fast-FoundationStereo repo.
It writes the intrinsic file expected by the NVlabs demo, monkey-patches GUI
display calls, saves the demo's raw disparity before its optional Open3D block,
and converts it to metric depth. The resulting depth map can be consumed by
``stereo_manual_points_arc_pose.py --depth-npy``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np


def load_k(path: Path) -> np.ndarray:
    k = np.loadtxt(path, delimiter=",").astype(np.float64)
    if k.shape != (3, 3):
        raise RuntimeError(f"Expected K shape (3,3), got {k.shape}")
    return k


def write_intrinsics(path: Path, k: np.ndarray, baseline_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    k_values = " ".join(f"{v:.12g}" for v in k.reshape(-1))
    path.write_text(f"{k_values}\n{float(baseline_m):.12g}\n")


def disparity_to_depth(disp: np.ndarray, k: np.ndarray, baseline_m: float) -> np.ndarray:
    disp = np.asarray(disp, dtype=np.float64)
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 1.0e-6)
    depth[valid] = (float(k[0, 0]) * float(baseline_m) / disp[valid]).astype(np.float32)
    return depth


def patch_gui() -> None:
    try:
        import cv2

        cv2.imshow = lambda *args, **kwargs: None
        cv2.waitKey = lambda *args, **kwargs: -1
        cv2.destroyAllWindows = lambda *args, **kwargs: None
    except Exception:
        pass

    try:
        import open3d as o3d

        class _DummyVisualizer:
            def create_window(self, *args, **kwargs):
                return True

            def get_render_option(self):
                class _Opt:
                    point_size = 1.0
                    background_color = np.zeros(3)

                return _Opt()

            def add_geometry(self, *args, **kwargs):
                return True

            def get_view_control(self):
                class _Control:
                    def set_front(self, *args, **kwargs):
                        return None

                    def set_lookat(self, *args, **kwargs):
                        return None

                    def set_up(self, *args, **kwargs):
                        return None

                return _Control()

            def run(self):
                return None

            def destroy_window(self):
                return None

        o3d.visualization.Visualizer = _DummyVisualizer
    except Exception:
        pass


def patch_run_demo(script: Path) -> None:
    """Patch compatibility gaps between current repo code and released weights."""
    text = script.read_text()
    if "model.args['normalize'] = args.get('normalize', True)" not in text:
        needle = "  model.args.max_disp = args.max_disp\n"
        replacement = (
            "  model.args.max_disp = args.max_disp\n"
            "  model.args['normalize'] = args.get('normalize', True)\n"
        )
        if needle not in text:
            raise RuntimeError(f"Could not find model args patch point in {script}")
        text = text.replace(needle, replacement, 1)

    save_disparity = "  np.save(f'{args.out_dir}/disp.npy', disp.astype(np.float32))\n"
    if save_disparity not in text:
        # The upstream demo only writes metric depth inside ``if args.get_pc``.
        # Save raw disparity just before that block so headless runs can disable
        # Open3D while preserving the full-precision model output.
        needle = "  if args.get_pc:\n"
        if needle not in text:
            raise RuntimeError(f"Could not find disparity save patch point in {script}")
        text = text.replace(needle, save_disparity + needle, 1)

    script.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Fast-FoundationStereo checkout directory")
    parser.add_argument("--model", required=True, help="Fast-FoundationStereo pretrained .pth file")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--K", required=True)
    parser.add_argument("--baseline-m", type=float, default=0.006820275085)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--max-disp", type=int, default=416)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zfar", type=float, default=10.0)
    parser.add_argument("--remove-invisible", type=int, default=1)
    parser.add_argument("--denoise-cloud", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    script = repo / "scripts" / "run_demo.py"
    if not script.exists():
        raise FileNotFoundError(f"Fast-FoundationStereo demo script not found: {script}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_file = out_dir.parent / f"{out_dir.name}_intrinsics_for_fast_foundationstereo.txt"
    k = load_k(Path(args.K))
    write_intrinsics(intrinsic_file, k, args.baseline_m)

    patch_run_demo(script)
    patch_gui()
    sys.path.insert(0, str(repo))
    sys.argv = [
        str(script),
        "--model_dir",
        str(Path(args.model).resolve()),
        "--left_file",
        str(Path(args.left).resolve()),
        "--right_file",
        str(Path(args.right).resolve()),
        "--intrinsic_file",
        str(intrinsic_file),
        "--out_dir",
        str(out_dir),
        "--scale",
        str(args.scale),
        "--valid_iters",
        str(args.valid_iters),
        "--max_disp",
        str(args.max_disp),
        "--zfar",
        str(args.zfar),
        "--get_pc",
        "0",
        "--remove_invisible",
        str(args.remove_invisible),
        "--denoise_cloud",
        str(args.denoise_cloud),
    ]
    runpy.run_path(str(script), run_name="__main__")

    depth_path = out_dir / "depth_meter.npy"
    if not depth_path.exists():
        disp_candidates = [
            out_dir / "disp.npy",
            out_dir / "disparity.npy",
            out_dir / "disp_pred.npy",
        ]
        disp_path = next((p for p in disp_candidates if p.exists()), None)
        if disp_path is None:
            produced = sorted(str(p) for p in out_dir.glob("*"))
            raise RuntimeError(
                f"Fast-FoundationStereo wrote neither depth_meter.npy nor disp.npy under {out_dir}. "
                f"Produced files: {produced}"
            )
        disp = np.load(disp_path)
        depth = disparity_to_depth(disp, k, args.baseline_m)
        np.save(depth_path, depth)
        print(f"converted disparity to depth: {disp_path} -> {depth_path}")
    depth = np.load(depth_path)
    print(f"depth: {depth_path}")
    print(f"shape: {depth.shape}")
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size:
        print(f"depth_m min/median/max: {float(finite.min()):.6g} {float(np.median(finite)):.6g} {float(finite.max()):.6g}")


if __name__ == "__main__":
    main()
