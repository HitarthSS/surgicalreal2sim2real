"""Inspect calibrated projection consistency and the separate synthetic fixture."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import numpy as np


def main():
    project = Path(__file__).resolve().parents[3]
    root = project / 'research/ismr_2027/newton_adapter_2026-10-08'
    a = np.load(root / 'real_frame_787/scene_bundle.npz')
    sys.path.insert(0, str(project / 'research/ismr_2027/code'))
    from newton_scene_adapter import project as project_points
    frames = project / 'research/ismr_2027/latest5_2026-10-06/hitarth_frame_000787/frames/episode_0000/colors'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    edge_ids = np.unique(np.sort(a['tool_faces'][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    colors = np.where(a['inferred'][:-1] | a['inferred'][1:], '#ffac35', '#30ee75')
    for ax, side, baseline in zip(axes, ('left', 'right'), (0, .006820275085)):
        im = np.array(Image.open(frames / f'{side}_image_000000.jpg'))
        ax.imshow(im)
        xy = project_points(a['tool_vertices_camera_m'], a['K'], baseline)
        ax.add_collection(LineCollection(xy[edge_ids], colors='#34bbff', linewidths=.25, alpha=.5))
        thread_xy = a[f'{side}_pixel_xy']
        ax.add_collection(LineCollection(np.stack((thread_xy[:-1], thread_xy[1:]), axis=1), colors=colors, linewidths=2))
        ax.set(xlim=(0, im.shape[1]), ylim=(im.shape[0], 0), title=f'{side.capitalize()} calibrated projection')
        ax.axis('off')
    fig.suptitle('Frame 787: newest static tool + estimated thread', fontsize=14)
    fig.text(.5, .04, 'Blue: fitted tool   Green: visible curve support   Orange: inferred bridge (87/200 samples)\nProjection consistency only; no independent accuracy or manipulation claim.', ha='center', fontsize=10)
    fig.subplots_adjust(top=.87, bottom=.15, wspace=.03)
    fig.savefig(root / 'projection_review.png', dpi=170)
    plt.close(fig)
    probe = np.load(root / 'real_frame_787/probe_v2/states.npz')
    from newton_scene_adapter import transform, CAMERA_TO_NEWTON
    final_camera = transform(probe['states'][-1], np.linalg.inv(CAMERA_TO_NEWTON))
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(10, 5))
    im = np.array(Image.open(frames / 'left_image_000000.jpg'))
    ax.imshow(im)
    for label, p, color in [('Initial estimate', a['points_camera_m'], '#37e77a'),
                             ('After 20 steps', final_camera, '#ff5252')]:
        xy = project_points(p, a['K']); ax.plot(*xy.T, color=color, linewidth=2, label=label)
    ax.set(xlim=(0, im.shape[1]), ylim=(im.shape[0], 0), title='No tool motion: before/after projection')
    ax.legend(fontsize=9); ax.axis('off')
    drift = np.linalg.norm(probe['body_q'][:, :, :3] - probe['body_q'][0, :, :3], axis=2).max(axis=1) * 1000
    curve.plot(probe['times'] * 1000, drift, color='#d33131')
    curve.set(xlabel='Simulated time (ms)', ylabel='Maximum body displacement (mm)',
              title='Static scene drift: data generation blocked')
    curve.grid(alpha=.2)
    fig.text(.5, .02, 'Zero gravity, stationary tool, uncalibrated prototype physics. No successful demonstration claimed.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(root / 'static_drift_review.png', dpi=160); plt.close(fig)
    s = np.load(root / 'synthetic_needle_fixture/scene_bundle.npz')
    fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(projection='3d')
    for name, key, color in [('Synthetic thread', 'points_newton_m', '#39ac60'),
                             ('Fixed synthetic needle', 'needle_points_newton_m', '#d87c17')]:
        p = s[key] * 1000
        ax.plot(*p.T, color=color, linewidth=3, label=name)
    ax.set(xlabel='Newton x (mm)', ylabel='Newton y (mm)', zlabel='Newton z (mm)',
           title='Separate synthetic rigid-needle import fixture')
    ax.legend(); ax.set_box_aspect((3, 1, 1))
    ax.set_yticks([145, 150, 155])
    fig.text(.5, .02, 'Known 10 mm curve radius; not a reconstructed real needle or measured ground truth.', ha='center', fontsize=9)
    fig.savefig(root / 'synthetic_fixture_review.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__': main()
