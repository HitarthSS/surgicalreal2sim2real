"""Prepare stereo evidence, then execute a version-bound attachment decision."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from needle_thread_connection_tool import (attachment_state, attachment_messages,
    prepare_connected_geometry, validate_attachment_decision)
from newton_scene_adapter import project, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--baseline-m', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--decision', type=Path)
    p.add_argument('--controller', choices=('agent', 'heuristic'), default='agent')
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as a:
        g = {k:a[k].copy() for k in a.files}
    state = attachment_state(g, args.bundle, dict(left=args.left, right=args.right), args.baseline_m)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out/'state.json'
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError('Prepared inputs changed; create a new experiment')
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    images = {}
    for side, path in [('left', args.left), ('right', args.right)]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode stereo image')
        for label, points, color in [('t', g['points_camera_m'], (0,255,0)),
                                     ('n', g['needle_points_camera_m'], (255,255,0))]:
            xy = project(points, g['K'], args.baseline_m if side == 'right' else 0.)
            for i in (0,len(points)-1):
                x,y = np.rint(xy[i]).astype(int)
                cv2.circle(image,(int(x),int(y)),7,color,2)
                # Offset labels in opposite directions to distinguish close endpoints.
                dy = -16 if label == 'n' else 24
                cv2.putText(image, f'{label}{i}', (int(x)+10,int(y)+dy), cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),4)
                cv2.putText(image, f'{label}{i}', (int(x)+10,int(y)+dy), cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        images[side] = (args.out/f'{side}_endpoints.png').resolve()
        cv2.imwrite(str(images[side]),image)
    (args.out/'messages.json').write_text(json.dumps(attachment_messages(state,images),indent=2)+'\n')
    if args.decision is None:
        print(json.dumps(dict(status='evidence_ready', scene_version=state['scene_version'])))
        return
    decision,candidate = validate_attachment_decision(args.decision.read_text(),state)
    report = dict(controller=args.controller, decision=decision, source_scene_sha256=sha256(args.bundle),
                  source_images=state['stereo_images'], source_geometry_preserved=True,
                  hypothesis_not_ground_truth=True, independent_accuracy_measured=False,
                  physics_validated=False, training_eligible=False)
    if candidate is None:
        report.update(status='deferred', attachment=None)
    else:
        connected, attachment = prepare_connected_geometry(g,candidate,args.baseline_m)
        np.savez_compressed(args.out/'connected_scene_bundle.npz',**connected)
        report.update(status='connection_constructed_not_physics_validated',attachment=attachment,
                      scene_bundle_sha256=sha256(args.out/'connected_scene_bundle.npz'))
    (args.out/'connection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
