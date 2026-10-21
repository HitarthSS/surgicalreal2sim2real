"""Show raw fragments/crop endpoints, approximate occluders and shared QP checks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from reconstruction_tools import DEFAULT_TOOLS, LIMITS, sha256
from reconstruction_verification import load_base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.checkpoint / 'summary.json').read_text())
    base = load_base(DEFAULT_TOOLS)
    canvas = Image.new('RGB', (1280, 60 + 535*len(summary['rows'])), '#141a21')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), 'DEVELOPMENT DIAGNOSTIC: predicted fragments and approximate instrument masks | unchanged production QP estimates', fill='white')
    draw.text((8, 29), 'Cyan = predicted thread. Blue = manual instrument mask. Yellow endpoints = within 8 px of crop. Orange QP = unsupported.', fill='white')
    records = []
    for i, row in enumerate(summary['rows']):
        case = row['case_id']; folder = a.checkpoint / case
        image = cv2.imread(str(a.data_root / 'recordings' / case / 'left_rect.png'))
        mask = base.load_mask(a.data_root / 'segmentation_pilot' / case / 'left_thread_mask.png', image.shape[:2])
        occ = base.load_mask(folder / 'left_occluder.png', image.shape[:2])
        skel, graph, _, components, discarded = base.skeleton_graph(mask, 0, LIMITS['segment_min_skeleton_px'])
        endpoints = []
        for rc in sorted(node for node in graph if graph.degree[node] == 1):
            distance = min(rc[0], rc[1], image.shape[0]-1-rc[0], image.shape[1]-1-rc[1])
            endpoints.append(dict(rc=list(map(int, rc)), distance_to_crop_px=int(distance), near_crop=bool(distance <= 8)))
        left = image.copy()
        left[occ] = (.6*left[occ]+.4*np.array([255, 100, 0])).astype('uint8')
        left[skel] = [255, 255, 0]
        for j, endpoint in enumerate(endpoints):
            rc = endpoint['rc']; xy = (rc[1], rc[0])
            color = (0, 255, 255) if endpoint['near_crop'] else (0, 80, 255)
            cv2.circle(left, xy, 5, color, 2)
            position = (max(4, min(xy[0]+6, image.shape[1]-25)), max(15, min(xy[1]+15, image.shape[0]-8)))
            cv2.putText(left, str(j), position, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        qp = folder / 'fixed/call_001/backend'
        right = cv2.imread(str(qp / 'shared_left_overlay.png'))
        verification = row['results']['fixed']['result']['verification']
        metrics = verification['metrics']
        near = sum(endpoint['near_crop'] for endpoint in endpoints)
        record = dict(case_id=case, component_count=components, endpoints=endpoints,
                      crop_proximity_is_diagnostic_not_connection_evidence=True,
                      qp_failures=verification['failures'], qp_metrics=metrics,
                      adaptive_rejections=row['candidate_rejection_counts'],
                      adaptive_joins=row['results']['heuristic']['result']['backend_report'].get('joins', []))
        records.append(record)
        y = 60+i*535
        captions = [f'{case[-6:]} input: {components} retained fragments; {near}/{len(endpoints)} endpoints near crop',
                    f'{case[-6:]} QP rejected: coverage {metrics["visible_skeleton_coverage"]:.1%}; {metrics["qp_unsupported_sample_count"]}/200 unsupported']
        for col, (array, caption) in enumerate(zip((left, right), captions)):
            draw.text((col*640+8, y+3), caption, fill='white')
            canvas.paste(Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)), (col*640, y+26))
        draw.text((8, y+511), 'Adaptive joins: 0 | rejected candidate counts: '+json.dumps(row['candidate_rejection_counts']), fill='white')
    canvas.save(a.checkpoint / 'review.jpg', quality=94)
    (a.checkpoint / 'endpoint_diagnostics.json').write_text(json.dumps(dict(rows=records,
        source_sha256=sha256(__file__), near_crop_distance_px=8,
        independent_accuracy_measured=False, training_eligible=False), indent=2)+'\n')
    print(json.dumps([dict(case_id=r['case_id'], endpoints=len(r['endpoints']),
                           near_crop=sum(e['near_crop'] for e in r['endpoints'])) for r in records]))


if __name__ == '__main__':
    main()
