"""Run unchanged production QP while observing native ordering decisions.

Tracing does not change arguments, locals, returns, assembly or optimization.
An actual native join decision is not independent evidence that a join is valid.
"""
import argparse
import ast
import json
from pathlib import Path
import runpy
import sys

import numpy as np


def decision_line(source):
    tree = ast.parse(Path(source).read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'keypt_ordering']
    lines = []
    for function in functions:
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                names = [element.id if isinstance(element, ast.Name) else None for element in node.targets[0].elts]
                if names == ['s_idx', 'action', 'direction', 'join_order_pt', 'join_seg_pt']:
                    lines.append(node.lineno)
    if len(lines) != 1:
        raise ValueError('Unrecognized native ordering source; cannot trace decisions safely')
    return lines[0]


class OrderingObserver:
    def __init__(self, source):
        self.source = str(Path(source).resolve())
        self.line = decision_line(source)
        self.joins, self.returns = [], []
        self.valid_filenames = {self.source}

    def trace(self, frame, event, arg):
        if frame.f_code.co_name != 'keypt_ordering':
            return None
        filename = frame.f_code.co_filename
        if filename not in self.valid_filenames:
            if str(Path(filename).resolve()) != self.source:
                return None
            self.valid_filenames.add(filename)
        if event == 'line' and frame.f_lineno == self.line:
            local = frame.f_locals
            index, action, direction, tail, head = local['best_match']
            segment = local['segments'][index]
            self.joins.append(dict(step=int(local['join_step']), action=str(action), direction=str(direction),
                candidate_segment_index=int(index), candidate_keypoint_count=len(segment),
                from_rc=np.asarray(tail, dtype=float).tolist(), to_rc=np.asarray(head, dtype=float).tolist(),
                native_cost=float(local['best_cost']), raw_gap_px=float(np.linalg.norm(np.asarray(tail)-head)),
                cap_px=float(local['self'].COLD_ASSEMBLY_MAX_JOIN_PX)))
        if event == 'return':
            local = frame.f_locals
            remaining = local.get('segments', [])
            entered = 'join_step' in local
            self.returns.append(dict(assembly_loop_entered=entered,
                remaining_segment_count=len(remaining) if entered else None,
                remaining_keypoint_count=sum(len(segment) for segment in remaining) if entered else None,
                returned_order_count=len(arg[3]) if isinstance(arg, tuple) and len(arg) == 4 and arg[3] is not None else None))
        return self.trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-tools', type=Path, required=True)
    parser.add_argument('--original-src', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a, native_args = parser.parse_known_args()
    source = a.original_src / 'thread_reconstruction/keypt_ordering.py'
    observer = OrderingObserver(source)
    a.out.mkdir(parents=True, exist_ok=False)
    previous_args, previous_trace = sys.argv, sys.gettrace()
    report = dict(kind='observed_native_ordering_decisions', source=str(source.resolve()),
        observation_only=True, geometry_modified=False, joins_validated=False,
        independent_accuracy_measured=False, joins=observer.joins, returns=observer.returns)
    try:
        script = a.pipeline_tools / 'reconstruct_thread_sam3_original.py'
        sys.argv = [str(script), '--original-src', str(a.original_src), '--out', str(a.out), *native_args]
        sys.settrace(observer.trace)
        runpy.run_path(str(script), run_name='__main__')
        report['native_completed'] = True
    except BaseException as error:
        report.update(native_completed=False, error=str(error))
        raise
    finally:
        sys.settrace(previous_trace)
        sys.argv = previous_args
        (a.out / 'qp_ordering_trace.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
