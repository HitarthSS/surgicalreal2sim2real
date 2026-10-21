"""Bind packaging metadata to the selected working inputs, preserving old receipts."""
from pathlib import Path
import sys,shutil
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'executed_sources')]
from contract import read,save,verify_binding,sha
from reconstruction_tools import fingerprints

def synchronize(folder):
    folder=Path(folder);original=read(folder/'case.json');rows=[]
    for method in ('heuristic','agent'):
        summary=read(folder/method/'summary.json')
        context=R/'export_context'/method/folder.name
        source=Path(summary.get('selected_case_path') or context/'case.json')
        case=read(source);verify_binding(original,case)
        previous=read(context/'preprocessing_cache.json')
        if previous and not (context/'pre_finalization_preprocessing_cache.json').exists():
            save(context/'pre_finalization_preprocessing_cache.json',previous)
        if (context/'case.json').exists() and not (context/'pre_finalization_case.json').exists():
            shutil.copy2(context/'case.json',context/'pre_finalization_case.json')
        save(context/'case.json',case)
        save(context/'preprocessing_cache.json',dict(input_fingerprints=fingerprints(case['inputs']),
            selected_case_path=str(source),selected_case_sha256=sha(source),
            reason='Selected mask/occlusion inputs after finalization; core image/calibration/depth paths unchanged',
            scientific_candidates_modified=False))
        rows.append(dict(controller=method,selected_case=str(source),core_path_identity_verified=True))
    save(folder/'export_context_binding.json',rows)

if __name__=='__main__':synchronize(sys.argv[1])
