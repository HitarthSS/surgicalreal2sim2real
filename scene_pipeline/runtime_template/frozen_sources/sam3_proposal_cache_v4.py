"""Reuse a real SAM3 proposal only under exact producing provenance.

The producer uses the actual model. Controllers can share these immutable
proposals, with logical attempts and preprocessing compute reported separately.
There is no learned-model substitute or filename-only cache match.
"""
import json
from pathlib import Path

from compact_agent_protocol import digest
from reconstruction_tools import sha256
from complex_case_diagnostics import write_json


def cache_key(inputs,checkpoint_sha,runner,helper,runtime_manifest,parameters):
    return {'schema':'sam3_proposal_cache_v1','parameters':parameters,
            'images':{s:sha256(inputs[s]) for s in ('left','right')},
            'checkpoint_sha256':checkpoint_sha,'runner_sha256':sha256(runner),
            'helper_sha256':sha256(helper),'producer_runtime_sha256':sha256(runtime_manifest)}


def safe_path(root,relative):
    if not isinstance(relative,str) or Path(relative).is_absolute():raise ValueError('Cache path must be relative')
    p=(Path(root)/relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('Cache path escapes declared root')
    return p


def record_proposal(root,backend,key):
    root=Path(root).resolve();backend=Path(backend).resolve()
    if not backend.is_relative_to(root):raise ValueError('Producer output outside cache root')
    report=json.loads((backend/'report.json').read_text())
    if report.get('status')!='proposal_generated' or report.get('parameters')!=key['parameters']:
        raise ValueError('Cache producer did not execute the requested proposal')
    provenance=report['input_sha256']
    expected={**key['images'],'checkpoint':key['checkpoint_sha256'],
              'runner':key['runner_sha256'],'source':key['helper_sha256']}
    if any(provenance.get(k,{}).get('sha256')!=v for k,v in expected.items()):
        raise ValueError('Producer provenance does not match cache key')
    artifacts={'report':backend/'report.json',**{s:backend/(s+'_thread_mask.png') for s in ('left','right')}}
    entry={'key':key,'artifacts':{k:{'relative_path':str(p.relative_to(root)),'sha256':sha256(p)}
                                 for k,p in artifacts.items()},'actual_model_producer':True,
           'training_eligible':False,'independent_accuracy_measured':False}
    index=root/'cache_index';index.mkdir(exist_ok=True)
    write_json(index/(digest(key)+'.json'),entry)
    return entry


def load_proposal(root,key):
    root=Path(root);entry=json.loads((root/'cache_index'/(digest(key)+'.json')).read_text())
    if entry.get('key')!=key or entry.get('actual_model_producer') is not True:
        raise ValueError('SAM3 cache provenance mismatch')
    if set(entry.get('artifacts',{}))!={'report','left','right'}:raise ValueError('Incomplete SAM3 cache')
    paths={}
    for name,item in entry['artifacts'].items():
        p=safe_path(root,item['relative_path'])
        if sha256(p)!=item['sha256']:raise ValueError('Cached SAM3 artifact changed: '+name)
        paths[name]=p
    report=json.loads(paths['report'].read_text())
    if report.get('status')!='proposal_generated' or report.get('parameters')!=key['parameters']:
        raise ValueError('Invalid cached backend result')
    return {s:str(paths[s]) for s in ('left','right')},report,entry
