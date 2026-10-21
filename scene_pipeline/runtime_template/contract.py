"""Exact rerun scope, immutable input identities, bounded charged attempts."""
from pathlib import Path
from collections import Counter
import hashlib,json

def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except FileNotFoundError:return {} if default is None else default

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def test_rows(root):
    root=Path(root)
    if not (root/'split_manifest.json').exists():root=root.parent
    manifest=read(root/'split_manifest.json');claimed=manifest.pop('manifest_sha256')
    canonical=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==claimed,'Locked split changed'
    rows=[r for r in manifest['scenes'] if r['partition']=='validation']
    assert len(rows)==30 and len({r['case_id'] for r in rows})==30
    assert Counter(r['difficulty'] for r in rows)==dict(easy=15,hard=15)
    return rows

def require_test(root,cid):
    scope=read(Path(root)/'recovery_scope.json')
    if scope and cid not in scope['cases']:raise ValueError('Outside the locked development scope')
    matches=[r for r in test_rows(root) if r['case_id']==cid]
    if len(matches)!=1:raise ValueError('Scene outside the locked30-scene development run')
    return matches[0]

def api_attempts(folder):
    rows=[]
    for p in Path(folder).rglob('input.json'):
        if not (p.parent.parent/'backend_manifest.json').exists():continue
        value=read(p)
        assert value['model']['name'] in ('gpt-6.1-sol','google/gemma-4-31B-it') and value['model']['retries'] in (0,2)
        rows.append(dict(path=str(p),stage=p.relative_to(folder).parts[0]))
    counts=Counter(r['stage'] for r in rows)
    assert len(rows)<=23 and counts['perception']<=5 and counts['agent_model_requests']<=12 and counts['tool_agent']<=5 and counts['needle_size_refinement']<=1,'API budget breach'
    return rows

def core_paths(case):
    return {name:case['inputs'][name] for name in ('left','right','K','depth_npy')}

def verify_binding(original,working):
    assert core_paths(original)==core_paths(working),'Core input path identity mismatch'

def rates(rows):
    result={}
    for difficulty,total in [('easy',15),('hard',15),('overall',30)]:
        selected=[r for r in rows if difficulty=='overall' or r['difficulty']==difficulty]
        result[difficulty]=dict(total=total,attempted=len(selected),independent_accuracy=None)
        for key in ('thread_checked','checked_single_curve','needle_available','attachment_available','two_tool_meshes','package_available'):
            count=sum(bool(r[key]) for r in selected)
            result[difficulty][key]=dict(count=count,denominator=total,rate=count/total)
    return result
