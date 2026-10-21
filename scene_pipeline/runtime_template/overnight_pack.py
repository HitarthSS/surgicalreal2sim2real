"""Model-free overview and audited archive of the latest126 controller attempts, including failures."""
from pathlib import Path
import sys,json,hashlib,zipfile,html,shutil,time,subprocess
R=Path(__file__).resolve().parent
from overnight_execution import save,read
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
 config=read(R/'overnight_manifest.json');out=[];seen=set()
 for item in config['coverage_roots']:
  root=Path(item)
  for c in read(root/'job_manifest.json')['cases']:
   assert c not in seen,'Overlapping latest-pipeline coverage';seen.add(c)
   F=root/('snapshot_20261015_'+c);row=dict(case=c,source_root=str(root),methods={})
   progress={v['case_id']:v for v in read(root/'case_progress.json',[])}
   row['attempt']=progress.get(F.name,{}).get('status','pending')
   for method in ['heuristic','agent']:
    summary=read(F/method/'summary.json');candidate=(summary.get('candidate') or {}).get('report') or {}
    package=root/'all_geometry_export/advisor_handoff'/F.name/method
    row['methods'][method]=dict(fragments=candidate.get('fragment_count'),thread_checks=candidate.get('numerical_checks_passed'),completion=summary.get('completion'),attachment=(summary.get('attachment') or {}).get('status'),package=str(package),available=(package/'package_verified.json').exists())
   out.append(row)
 assert len(seen)==63
 return sorted(out,key=lambda x:x['case'])
def dashboard(rows):
 folder=R/'review';folder.mkdir(exist_ok=True);cards=[];table=[]
 for row in rows:
  c=row['case'];a=row['methods']['agent'];h=row['methods']['heuristic']
  table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [c,row['attempt'],h['fragments'],a['fragments'],a['thread_checks'],(a['completion'] or {}).get('unassessed_count',0),a['attachment']])+'</tr>')
  original=Path(row['source_root'])/'all_geometry_export/review'/(c+'_all_geometry.jpg');image=folder/original.name
  if original.exists():
   if not image.exists():shutil.copy2(original,image)
   cards.append('<section><h2>'+c+'</h2><p>'+html.escape('Heuristic / agent thread fragments: '+str(h['fragments'])+' / '+str(a['fragments'])+'. Agent thread checks: '+str(a['thread_checks'])+'. Pending endpoints: '+str((a['completion'] or {}).get('unassessed_count',0))+'.')+'</p><a href="'+image.name+'"><img src="'+image.name+'"></a></section>')
 count=sum(x['attempt']!='pending' for x in rows)
 page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Overnight current pipeline · all63</title><style>body{font:16px system-ui;background:#111820;color:#ecf2f8;max-width:1280px;margin:auto;padding:24px}td,th{padding:10px;border-bottom:1px solid #425063;text-align:left}img{width:100%}section{margin-top:36px}p{line-height:1.5}</style><h1>'+str(count)+'/63 latest-pipeline attempts</h1><p>Same-session development images. All full stereo views are retained. Partial and failed cases are shown. Geometry, attachments and tool fits are estimates; numerical checks are not independent accuracy. No Newton motion or policy validation.</p><table><tr><th>Case</th><th>Attempt</th><th>H fragments</th><th>A fragments</th><th>A thread checks</th><th>A pending</th><th>A attachment</th></tr>'+''.join(table)+'</table>'+''.join(cards)
 (folder/'index.html').write_text(page);save(R/'all63_progress.json',dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),total_cases=63,attempted_cases=count,rows=rows))
def finalize(rows):
 config=read(R/'overnight_manifest.json');roots=[Path(x) for x in config['coverage_roots']];missing=[];files={};packages=0;calls=0
 for root in roots:
  assert all(sha(root/n)==h for n,h in read(root/'runtime_manifest.json').items()),'Protected runtime changed'
  manifest=read(root/'job_manifest.json')
  assert all(sha(Path(manifest['raw_root'])/('snapshot_20261015_'+c)/n)==h for c in manifest['cases'] for n,h in manifest['raw_input_sha256'][c].items())
  for c in manifest['cases']:
   F=root/('snapshot_20261015_'+c);attempts=[]
   for p in F.rglob('input.json'):
    if not (p.parent.parent/'backend_manifest.json').exists():continue
    value=read(p);m=value['model'];assert m['name']=='gpt-6.1-sol' and m['retries']==0 and m['store']==False
    assert all(sha(Path(n))==h for n,h in value['image_sha256'].items())
    for side in ['left','right']:
     raw=str((Path(manifest['raw_root'])/F.name/(side+'_rect.png')).resolve());assert value['image_sha256'][raw]==manifest['raw_input_sha256'][c][side+'_rect.png']
    attempts.append(p.relative_to(F).parts[0])
   assert len(attempts)<=23 and attempts.count('perception')<=5 and attempts.count('agent_model_requests')<=12 and attempts.count('tool_agent')<=6
   calls+=len(attempts)
 for row in rows:
  assert row['attempt']!='pending','Unattempted scene remains'
  for method,entry in row['methods'].items():
   P=Path(entry['package']);record=read(P/'package_verified.json')
   if not record:missing.append(dict(case=row['case'],method=method,attempt=row['attempt']));continue
   for name,h in record['files'].items():
    p=P/name;assert sha(p)==h
    relative='advisor_handoff/snapshot_20261015_'+row['case']+'/'+method+'/'+name;files[relative]=p
   files['advisor_handoff/snapshot_20261015_'+row['case']+'/'+method+'/package_verified.json']=P/'package_verified.json';packages+=1
 common=roots[0]/'all_geometry_export/advisor_handoff'
 for root in roots[1:]:
  assets=root/'all_geometry_export/advisor_handoff/native_model'
  if assets.exists():
   for p in assets.rglob('*'):
    if p.is_file():assert sha(p)==sha(common/'native_model'/p.relative_to(assets)),'Native assets changed'
 code="""from pathlib import Path
import sys,json,numpy as np
R=Path(%r);sys.path.insert(0,%r);import load_scene
rows=json.loads((R/'all63_progress.json').read_text())['rows'];count=0
for row in rows:
 for method,entry in row['methods'].items():
  P=Path(entry['package'])
  if not (P/'package_verified.json').exists():continue
  data=load_scene.load(P/'scene.json');scene=data['scene']
  assert len(data['thread_fragments_newton_m'])==(scene.get('thread') or {}).get('fragment_count',0)
  assert len(data['tools'])==scene['available_visual_tool_count']
  assert all(np.isfinite(v).all() for v in data['geometry'].values() if np.asarray(v).dtype.kind in 'fi')
  count+=1
(R/'all63_portable_loader.json').write_text(json.dumps(dict(packages_loaded=count,physics_validated=False),indent=2))
"""%(str(R),str(roots[0]/'all_geometry_export'))
 p=subprocess.run(['docker','run','--rm','--network','none','--cpus','2','--memory','8g','--user','1009:1009','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-v','/home/hitarth:/home/hitarth:ro','-v',str(R)+':'+str(R),'--entrypoint','python3','sha256:bd675f584cb274db81af062e8fc0934fdfbdf87b228f57a6ff99e6a8529bb12e','-c',code],text=True,capture_output=True)
 assert p.returncode==0,p.stderr
 assert read(R/'all63_portable_loader.json')['packages_loaded']==packages
 for p in (common/'native_model').rglob('*'):
  if p.is_file():files['advisor_handoff/native_model/'+str(p.relative_to(common/'native_model'))]=p
 files['advisor_handoff/load_scene.py']=common/'load_scene.py'
 save(R/'all63_index.json',dict(rows=rows,expected_packages=126,available_packages=packages,missing_packages=missing,all_attempts_retained=True,independent_accuracy_measured=False,physics_validated=False))
 (R/'advisor_README.md').write_text('All63 October15 scenes through the current full reconstruction pipeline, heuristic and actual VLM. Read all63_index.json for failures/missing packages and scene.json for estimated/inferred/approximate status. Full stereo, all available thread fragments, needle and distal tools, poses/joints and attachment contracts are included. Static development snapshots, provisional metric calibration, unassigned PSM identity; no full-arm mapping, independent accuracy, Newton motion or imitation-policy validation.\n')
 files['advisor_handoff/index.json']=R/'all63_index.json';files['advisor_handoff/README.md']=R/'advisor_README.md'
 files['advisor_handoff/portable_loader_audit.json']=R/'all63_portable_loader.json'
 archive=R/'advisor_all63_current_pipeline.zip';pending=archive.with_suffix('.zip.part')
 assert shutil.disk_usage(R).free>sum(p.stat().st_size for p in files.values())+1024**3
 with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
  for name,p in files.items():z.write(p,name)
 with zipfile.ZipFile(pending) as z:
  assert z.testzip() is None and sum(n.endswith('/scene.json') for n in z.namelist())==packages
 pending.replace(archive)
 save(R/'all63_verification.json',dict(total_cases=63,attempted_cases=len(rows),expected_controller_packages=126,available_controller_packages=packages,missing_packages=missing,model_attempts=calls,API_budgets_verified=True,full_stereo_verified=True,sources_and_inputs_unchanged=True,archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,independent_accuracy_measured=False,physics_validated=False))
 print(json.dumps(read(R/'all63_verification.json')))
if __name__=='__main__':
 rows=inventory();dashboard(rows)
 if '--finalize' in sys.argv:finalize(rows)
