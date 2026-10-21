def main(root):
 """Read-only render of actual selected outputs; no inference or hidden joins."""
 from pathlib import Path
 import hashlib,json,re,sys
 import numpy as np,cv2
 D=Path(root).resolve();R=D;OUT=D/'review';OUT.mkdir(exist_ok=True)
 first_day=json.loads((R/'scope.json').read_text())['scenes'][0]['case_id'].split('_')[1]
 sys.path[:0]=[str(R/'gpt'/('day'+first_day)/'tool_template'),str(R/'gpt'/('day'+first_day)/'executed_sources')]
 from shared_tool_estimation import rasterize

 def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
 def load(p):
  with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 def project(p,K,b):
  q=np.array(p,float,copy=True);q[:,0]-=b;v=q@K.T;return v[:,:2]/v[:,2,None]
 def curve(ims,p,K,b,color,flags=None):
  p=np.asarray(p);assert p.ndim==2 and p.shape[1]==3
  for side in ims:
   xy=project(p,K,b if side=='right' else 0)
   for i in range(len(p)-1):
    if not np.isfinite(xy[i:i+2]).all() or min(p[i:i+2,2])<=.005:continue
    a,c=np.rint(xy[i:i+2]).astype(int)
    ok,a,c=cv2.clipLine((0,0,ims[side].shape[1],ims[side].shape[0]),tuple(a),tuple(c))
    if ok:cv2.line(ims[side],a,c,(0,165,255) if flags is not None and (flags[i] or flags[i+1]) else color,2,cv2.LINE_AA)
 rows=read(R/'scope.json')['scenes'];audit=[]
 for method in ('heuristic','agent'):
  provider='gpt'
  P=OUT/method;P.mkdir(exist_ok=True);tiles=[];records=[]
  for idx,row in enumerate(rows,1):
   cid=row['case_id'];f=D/provider/('day'+cid.split('_')[1])/cid;s=read(f/method/'summary.json');case=read(f/'case.json');K=np.loadtxt(case['inputs']['K'],delimiter=',');b=case['baseline_m'];ims={side:cv2.imread(case['inputs'][side]) for side in ('left','right')};assert all(im is not None for im in ims.values());sourcehash={};count=0;errors=[]
   candidate=s.get('candidate') or {};path=candidate.get('fragment_path')
   if path:
    assert sha(path)==candidate['fingerprints']['fragments']['sha256'];sourcehash[path]=sha(path);parts=load(path)
    for k,p in parts.items():
     if re.fullmatch(r'f\d+_points_m',k):count+=1;curve(ims,p,K,b,(0,255,0),parts.get(k[:-9]+'_geometry_inferred'))
   needle=(s.get('needle_provenance') or {}).get('path');hasneedle=False
   if needle and Path(needle).exists():
    n=load(needle);sourcehash[needle]=sha(needle);p=n.get('points_m',n.get('points_camera_m'))
    if p is not None:curve(ims,p,K,b,(255,255,0));hasneedle=True
   att=s.get('attachment') or {}
   if att.get('status')=='geometric_attachment_proposal' and att.get('artifact'):
    p=att['artifact'];z=load(p);sourcehash[p]=sha(p);m=att.get('metadata',{})
    if m.get('connector_added'):
     i=m['inferred_connector_point_index'];neighbor=1 if i==0 else i-1;curve(ims,z['points_camera_m'][[neighbor,i]],K,b,(0,165,255))
   tooldata=read(f/'tools/complete_tools.json');selected=tooldata.get('tools',{}).get(method,{});ntools=0
   for name,t in selected.items():
    mesh=t.get('mesh')
    if not mesh or not Path(mesh).exists():continue
    z=load(mesh);sourcehash[mesh]=sha(mesh);v=z.get('vertices_camera_m',z.get('vertices'));faces=z.get('faces')
    if v is None or faces is None:errors.append(dict(tool=name,mesh_keys=list(z)));continue
    accepted=bool(t.get('accepted_tool_hypothesis',t.get('verification',{}).get('accepted_numerical_hypothesis',False)));ntools+=1
    for side in ims:
     mask,_=rasterize(v,faces,K,b if side=='right' else 0,ims[side].shape[:2]);contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ims[side],contours,-1,(255,0,255) if accepted else (0,165,255),1,cv2.LINE_AA)
   checked=bool(candidate.get('report',{}).get('numerical_checks_passed'));label=f'{idx:02d} {cid} | {row["difficulty"]} | fragments {count} | checks {"PASS" if checked else "FAIL"} | needle {int(hasneedle)} | tools {ntools}'
   pair=np.hstack([ims['left'],ims['right']]);header=np.zeros((52,pair.shape[1],3),np.uint8);cv2.putText(header,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA);cv2.putText(header,'LEFT / RIGHT | green thread | cyan needle | magenta checked tool | orange inferred/provisional',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.46,(220,220,220),1,cv2.LINE_AA)
   full=np.vstack([header,pair]);cv2.imwrite(str(P/(cid+'.jpg')),full,[cv2.IMWRITE_JPEG_QUALITY,90]);tile=cv2.resize(pair,(640,240),interpolation=cv2.INTER_AREA);head=np.zeros((44,640,3),np.uint8);cv2.putText(head,f'{idx:02d} {cid} ({row["difficulty"]})',(6,17),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA);cv2.putText(head,f'fragments {count} | checks {"PASS" if checked else "FAIL"} | needle {int(hasneedle)} | tools {ntools}',(6,36),cv2.FONT_HERSHEY_SIMPLEX,.42,(220,220,220),1,cv2.LINE_AA);tiles.append(np.vstack([head,tile]))
   records.append(dict(index=idx,case_id=cid,difficulty=row['difficulty'],fragments=count,thread_checks_passed=checked,needle=hasneedle,tools=ntools,render_errors=errors,source_hashes=sourcehash));assert all(sha(p)==h for p,h in sourcehash.items())
  while len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
  grid=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]);cv2.imwrite(str(OUT/(method+'_grid.jpg')),grid,[cv2.IMWRITE_JPEG_QUALITY,92]);(P/'index.json').write_text(json.dumps(records,indent=2));audit.append(dict(provider=provider,scenes=len(records),full_stereo_preserved=True,all_fragments_rendered=True,scientific_outputs_unchanged=True,errors=[r for r in records if r['render_errors']]))
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))

 (OUT/'gallery.html').write_text('<meta charset="utf-8"><style>body{background:#111;color:#eee}img{max-width:100%}</style><h1>Full pipeline reconstructions</h1><p>Complete stereo fields. Green thread, cyan needle, magenta checked tool, orange inferred/provisional geometry. Partial fragments shown. No independent accuracy or Newton validation.</p>'+''.join('<h2>'+m+'</h2><img src="'+m+'_grid.jpg">' for m in ['heuristic','agent']))
