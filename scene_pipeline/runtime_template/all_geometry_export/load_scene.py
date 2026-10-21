"""Portable numpy-only loader for advisor geometry, not a physics/VR controller."""
from pathlib import Path
import argparse,json,re
import numpy as np

def load(path):
 path=Path(path);root=path.parent;scene=json.loads(path.read_text())
 with np.load(root/scene['all_available_geometry_npz'],allow_pickle=False) as data:geometry={k:data[k].copy() for k in data.files}
 fragments={k[:-16]:points for k,points in geometry.items() if re.fullmatch(r'f\d+_points_newton_m',k)}
 tools={}
 for item in scene.get('visual_tools',[]):
  if item.get('geometry'):
   with np.load(root/item['geometry'],allow_pickle=False) as d:tools[item['visual_instance']]={k:d[k].copy() for k in d.files}
 return dict(scene=scene,thread_fragments_newton_m=fragments,geometry=geometry,tools=tools)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('scene',type=Path);args=parser.parse_args();data=load(args.scene)
 print(json.dumps(dict(scene=str(args.scene),thread_fragments=list(data['thread_fragments_newton_m']),tools=list(data['tools']),needle_available='needle_points_newton_m' in data['geometry'],physics_validated=False),indent=2))
