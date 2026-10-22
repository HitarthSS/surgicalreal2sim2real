import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_pipeline.background import build, decision, configuration, similarity
from scene_pipeline.background_export import export_background
from scene_pipeline.background_stage import run_case


class BackgroundTests(unittest.TestCase):
    def fixture(self, root, curved=False):
        h,w=64,80; y,x=np.mgrid[:h,:w]
        texture=np.stack([x*2+20,y*2+20,np.full((h,w),100)],axis=-1).astype('uint8')
        inputs={}
        for side in ['left','right']:
            p=root/(side+'.png'); Image.fromarray(texture).save(p);inputs[side]=str(p)
            mask=np.zeros((h,w),'uint8'); mask[28:40,30:45]=255
            p=root/(side+'_mask.png');Image.fromarray(mask).save(p);inputs['thread_'+side+'_mask']=str(p)
        depth=np.full((h,w),.15) if not curved else .15+.02*((x-w/2)/(w/2))**2
        depth[28:40,30:45]=.06
        p=root/'depth.npy';np.save(p,depth);inputs['depth_npy']=str(p)
        p=root/'K.csv';np.savetxt(p,[[100,0,w/2],[0,100,h/2],[0,0,1]],delimiter=',');inputs['K']=str(p)
        p=root/'case.json';p.write_text(json.dumps(dict(inputs=inputs,baseline_m=.005)))
        return p

    def test_plane_metric_geometry_texture_and_handoff_transform_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            result=run_case(case,root/'background',dict(stride=4,min_support_pixels=100))
            self.assertEqual(result['status'],'accepted');self.assertEqual(result['model'],'plane')
            self.assertLess(result['diagnostics']['residual_p90_m'],1e-6)
            with np.load(root/'background/background_geometry.npz') as z:
                self.assertTrue(np.allclose(z['vertices_camera_m'][:,2],.15))
                self.assertTrue(z['geometry_inferred'].any())
            output=root/'package';output.mkdir();T=np.eye(4);T[1,3]=.2
            delivered=export_background(root,output,T)
            self.assertEqual(delivered['status'],'accepted')
            with np.load(output/delivered['geometry']) as z:
                self.assertTrue(np.allclose(z['vertices_newton_m'][:,1],z['vertices_camera_m'][:,1]+.2))
            self.assertTrue((output/'background/texture.png').exists())
            for rel, expected in result['files'].items():
                self.assertEqual(hashlib.sha256((output/'background'/rel).read_bytes()).hexdigest(), expected)
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
            with self.assertRaises(FileExistsError):run_case(case,root/'background')
            (root/'background/texture.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):export_background(root,root/'second_package',T)

    def test_curved_surface_chooses_mesh_and_never_bridges_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root,curved=True)
            result=build(case,root/'background',dict(stride=4,min_support_pixels=100,plane_tolerance_m=.0003))
            self.assertEqual(result['model'],'depth_mesh')
            with np.load(root/'background/background_geometry.npz') as z:
                v=z['vertices_camera_m'];xy=v[:,:2]/v[:,2,None]*100+[40,32]
                centers=xy[z['faces']].mean(1)
                self.assertFalse(((centers[:,0]>=27)&(centers[:,0]<=47)&(centers[:,1]>=25)&(centers[:,1]<=42)).any())

    def test_selector_is_bounded_and_failure_preserves_deterministic_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root)
            def bad_selector(*args):return dict(surface_model='shell_command',material_preset='metal',roughness=0,metallic=1,reason='bad')
            result=build(case,root/'background',dict(min_support_pixels=100),bad_selector)
            self.assertEqual(result['status'],'accepted')
            self.assertEqual(result['selection_source'],'deterministic_after_selector_failure')
            self.assertEqual(result['model'],'plane')
        with self.assertRaises(ValueError):configuration(dict(stride=0))
        with self.assertRaises(ValueError):decision(dict(surface_model='plane',material_preset='cloth',roughness=.5,metallic=0,reason='ok',stiffness=10))

    def test_missing_masks_fail_instead_of_reconstructing_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root);c=json.loads(case.read_text())
            c['inputs']={k:v for k,v in c['inputs'].items() if 'mask' not in k};case.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'Foreground masks required'):run_case(case,root/'background')
            self.assertEqual(json.loads((root/'background_status.json').read_text())['state'],'failed_preserved')

    def test_sam3d_similarity_anchors_and_safe_fallback(self):
        rng=np.random.default_rng(1);a=rng.normal(size=(12,3));rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        b=2*(a@rotation.T)+[1,2,3];s,r,t=similarity(a,b)
        self.assertAlmostEqual(s,2);self.assertTrue(np.allclose(s*(a@r.T)+t,b))
        with self.assertRaises(ValueError):similarity(np.ones((6,3)),np.ones((6,3)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root)
            choose=lambda *a:dict(surface_model='sam3d',material_preset='cloth',roughness=.9,metallic=0,reason='Try candidate')
            result=build(case,root/'background',dict(min_support_pixels=100),choose)
            self.assertEqual(result['model'],'plane');self.assertTrue(result['fallback_reasons'])
            self.assertFalse(result['sam3d_model_executed'])

    def test_external_sam3d_mesh_is_registered_to_stereo_and_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=self.fixture(root)
            xy=np.array([[4,4],[12,4],[20,4],[4,12],[12,12],[20,12]],float)
            camera=np.c_[(xy-[40,32])*.15/100,np.full(len(xy),.15)]
            local=(camera-[.01,.02,.04])/2
            path=root/'candidate.npz'
            np.savez(path,vertices=local,faces=np.array([[0,3,1],[1,3,4],[1,4,2],[2,4,5]]),anchor_points_local=local,anchor_pixels_left=xy)
            choose=lambda *a:dict(surface_model='sam3d',material_preset='rigid_plastic',roughness=.4,metallic=0,reason='Registered candidate')
            result=build(case,root/'background',dict(min_support_pixels=100,sam3d_candidate=str(path)),choose)
            self.assertEqual(result['model'],'sam3d');self.assertEqual(result['status'],'accepted')
            self.assertAlmostEqual(result['sam3d_candidate_alignment']['scale'],2)
            with np.load(root/'background/background_geometry.npz') as z:
                self.assertTrue(np.allclose(z['vertices_camera_m'],camera))


if __name__=='__main__':unittest.main()
