"""Wait for available GPU memory, then run the authorized 21-case batch."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parent
RAW=Path('/home/hitarth/ismr_2027_data/incoming/2026-10-15')
CHECK=Path('/data/xliang/ismr_2027/sam3_pipeline10_20261019_v1/model/sam3_human27_best_epoch2.pt')
FFS=Path('/home/hitarth/Fast-FoundationStereo/weights/23-36-37')
OLD=Path('/home/hitarth/integrated_surgical_scene_outputs/ismr_requested_six_20261013_v1')
CACHE6=Path('/home/hitarth/integrated_surgical_scene_outputs/ismr_shared_refiner_v6_20261013/mask_cache')
REG=json.loads((ROOT/'preregistration.json').read_text())
LOCK=threading.Lock()
PENDING=[s for s in REG['cases'] if s not in REG['cached_preprocessing_only_cases']]
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def gpu_free():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):int(b.strip()) for a,b in (row.split(',') for row in text.splitlines())}
def wait_gpu(gpu):
    while gpu_free().get(gpu,0)<6144:time.sleep(60)
def case_json(short):
    cid='snapshot_20261015_'+short;folder=ROOT/cid;folder.mkdir(exist_ok=True)
    raw=RAW/cid;cal=json.loads((raw/'calibration.json').read_text())
    K=[row[:3] for row in cal['P1']]
    baseline=-float(cal['P2'][0][3])/float(cal['P2'][0][0])*.001
    assert abs(baseline-.006820275089549331)<1e-9,'Unexpected calibration'
    (folder/'K.csv').write_text('\n'.join(','.join(str(float(x)) for x in row) for row in K)+'\n')
    case=dict(case_id=cid,session_id='dvrk_snapshots_2026-10-15',split='development',
        calibration_id='prior_matched_calibration_units_provisional',baseline_m=baseline,
        manual_interventions='No new masks/manual tool fits; inherited one-thread setting; physical units provisional.',
        scene_priors={'single_thread':True},
        inputs=dict(left=str(raw/'left_rect.png'),right=str(raw/'right_rect.png'),
            K=str(folder/'K.csv'),depth_npy=str(folder/'stereo/depth_meter.npy'),
            **{f'{target}_{side}_mask':str(folder/'segmentation'/f'{side}_{target}_mask.png')
              for target in ['thread','needle'] for side in ['left','right']}))
    write(folder/'case.json',case);return folder,case
def docker_base(gpu):
    return ['docker','run','--rm','--gpus','device='+str(gpu),'--cpus','8','--memory','32g',
        '-e','PYTHONPATH='+str(ROOT/'executed_sources'),'-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','OMP_NUM_THREADS=8','-e','MPLBACKEND=Agg',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent),'-v','/home/hitarth/integrated_surgical_scene_outputs/ismr_selected26_20261015_v1:/home/hitarth/integrated_surgical_scene_outputs/ismr_selected26_20261015_v1:ro','-v',str(RAW)+':'+str(RAW)+':ro',
        '-v',str(CHECK)+':/checkpoint.pt:ro','-v',str(FFS)+':/ffs_weights:ro']
def execute(command,log):
    began=time.perf_counter()
    with log.open('w') as stream:
        try:
            p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            code=p.returncode;error=None
        except subprocess.TimeoutExpired:code=None;error='timeout'
    return dict(exit_code=code,error=error,seconds=time.perf_counter()-began,command=command)
def process_chunk(gpu,batch,shorts):
    successes=[]
    for short in shorts:
        folder,case=case_json(short);wait_gpu(gpu)
        command=docker_base(gpu)+['--entrypoint','python3','sha256:901ee73c7e4b7904e4c52d93224635f471bb14fd1ad536cd207971ff530e14b2',
            str(ROOT/'executed_sources/run_two_arm_segmentation_checkpoint.py'),
            '--snapshot',str(RAW/folder.name),'--pipeline-tools',str(ROOT/'pipeline-tools'),
            '--checkpoint','/checkpoint.pt','--out',str(folder/'segmentation')]
        seg=execute(command,folder/'segmentation.log');wait_gpu(gpu)
        wrapper=str(ROOT/'pipeline-tools/run_fast_foundationstereo_depth.py')
        boot="import torch,runpy,sys,numpy as np; torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8); free,total=torch.cuda.mem_get_info(); assert free>=6*1024**3,'Need6GiBfree'; torch.cuda.set_per_process_memory_fraction(8*1024**3/total); sys.argv=["+repr(wrapper)+"]+sys.argv[1:]; runpy.run_path("+repr(wrapper)+",run_name='__main__')"
        command=docker_base(gpu)+['--entrypoint','python3','sha256:ee0f672b3cf5f11f7c5485eb151fb6b199d27bb3654058b82fbc9fe4658bbd5d','-c',boot,
            '--repo',str(ROOT/('ffs_checkout_gpu'+str(gpu))),
            '--model','/ffs_weights/model_best_bp2_serialize.pth','--left',case['inputs']['left'],
            '--right',case['inputs']['right'],'--K',case['inputs']['K'],'--baseline-m',str(case['baseline_m']),
            '--valid-iters','32','--max-disp','416','--scale','1','--out-dir',str(folder/'stereo')]
        stereo=execute(command,folder/'stereo.log')
        passed=seg['exit_code']==0 and stereo['exit_code']==0
        report=dict(case_id=folder.name,status='preprocessed' if passed else 'failed',gpu=gpu,
            input_sha256=REG['input_sha256'][short],segmentation=seg,stereo=stereo,
            physical_units_confirmed=False,baseline_m_hypothesis=case['baseline_m'])
        write(folder/'preprocessing.json',report)
        print(json.dumps({'case_id':folder.name,'preprocessing':report['status'],'gpu':gpu}),flush=True)
        if passed:successes.append(short)
        else:write(ROOT/'ready'/(folder.name+'.json'),dict(status='failed',error='GPU preprocessing failed; inspect logs'))
    if successes:
        wait_gpu(gpu);cache=ROOT/'mask_caches'/('gpu'+str(gpu)+'_batch'+str(batch))
        command=docker_base(gpu)+['--entrypoint','python3','sha256:901ee73c7e4b7904e4c52d93224635f471bb14fd1ad536cd207971ff530e14b2',
            str(ROOT/'executed_sources/produce_shared_mask_repairs_v6.py'),
            '--input-root',str(RAW),'--pipeline-tools',str(ROOT/'pipeline-tools'),
            '--checkpoint','/checkpoint.pt','--out',str(cache),'--case-ids']+['snapshot_20261015_'+s for s in successes]
        result=execute(command,ROOT/('repair_gpu'+str(gpu)+'_batch'+str(batch)+'.log'))
        write(ROOT/('repair_gpu'+str(gpu)+'_batch'+str(batch)+'.json'),result)
        for short in successes:
            write(ROOT/'ready'/('snapshot_20261015_'+short+'.json'),dict(status='ready' if result['exit_code']==0 else 'failed',
                cache=str(cache),error=None if result['exit_code']==0 else 'Shared repair producer failed'))
def gpu_worker(gpu):
    batch=0
    while True:
        with LOCK:
            if not PENDING:return
        if gpu_free().get(gpu,0)<6144:time.sleep(60);continue
        with LOCK:
            if not PENDING:return
            chunk=PENDING[:2];del PENDING[:2]
        batch+=1
        try:process_chunk(gpu,batch,chunk)
        except Exception as error:
            for short in chunk:
                folder,_=case_json(short);marker=ROOT/'ready'/(folder.name+'.json')
                if not marker.exists():write(marker,dict(status='failed',error=f'{type(error).__name__}: {error}'))
            print('GPU worker error: '+type(error).__name__,flush=True)
def cpu_command(script,interactive=False):
    command=['docker','run','--rm','--cpus','8','--memory','16g']
    if interactive:command+=['-i']
    return command+['-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2',
        '-e','OPENBLAS_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','MPLBACKEND=Agg',
        '-v',str(ROOT.parent)+':'+str(ROOT.parent),'-v','/home/hitarth/integrated_surgical_scene_outputs/ismr_selected26_20261015_v1:/home/hitarth/integrated_surgical_scene_outputs/ismr_selected26_20261015_v1:ro','-v',str(RAW)+':'+str(RAW)+':ro',
        '-v',str(CACHE6)+':'+str(CACHE6)+':ro','--entrypoint','python3',
        'sha256:bd675f584cb274db81af062e8fc0934fdfbdf87b228f57a6ff99e6a8529bb12e',str(ROOT/script)]
def main():
    credential=sys.stdin.readline().strip()
    if not credential:raise RuntimeError('Missing protected stdin credential')
    expected=json.loads((ROOT/'execution_environment.json').read_text())['model_sha256']
    for label,path in [('sam3',CHECK),('ffs',FFS/'model_best_bp2_serialize.pth'),('ffs_config',FFS/'cfg.yaml')]:
        if sha(path)!=expected[label]:raise RuntimeError('Frozen model/config hash changed: '+label)
    while True:
        free=gpu_free()
        write(ROOT/'queue_status.json',dict(state='waiting_for_gpu',remaining_cases=len(REG['cases']),
            skipped_completed_cases=REG['cached_controller_cases'],gpu_free_mib=free,required_free_mib=6144,
            updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid()))
        if free.get(1,0)>=6144:break
        time.sleep(60)
    write(ROOT/'queue_status.json',dict(state='running',remaining_cases=len(REG['cases']),
        skipped_completed_cases=REG['cached_controller_cases'],started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid()))
    with (ROOT/'controllers_stdout.log').open('w') as log:
        cpu=subprocess.Popen(cpu_command('run_cpu_batch.py',True),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        cpu.stdin.write((credential+'\n').encode());cpu.stdin.close()
    del credential
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures=[pool.submit(gpu_worker,gpu) for gpu in [1]]
        for f in futures:f.result()
    write(ROOT/'gpu_complete.json',dict(ready_markers=len(list((ROOT/'ready').glob('*.json')))))
    code=cpu.wait()
    if code==0:
        with (ROOT/'review_stdout.log').open('w') as log:
            code=subprocess.run(cpu_command('finish_report.py'),stdout=log,stderr=subprocess.STDOUT).returncode
    write(ROOT/'queue_status.json',dict(state='complete' if code==0 else 'failed',
        exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        expected_cases=len(REG['cases']),skipped_completed_cases=REG['cached_controller_cases']))
if __name__=='__main__':
    try:main()
    except Exception as error:
        write(ROOT/'queue_status.json',dict(state='failed',error=f'{type(error).__name__}: {error}',pid=os.getpid()))
        raise
