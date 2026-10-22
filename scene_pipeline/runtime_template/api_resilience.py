"""Provider admission, explicit outage state, and raw-attempt accounting."""
from pathlib import Path
from contextlib import contextmanager
import email.utils,fcntl,json,os,random,re,time

QUOTA_CODES={'insufficient_quota','credit_balance_exhausted','organization_spend_limit_exceeded','project_spend_limit_exceeded','organization_usage_limit_exceeded','billing_hard_limit_reached'}
STAGE_CAPS={'perception':5,'agent_model_requests':12,'tool_agent':5,'needle_size_refinement':1,'background':1}

class ProviderBlocked(RuntimeError):pass
class AttemptBudgetExhausted(RuntimeError):pass

def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
    os.chmod(p,0o600)

def scrub(value,key=''):
    value=str(value or '')
    if key:value=value.replace(key,'[REDACTED]')
    value=re.sub(r'(?i)bearer\s+\S+','Bearer [REDACTED]',value)
    return re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',value)[:2000]

def retry_after(headers,now=None):
    headers={k.lower():v for k,v in headers.items()};value=headers.get('retry-after')
    if value is None:return None
    try:return max(0.,float(value))
    except (ValueError,TypeError):
        try:return max(0.,email.utils.parsedate_to_datetime(value).timestamp()-(time.time() if now is None else now))
        except (ValueError,TypeError,OverflowError):return None

def classify_http(status,body,headers,key=''):
    try:error=json.loads(body).get('error',{})
    except (ValueError,TypeError):error={}
    if not isinstance(error,dict):error={}
    code=scrub(error.get('code'),key);kind=scrub(error.get('type'),key)
    terminal=code in QUOTA_CODES or kind=='insufficient_quota' or status in (401,403)
    transient=not terminal and status in (429,500,502,503,504)
    safe_headers={k:v for k,v in headers.items() if k.lower() in ('retry-after','x-request-id') or k.lower().startswith('x-ratelimit-')}
    return dict(http_status=status,error_code=code,error_type=kind,error_message=scrub(error.get('message'),key),
        response_headers=safe_headers,retry_after_seconds=retry_after(headers),
        retryable=transient,requires_account_action=terminal,
        classification='account_blocked' if terminal else 'temporary_provider_failure' if transient else 'request_failed')

@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield

def gate_path():
    value=os.environ.get('ISMR_API_STATE_DIR')
    return Path(value)/'provider_state.json' if value else None

def admission(clock=time.time,sleep=time.sleep):
    p=gate_path()
    if p is None:return
    with locked(p.with_suffix('.lock')):
        state=read(p)
        if state.get('blocked'):raise ProviderBlocked(state.get('error_code') or state.get('classification','Provider blocked'))
        wait=max(0.,state.get('next_allowed_at',0)-clock())
        if wait>300:raise ProviderBlocked('Provider cooldown exceeds this stage deadline; pending work retained')
        if wait:sleep(wait)
        state.update(next_allowed_at=clock()+2.,updated=clock());write(p,state)

def record_failure(error,clock=time.time):
    p=gate_path()
    if p is None:return
    with locked(p.with_suffix('.lock')):
        state=read(p);state.update(error_code=error.get('error_code'),classification=error['classification'],updated=clock())
        failures=state.get('consecutive_failures',0)+1;state['consecutive_failures']=failures
        if error['requires_account_action'] or failures>=3:state['blocked']=True
        elif error['retryable']:state['next_allowed_at']=clock()+max(10.,error.get('retry_after_seconds') or 0.)
        write(p,state)

def record_success():
    p=gate_path()
    if p is None:return
    with locked(p.with_suffix('.lock')):
        state=read(p);state.update(consecutive_failures=0,last_success=time.time());write(p,state)

def retry_delay(error,attempt,elapsed,remaining_attempts,jitter=random.random):
    if not error['retryable'] or remaining_attempts<=0 or attempt>=3:return None
    delay=(error.get('retry_after_seconds') if error.get('retry_after_seconds') is not None else min(60.,2.**attempt))+jitter()
    return delay if elapsed+delay<=300 else None

def reserve_input(folder,value):
    """A retry consumes a distinct physical attempt in the unchanged scene cap."""
    scene=os.environ.get('ISMR_API_SCENE_DIR');folder=Path(folder)
    if not scene:folder.mkdir();write(folder/'input.json',value);return
    scene=Path(scene);stage=folder.resolve().relative_to(scene.resolve()).parts[0]
    with locked(scene/'.api_budget.lock'):
        counts={k:0 for k in STAGE_CAPS}
        for p in scene.rglob('input.json'):
            if (p.parent.parent/'backend_manifest.json').exists():
                s=p.relative_to(scene).parts[0];counts[s]=counts.get(s,0)+1
        if sum(counts.values())>=23 or counts.get(stage,0)>=STAGE_CAPS.get(stage,0):
            raise AttemptBudgetExhausted('Physical API attempt cap reached: '+stage)
        folder.mkdir();write(folder/'input.json',value)

def outcomes(scene):
    scene=Path(scene);completed=failed=missing=0;errors={};deferred=[];decisions={}
    for p in scene.rglob('input.json'):
        if not (p.parent.parent/'backend_manifest.json').exists():continue
        r=read(p.parent/'result.json')
        logical=(str(p.parent.parent),r.get('logical_decision',p.parent.name))
        decisions[logical]=decisions.get(logical,False) or r.get('status')=='completed'
        if r.get('status')=='completed':completed+=1
        elif r.get('status')=='failed':
            failed+=1;code=r.get('error_code') or str(r.get('http_status') or r.get('error_type'));errors[code]=errors.get(code,0)+1
        else:missing+=1
    for p in scene.rglob('provider_deferred_*.json'):
        deferred.append(str(p));logical=(str(p.parent),int(p.stem.rsplit('_',1)[1]))
        decisions.setdefault(logical,False)
    gate=read(gate_path()) if gate_path() else {}
    pending=any(not resolved for resolved in decisions.values())
    return dict(physical_attempts=completed+failed+missing,completed_responses=completed,failed_attempts=failed,
        missing_results=missing,deferred_decisions=len(deferred),errors=errors,agent_work_pending=pending,
        logical_decisions_completed=sum(decisions.values()),logical_decisions_pending=sum(not x for x in decisions.values()),
        provider_blocked=bool(gate.get('blocked')),provider_error_code=gate.get('error_code'))
