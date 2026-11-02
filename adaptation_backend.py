import csv
import json
import os
import sys
import time
import uuid
import tempfile
import subprocess
import traceback
from collections import deque
from pathlib import Path
import numpy as np
from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model

SPECIES=['A','B','C','L']
PARAMETER_NAMES=['k1','k2','k3','k4','k5','k6','KIA','KFA','KB','KFB','KAC','KBC']
WARMUP=500.0
DURATION=300.0
REPORT_DT=0.2
FINAL_WINDOW=20.0
ERROR_LIMIT=.02
SPAN_LIMIT=.002
RANDOM_CANDIDATES=12
TRAINING_SAMPLES=8
VALIDATION_SAMPLES=30
SHORTLIST_SIZE=5
SENSITIVITY_LOG_STEP=.01
MAX_LOG_POINTS=100000
revision=0
last_protocol=None
process=None
job_folder=None
prepared_job=None
completed_result=None
last_result=None
candidate_bank=[]
run_log=deque(maxlen=MAX_LOG_POINTS)
change_log=[]


def validate(mode,p,initial):
    if mode not in ('NFBLB','IFFLP'): raise ValueError('Choose NFBLB or IFFLP')
    if len(p)!=12 or not np.isfinite(p).all() or min(p)<=0:
        raise ValueError('Expected 12 finite positive parameters')
    if len(initial)!=3 or not np.isfinite(initial).all() or min(initial)<0 or max(initial)>1:
        raise ValueError('Initial activities must be finite and in [0,1]')


def derivative(y,mode,p,ligand):
    a,b,c=y
    k1,k2,k3,k4,k5,k6,kia,kfa,kb,kfb,kac,kbc=p
    driver=c if mode=='NFBLB' else a
    return np.array([
        ligand*k1*(1-a)/(1-a+kia)-k2*a/(a+kfa),
        driver*k3*(1-b)/(1-b+kb)-k4*b/(b+kfb),
        a*k5*(1-c)/(1-c+kac)-b*k6*c/(c+kbc)])


def make_model(mode,p,initial,ligand):
    validate(mode,p,initial)
    if not np.isfinite(ligand) or ligand<=0: raise ValueError('Invalid ligand')
    driver='C' if mode=='NFBLB' else 'A'
    rates=[('A',True,'L*k1*(1-A)/(1-A+KIA)'),('A',False,'k2*A/(A+KFA)'),
           ('B',True,driver+'*k3*(1-B)/(1-B+KB)'),('B',False,'k4*B/(B+KFB)'),
           ('C',True,'A*k5*(1-C)/(1-C+KAC)'),('C',False,'B*k6*C/(C+KBC)')]
    reactions=[([], [s], 'general', {'rate':r}) if positive else
               ([s], [], 'general', {'rate':r}) for s,positive,r in rates]
    return Model(species=SPECIES,reactions=reactions,
        parameters=list(zip(PARAMETER_NAMES,map(float,p))),
        initial_condition_dict=dict(zip(SPECIES,list(map(float,initial))+[float(ligand)])))


def simulate(model,values,times):
    values=np.asarray(values,dtype=float); times=np.asarray(times,dtype=float)
    if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite starting state')
    model.set_species(dict(zip(SPECIES,map(float,values))))
    result=py_simulate_model(timepoints=times,Model=model,stochastic=False)
    data=result[SPECIES].to_numpy(dtype=float).copy()
    bad=np.argwhere(~np.isfinite(data))
    if len(bad):
        i,j=map(int,bad[0])
        raise FloatingPointError('Nonfinite trajectory: species='+SPECIES[j]+
            ', requested time='+str(float(times[i]))+', starting state='+repr(values.tolist()))
    if data[:,:3].min() < -1e-5 or data[:,:3].max() > 1.00001:
        raise FloatingPointError('Activity outside [0,1]: min='+str(float(data[:,:3].min()))+
                                 ', max='+str(float(data[:,:3].max())))
    return data


def settled(state,mode,p,ligand):
    return bool(np.max(np.abs(derivative(state[:3],mode,p,ligand)))<1e-6)


def metrics(times,outputs,base,old,new,valid,complete):
    times=np.asarray(times,dtype=float); outputs=np.asarray(outputs,dtype=float)
    scale=max(abs(base),1e-12); relative=np.abs(outputs-base)/scale
    change=abs(new-old)/old; peak=float(relative.max())
    tail=outputs[times>=max(0.0,float(times[-1])-FINAL_WINDOW)]
    error=float(np.max(np.abs(tail-base))/scale); span=float(np.ptp(tail)/scale)
    sensitivity=peak/max(change,1e-12); precision=min(1e9,change/max(error,1e-12))
    recovery=-1.0; outside=np.where(relative>ERROR_LIMIT)[0]
    if len(outside) and relative[-1]<=ERROR_LIMIT:
        i=int(outside[-1])+1
        if i<len(times) and times[-1]-times[i]>=FINAL_WINDOW: recovery=float(times[i])
    failures=[]
    if not valid: failures.append('unsettled baseline')
    if peak<=.01: failures.append('peak <= 1%')
    if sensitivity<=1: failures.append('S <= 1')
    if precision<=10: failures.append('P <= 10')
    if error>ERROR_LIMIT: failures.append('recovery error > 2%')
    if span>SPAN_LIMIT: failures.append('tail variation > 0.2%')
    passed=not failures; reason='PASS' if passed else '; '.join(failures)
    status='RUNNING: provisional metrics'
    if complete: status='PASS: sensitive near-adaptation' if passed else 'FAIL: '+reason
    return dict(error=100*error,peak=100*peak,S=float(sensitivity),P=float(precision),
        recovery=recovery,passed=bool(passed),reason=reason,status=status,numerical_error=False)


def log_point():
    run_log.append([float(clock)]+list(map(float,live_state))+[
        float(baseline),bool(baseline_valid),bool(event_valid),int(revision)])


def initialize(mode,parameters,initial):
    global live_model,live_state,live_config,clock,baseline,baseline_valid
    global event_time,old_ligand,history,last_protocol,revision,event_valid,completed_result
    p=list(map(float,parameters)); initial=list(map(float,initial)); validate(mode,p,initial)
    model=make_model(mode,p,initial,1.0)
    state=simulate(model,initial+[1.0],[0.0,WARMUP])[-1].copy()
    if state[2]<=1e-8: raise ValueError('Baseline C too small')
    stop_job()
    live_model=model; live_state=state; live_config=(mode,p,initial)
    clock=0.0; baseline=float(state[2]); baseline_valid=settled(state,mode,p,1.0)
    event_time=None; old_ligand=1.0; history=[]; last_protocol=None
    event_valid=True; completed_result=None; revision+=1
    run_log.clear(); change_log.clear()
    change_log.append(dict(time=0.0,type='setup',revision=revision,parameters=p.copy()))
    log_point()


def update_parameters(parameters):
    global live_model,live_config,event_valid,revision,completed_result
    mode,old,initial=live_config
    p=list(map(float,parameters)); validate(mode,p,initial)
    if np.array_equal(p,old): return False
    model=make_model(mode,p,initial,float(live_state[3]))
    live_model=model; live_config=(mode,p,initial)
    if event_time is not None: event_valid=False
    revision+=1; completed_result=None
    change_log.append(dict(time=float(clock),type='parameters',revision=revision,parameters=p.copy()))
    log_point()
    return True


def apply_stimulus(level):
    global baseline,baseline_valid,event_time,old_ligand,history,last_protocol
    global revision,event_valid,completed_result
    level=float(level); old=float(live_state[3])
    if not np.isfinite(level) or level<=0: raise ValueError('Ligand must be positive and finite')
    if abs(old-level)<1e-12: raise ValueError('Choose a different ligand')
    if live_state[2]<=1e-8: raise ValueError('Current C too small for relative metrics')
    mode,p,_=live_config
    baseline=float(live_state[2]); baseline_valid=settled(live_state,mode,p,old)
    event_time=float(clock); old_ligand=old; live_state[3]=level
    history=[(0.0,baseline)]; last_protocol=(old,level); event_valid=True
    revision+=1; completed_result=None
    change_log.append(dict(time=float(clock),type='stimulus',revision=revision,old=old,new=level,
                           baseline=baseline,settled_baseline=bool(baseline_valid)))
    log_point()


def finished():
    return bool(event_time is not None and clock-event_time>=DURATION-1e-9)


def advance():
    global live_state,clock
    step=REPORT_DT
    if event_time is not None and not finished(): step=min(step,DURATION-(clock-event_time))
    live_state=simulate(live_model,live_state,[0.0,step])[-1].copy(); clock+=step
    if event_time is not None and clock-event_time<=DURATION+1e-8:
        history.append((float(min(DURATION,clock-event_time)),float(live_state[2])))
    log_point()


def export_state():
    mode,p,_=live_config
    now=settled(live_state,mode,p,float(live_state[3]))
    if event_time is None:
        r=dict(error=0.0,peak=0.0,S=0.0,P=0.0,recovery=-1.0,status='WAITING: apply stimulus')
    else:
        h=np.asarray(history)
        r=metrics(h[:,0],h[:,1],baseline,old_ligand,float(live_state[3]),baseline_valid,finished())
        if not event_valid: r['status']='INTERRUPTED: parameters changed; settle then stimulate'
    return [float(clock)]+list(map(float,live_state))+[float(baseline),bool(baseline_valid),
        r['error'],r['peak'],r['S'],r['P'],r['recovery'],r['status'],finished(),bool(now),bool(event_valid)]


def trial(mode,p,initial,old,new,curve=False):
    try:
        model=make_model(mode,p,initial,old)
        state=simulate(model,list(initial)+[float(old)],[0.0,WARMUP])[-1].copy()
        base=float(state[2])
        if base<=1e-8: raise ValueError('Baseline C too small')
        valid=settled(state,mode,p,old); state[3]=float(new)
        times=np.linspace(0,DURATION,int(round(DURATION/REPORT_DT))+1)
        trajectory=simulate(model,state,times)
        r=metrics(times,trajectory[:,2],base,old,new,valid,True)
        r.update(baseline=base,curve=list(map(float,trajectory[::5,2]/base)) if curve else [],
                 raw_curve=list(map(float,trajectory[::5,2])) if curve else [])
        return r
    except Exception as exc:
        return dict(passed=False,error=-1.0,peak=-1.0,S=-1.0,P=-1.0,recovery=-1.0,
            baseline=-1.0,reason='ERROR: '+str(exc),status='ERROR',numerical_error=True,curve=[],raw_curve=[])


def signature(): return json.dumps([revision,live_config,last_protocol])


def stop_job():
    global process,prepared_job
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
        except (PermissionError,OSError) as exc:
            raise RuntimeError('Stopping worker failed: '+str(exc)+'. Job folder: '+str(job_folder)) from exc
    process=None; prepared_job=None
    # Deliberately retain job folders/logs; no Windows file deletion during polling.


def prepare_job(kind,selected_ligand,factor):
    global prepared_job,completed_result,candidate_bank
    mode,p,initial=live_config; new=float(selected_ligand); factor=float(factor)
    if kind not in ('report','optimize','sensitivity'): raise ValueError('Unknown job kind')
    if not np.isfinite([new,factor]).all() or new<=0 or not 1<factor<=20:
        raise ValueError('Positive ligand and range factor >1 and <=20 required')
    old=1.0
    if last_protocol is not None and abs(last_protocol[1]-new)<1e-10: old=float(last_protocol[0])
    if abs(old-new)<1e-12: raise ValueError('No input step: select another ligand')
    stop_job(); completed_result=None
    p=np.asarray(p,dtype=float); rng=np.random.default_rng(12345)
    candidate_bank=[dict(label='original',p=p.copy())]
    for i in range(12):
        for multiplier in np.geomspace(1/factor,factor,5):
            if np.isclose(multiplier,1): continue
            q=p.copy(); q[i]*=float(multiplier)
            candidate_bank.append(dict(label=PARAMETER_NAMES[i],p=q))
    for _ in range(RANDOM_CANDIDATES):
        q=p*np.exp(rng.uniform(-np.log(factor),np.log(factor),12))
        candidate_bank.append(dict(label='joint',p=q))
    prepared_job=dict(kind=kind,mode=mode,parameters=list(map(float,p)),initial=initial,
        old=old,new=new,factor=factor,signature=signature(),candidates=[])
    return [len(candidate_bank),old,new]


def get_candidate(index):
    c=candidate_bank[int(index)]
    return [c['label'],list(map(float,c['p']))]


def writable_workspace():
    roots=[]
    if os.environ.get('LOCALAPPDATA'): roots.append(Path(os.environ['LOCALAPPDATA'])/'AdaptationTool'/'jobs')
    roots.extend([Path(tempfile.gettempdir())/'AdaptationToolJobs',Path.home()/'AdaptationToolJobs'])
    errors=[]
    for root in roots:
        try:
            root.mkdir(parents=True,exist_ok=True)
            folder=root/('job_'+uuid.uuid4().hex)
            folder.mkdir()
            probe=folder/'write_test.txt'; probe.write_text('ok',encoding='utf-8'); probe.unlink()
            return folder
        except OSError as exc: errors.append(str(root)+': '+str(exc))
    raise PermissionError('No writable worker folder. '+' | '.join(errors))


def launch_job(backend_path,python_executable,fitted):
    global process,job_folder
    if prepared_job is None: raise ValueError('Prepare a job first')
    executable=Path(python_executable)
    if not executable.is_file(): raise ValueError('Python executable is not a file: '+str(executable))
    if prepared_job['kind']=='optimize' and len(fitted)!=len(candidate_bank):
        raise ValueError('Incomplete candidate list')
    prepared_job['candidates']=[dict(label=c['label'],p=list(map(float,p))) for c,p in zip(candidate_bank,fitted)]
    stage='create writable workspace'
    try:
        job_folder=writable_workspace()
        stage='copy backend into workspace'
        worker_path=job_folder/'worker_backend.py'
        worker_path.write_text(Path(backend_path).read_text(encoding='utf-8'),encoding='utf-8')
        stage='write job input'
        (job_folder/'job.json').write_text(json.dumps(prepared_job),encoding='utf-8')
        command=[str(executable),str(worker_path),'--adaptive-worker',
                 str(job_folder/'job.json'),str(job_folder/'result.json')]
        stage='launch configured Python executable'
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with open(job_folder/'worker.log','w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=str(job_folder),stdout=log,stderr=log,creationflags=flags)
    except Exception as exc:
        raise RuntimeError('Worker stage ['+stage+'] failed: '+repr(exc)+
            ' | executable='+str(executable)+' | job folder='+str(job_folder)) from exc
    return str(job_folder)


def write_json(path,value):
    path=Path(path)
    for attempt in range(4):
        try:
            temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            temp.write_text(json.dumps(value),encoding='utf-8'); os.replace(temp,path); return
        except PermissionError:
            if attempt==3: raise
            time.sleep(.1*(attempt+1))


def analyze_sensitivity(job,status):
    p=np.asarray(job['parameters'],dtype=float); h=SENSITIVITY_LOG_STEP
    nominal=trial(job['mode'],p,job['initial'],job['old'],job['new'],True)
    times=np.linspace(0,DURATION,int(round(DURATION/REPORT_DT))+1)[::5]
    rows=[]; summaries=[]
    for i,name in enumerate(PARAMETER_NAMES):
        status('Sensitivity '+str(i+1)+'/12: '+name)
        plus=p.copy(); minus=p.copy(); plus[i]*=np.exp(h); minus[i]*=np.exp(-h)
        rp=trial(job['mode'],plus,job['initial'],job['old'],job['new'],True)
        rm=trial(job['mode'],minus,job['initial'],job['old'],job['new'],True)
        if nominal['numerical_error'] or rp['numerical_error'] or rm['numerical_error']:
            summaries.append(dict(parameter=name,ok=False,reason='Numerical trial failure',priority=None))
            continue
        raw=(np.asarray(rp['raw_curve'])-np.asarray(rm['raw_curve']))/(2*h*nominal['baseline'])
        normalized=(np.asarray(rp['curve'])-np.asarray(rm['curve']))/(2*h)
        tail=times>=DURATION-FINAL_WINDOW; early=times<=30
        priority=float(np.max(np.abs(normalized)))
        peak_index=int(np.argmax(np.abs(normalized)))
        summaries.append(dict(parameter=name,ok=True,priority=priority,
            peak_time=float(times[peak_index]),early_max=float(np.max(np.abs(normalized[early]))),
            final_window_max=float(np.max(np.abs(normalized[tail]))),
            raw_max=float(np.max(np.abs(raw))),plus_pass=bool(rp['passed']),minus_pass=bool(rm['passed'])))
        rows.extend([[name,float(t),float(a),float(b)] for t,a,b in zip(times,raw,normalized)])
    summaries.sort(key=lambda r:(not r['ok'],-r['priority'] if r['ok'] else 0))
    return dict(log_step=h,raw_normalization='dC/dlog(parameter) divided by nominal pre-stimulus C',
        adaptation_normalization='d(C/own_pre_stimulus_C - 1)/dlog(parameter)',
        summary=summaries,columns=['parameter','time','raw_sensitivity','adaptation_sensitivity'],rows=rows)


def worker(job_path,result_path):
    job=json.loads(Path(job_path).read_text(encoding='utf-8'))
    progress=Path(result_path).with_name('progress.json')
    rng=np.random.default_rng(67890)
    tn=rng.uniform(.9,1.1,(TRAINING_SAMPLES,12)); tw=rng.uniform(.5,1.5,(TRAINING_SAMPLES,12))
    vn=rng.uniform(.9,1.1,(VALIDATION_SAMPLES,12)); vw=rng.uniform(.5,1.5,(VALIDATION_SAMPLES,12))
    def status(text): write_json(progress,dict(message=text))
    def evaluate(p,curve=False): return trial(job['mode'],p,job['initial'],job['old'],job['new'],curve)
    def score(p,factors):
        results=[evaluate(np.asarray(p)*m) for m in factors]
        return sum(int(r['passed']) for r in results),sum(int(r['numerical_error']) for r in results)
    try:
        if job['kind']=='report':
            nominal=evaluate(job['parameters'])
            status('Nearby robustness: 30 samples'); near,ne=score(job['parameters'],vn)
            status('Wider robustness: 30 samples'); wide,we=score(job['parameters'],vw)
            payload=dict(kind='report',near=near,wide=wide,n=VALIDATION_SAMPLES,
                nominal_pass=nominal['passed'],reason=nominal['reason'],errors=ne+we,plot=[[],[]])
        else:
            sensitivity=analyze_sensitivity(job,status)
            if job['kind']=='sensitivity':
                payload=dict(kind='sensitivity',sensitivity=sensitivity,plot=[[],[]])
            else:
                rank={r['parameter']:i for i,r in enumerate(sensitivity['summary'])}
                candidates=sorted(job['candidates'],key=lambda c:
                    -1 if c['label']=='original' else rank.get(c['label'],99))
                rows=[]
                for i,c in enumerate(candidates):
                    status('Candidate '+str(i+1)+'/'+str(len(candidates)))
                    rows.append(dict(p=c['p'],label=c['label'],result=evaluate(c['p'],True)))
                original=rows[0]
                eligible=sorted([r for r in rows[1:] if r['result']['passed']],key=lambda r:r['result']['error'])
                shortlist=[original]
                for row in eligible:
                    if not any(np.array_equal(row['p'],x['p']) for x in shortlist): shortlist.append(row)
                    if len(shortlist)>=SHORTLIST_SIZE+1: break
                training=[]
                for i,row in enumerate(shortlist):
                    status('Robustness screening '+str(i+1)+'/'+str(len(shortlist)))
                    n,ne=score(row['p'],tn); w,we=score(row['p'],tw)
                    training.append(dict(index=i,near=n,wide=w,errors=ne+we))
                result=[False,[],0,0,0,0,0,'No nominally passing alternative found.',0,0]
                if len(shortlist)>1:
                    best=max(training[1:],key=lambda r:(min(r['near'],r['wide']),r['near']+r['wide']))
                    p=shortlist[best['index']]['p']
                    status('Validation: original nearby'); on,one=score(original['p'],vn)
                    status('Validation: original wider'); ow,owe=score(original['p'],vw)
                    status('Validation: candidate nearby'); cn,cne=score(p,vn)
                    status('Validation: candidate wider'); cw,cwe=score(p,vw)
                    accepted=cn>=on and cw>=ow and (cn>on or cw>ow)
                    result=[bool(accepted),list(map(float,p)) if accepted else [],on,cn,ow,cw,
                        VALIDATION_SAMPLES,'All-12-parameter improvement ready to apply.' if accepted else
                        'No validated improvement without a tradeoff.',one+owe,cne+cwe]
                chosen=[original]+eligible[:5]
                times=list(map(float,np.linspace(0,DURATION,int(round(DURATION/REPORT_DT))+1)[::5]))
                curves=[[str(i)+' '+r['label'],r['result']['curve']] for i,r in enumerate(chosen) if r['result']['curve']]
                table=[dict(label=r['label'],parameters=r['p'],passed=r['result']['passed'],
                    error=r['result']['error'],S=r['result']['S'],reason=r['result']['reason']) for r in rows]
                payload=dict(kind='optimize',validation=result,training=training,candidates=table,
                             sensitivity=sensitivity,plot=[times,curves])
        payload['job']=job
        write_json(result_path,dict(ok=True,payload=payload))
    except Exception:
        write_json(result_path,dict(ok=False,error=traceback.format_exc()))


def poll_job():
    global completed_result,last_result,process
    if process is None: return ['idle','']
    if process.poll() is None:
        message='Working in background'
        try: message=json.loads((job_folder/'progress.json').read_text(encoding='utf-8'))['message']
        except (OSError,ValueError,KeyError): pass
        return ['running',message]
    try:
        result=json.loads((job_folder/'result.json').read_text(encoding='utf-8'))
        if not result['ok']: return ['error',result['error']+' | folder='+str(job_folder)]
        payload=result['payload']; last_result=payload
        if payload['job']['signature']!=signature(): return ['stale','Earlier snapshot; not applied']
        completed_result=payload
        if payload['kind']=='report':
            return ['report',payload['near'],payload['wide'],payload['n'],
                payload['nominal_pass'],payload['reason'],payload['errors']]
        if payload['kind']=='sensitivity': return ['sensitivity','Sensitivity analysis complete']
        r=payload['validation']
        return ['optimized',r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]]
    except Exception as exc:
        log=''
        try: log=(job_folder/'worker.log').read_text(encoding='utf-8')[-4000:]
        except OSError: pass
        return ['error',repr(exc)+' | '+log+' | folder='+str(job_folder)]
    finally:
        # Process has exited. No file cleanup or handle deletion here.
        process=None


def ready_values():
    if completed_result is None or completed_result['kind']!='optimize': return []
    if completed_result['job']['signature']!=signature(): return []
    r=completed_result['validation']
    return r[1] if r[0] else []


def apply_result():
    p=ready_values()
    if not p: raise ValueError('No current validated improvement')
    update_parameters(p)
    return p


def result_plot(): return last_result.get('plot',[[],[]]) if last_result else [[],[]]


def sensitivity_summary():
    if not last_result or 'sensitivity' not in last_result: return []
    return [[r['parameter'],r['ok'],r.get('priority') if r['ok'] else -1,
             r.get('peak_time',-1),r.get('final_window_max',-1)] for r in last_result['sensitivity']['summary']]


def save_results(path):
    if last_result is None: raise ValueError('No completed result to save')
    target=Path(path)
    target.write_text(json.dumps(last_result,indent=2),encoding='utf-8')
    if 'sensitivity' in last_result:
        with open(str(target)+'.sensitivity.csv','w',newline='',encoding='utf-8') as handle:
            w=csv.writer(handle); w.writerow(last_result['sensitivity']['columns']); w.writerows(last_result['sensitivity']['rows'])
    if 'candidates' in last_result:
        with open(str(target)+'.candidates.csv','w',newline='',encoding='utf-8') as handle:
            w=csv.writer(handle); w.writerow(['label']+PARAMETER_NAMES+['passed','error_pct','S','reason'])
            for r in last_result['candidates']: w.writerow([r['label']]+r['parameters']+[r['passed'],r['error'],r['S'],r['reason']])


def save_live_data(path):
    mode,p,initial=live_config
    payload=dict(configuration=dict(mode=mode,parameters=p,parameter_names=PARAMETER_NAMES,initial=initial),
        state=list(map(float,live_state)),simulation_time=float(clock),events=change_log,
        columns=['time','A','B','C','input','event_baseline','settled_baseline','valid_event','revision'],
        trajectory=list(run_log),retained_points=len(run_log),maximum_retained_points=MAX_LOG_POINTS)
    target=Path(path); target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with open(str(target)+'.trajectory.csv','w',newline='',encoding='utf-8') as handle:
        w=csv.writer(handle); w.writerow(payload['columns']); w.writerows(payload['trajectory'])


def read_saved(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    return [[str(k),str(v)[:500]] for k,v in data.items() if k not in ('trajectory','candidates','sensitivity','job')]


def saved_parameters(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if 'configuration' in data: config=data['configuration']; mode=config['mode']; p=config['parameters']
    elif 'job' in data:
        mode=data['job']['mode']; p=data['job']['parameters']
        if data.get('validation') and data['validation'][0]: p=data['validation'][1]
    else: raise ValueError('Unrecognized saved file')
    validate(mode,p,[.5,.5,.5])
    return [mode,list(map(float,p))]


if __name__=='__main__' and len(sys.argv)==4 and sys.argv[1]=='--adaptive-worker':
    worker(sys.argv[2],sys.argv[3])
