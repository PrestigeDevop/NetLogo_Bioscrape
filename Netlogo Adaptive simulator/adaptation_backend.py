import numpy as np
from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model

PARAMETERS = ['k1','k2','k3','k4','k5','k6','KIA','KFA','KB','KFB','KAC','KBC']
PRESETS = {
    'NFBLB': [1,1,0.7,0.35,10,10,1,1,0.0001,0.0001,1,1],
    'IFFLP': [1,1,1,100,10,10,1,1,0.0001,100,1,1]
}
SPECIES = ['A','B','C','L']

def make_model(mode, p, initial, ligand):
    if mode not in PRESETS:
        raise ValueError('Mode must be NFBLB or IFFLP')
    if len(p) != 12 or min(p) <= 0 or ligand <= 0:
        raise ValueError('All rate constants, K values and input must be positive')
    if len(initial) != 3 or not all(0 <= x <= 1 for x in initial):
        raise ValueError('Initial activities must lie in [0,1]')
    driver = 'C' if mode == 'NFBLB' else 'A'
    rates = [
        ('A',True,'L*k1*(1-A)/(1-A+KIA)'),
        ('A',False,'k2*A/(A+KFA)'),
        ('B',True,driver+'*k3*(1-B)/(1-B+KB)'),
        ('B',False,'k4*B/(B+KFB)'),
        ('C',True,'A*k5*(1-C)/(1-C+KAC)'),
        ('C',False,'B*k6*C/(C+KBC)')
    ]
    reactions = [([], [s], 'general', {'rate':r}) if positive
                 else ([s], [], 'general', {'rate':r})
                 for s,positive,r in rates]
    return Model(species=SPECIES, reactions=reactions,
                 parameters=list(zip(PARAMETERS,map(float,p))),
                 initial_condition_dict=dict(zip(SPECIES,list(map(float,initial))+[float(ligand)])))

def simulate(m, values, times):
    m.set_species(dict(zip(SPECIES,map(float,values))))
    result = py_simulate_model(timepoints=np.asarray(times,dtype=float),Model=m,stochastic=False)
    data = result[SPECIES].to_numpy(dtype=float)
    if not np.isfinite(data).all():
        raise FloatingPointError('Nonfinite trajectory')
    if data[:,:3].min() < -1e-5 or data[:,:3].max() > 1.00001:
        raise FloatingPointError('Activity outside [0,1]; inspect solver and parameters')
    return data

def derivative(y,mode,p,I):
    a,b,c=y
    k1,k2,k3,k4,k5,k6,ki,kf,kb,kfb,kac,kbc=p
    return np.array([I*k1*(1-a)/(1-a+ki)-k2*a/(a+kf),
        (c if mode=='NFBLB' else a)*k3*(1-b)/(1-b+kb)-k4*b/(b+kfb),
        a*k5*(1-c)/(1-c+kac)-b*k6*c/(c+kbc)])

def initialize(mode,p,initial,I0,I1,warm,step_time,stop,dt,tol):
    global model,state,configuration,t,baseline,baseline_ok,history,peak,shocked
    if min(I0,I1,warm,dt,tol) <= 0 or I0 == I1 or not 0 < step_time < stop:
        raise ValueError('Use distinct positive inputs, positive times, and 0 < step-time < end-time')
    model=make_model(mode,p,initial,I0)
    state=simulate(model,list(initial)+[I0],[0,warm])[-1]
    baseline=float(state[2])
    if baseline < 1e-8:
        raise ValueError('Baseline too small for relative metrics')
    baseline_ok=bool(max(abs(derivative(state[:3],mode,p,I0))) < 1e-7)
    configuration=(mode,list(p),float(I0),float(I1),float(step_time),float(stop),float(dt),float(tol))
    t=0.0
    history=[]
    peak=0.0
    shocked=False
    return export_state()

def advance():
    global t,state,shocked,peak
    mode,p,I0,I1,event,stop,dt,tol=configuration
    if t >= stop-1e-10:
        return export_state()
    finish=min(t+dt,stop)
    while t < finish-1e-10:
        if not shocked and t >= event-1e-10:
            state[3]=I1
            shocked=True
        boundary=min(finish,event) if not shocked else finish
        h=boundary-t
        if h > 1e-10:
            state=simulate(model,state,[0,h])[-1]
            t=boundary
        if not shocked and t >= event-1e-10:
            state[3]=I1
            shocked=True
    if shocked:
        history.append((float(t),float(state[2])))
        peak=max(peak,abs(float(state[2])-baseline)/baseline*100)
    return export_state()

def summarize():
    mode,p,I0,I1,event,stop,dt,tol=configuration
    if not history:
        return [0.,0.,-1.,'PRE-STEP',0.]
    h=np.asarray(history)
    tail=h[h[:,0] >= max(event,t-10)]
    error=float(abs(tail[:,1].mean()-baseline)/baseline*100)
    span=float(np.ptp(tail[:,1])/baseline*100)
    relative=abs(h[:,1]-baseline)/baseline*100
    outside=np.where(relative > tol)[0]
    recovery=-1.
    if peak > tol and relative[-1] <= tol and len(outside):
        j=int(outside[-1])+1
        if j < len(h):
            recovery=float(h[j,0]-event)
    status='RUNNING'
    if t >= stop-1e-10:
        if not baseline_ok:
            status='INVALID: warm-up not settled'
        elif peak <= 1:
            status='LOW RESPONSE'
        elif error <= tol and span <= 0.2:
            status='PASS: near adaptation'
        else:
            status='FAIL / not settled'
    return [error,float(peak),recovery,status,span]

def export_state():
    return [float(t)]+list(map(float,state))+[baseline,bool(baseline_ok)]+summarize()

def run_sweep():
    global sweep_rows
    mode,p,I0,I1,event,stop,dt,tol=configuration
    trials=[('reference',list(p),I1)]
    for factor in [0.5,1.5,2.0]:
        trials.append(('input delta x'+str(factor),list(p),I0+(I1-I0)*factor))
    for factor in [0.5,2.0]:
        q=list(p);q[0]*=factor
        trials.append(('k1 x'+str(factor),q,I1))
        q=list(p);q[2]*=factor;q[3]*=factor
        trials.append(('B rates x'+str(factor),q,I1))
    q=list(p)
    if mode=='NFBLB':
        q[8]=0.1;q[9]=0.1
    else:
        q[8]=0.5
    trials.append(('weaken saturation (stress)',q,I1))
    sweep_rows=[]
    duration=stop-event
    for name,q,after in trials:
        if after <= 0:
            continue
        m=make_model(mode,q,[0.5]*3,I0)
        y=simulate(m,[0.5,0.5,0.5,I0],[0,100])[-1]
        settled=bool(max(abs(derivative(y[:3],mode,q,I0))) < 1e-7)
        base=float(y[2]);y[3]=after
        times=np.linspace(0,duration,max(2,int(duration/0.1)+1))
        z=simulate(m,y,times)
        tail=z[times>=max(0,duration-10),2]
        error=float(abs(tail.mean()-base)/base*100)
        peak_value=float(max(abs(z[:,2]-base))/base*100)
        span=float(np.ptp(tail)/base*100)
        passed=bool(settled and error<=tol and peak_value>1 and span<=0.2)
        sweep_rows.append([mode,name,float(after),base,float(tail.mean()),error,peak_value,span,settled,passed])
    return sweep_rows

def save_sweep(path):
    import csv
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['mode','test','post_input','own_baseline','final_mean_C','recovery_error_pct','peak_response_pct','tail_span_pct','baseline_settled','pass'])
        w.writerows(sweep_rows)
