import numpy as np, time, sys, json
from ppcore import *
L=Like()

def make_model(name):
    if name=='flat':      # 1 param: Om
        lo=np.array([0.0]); hi=np.array([1.0]); names=['Om']
        f=lambda th: L.chi2_marg(th[0],None,-1.0)
    elif name=='wcdm':
        lo=np.array([0.0,-3.0]); hi=np.array([1.0,0.0]); names=['Om','w']
        f=lambda th: L.chi2_marg(th[0],None,th[1])
    elif name=='curved':
        lo=np.array([0.0,0.0]); hi=np.array([1.5,1.5]); names=['Om','OL']
        f=lambda th: L.chi2_marg(th[0],th[1],-1.0)
    elif name=='wcdmM':
        lo=np.array([0.0,-3.0,-20.0]); hi=np.array([1.0,0.0,-18.5]); names=['Om','w','M']
        f=lambda th: L.chi2_M(th[0],None,th[1],th[2])
    def logp(th):
        if np.any(th<lo) or np.any(th>hi): return -np.inf
        c=f(th)
        return -0.5*c if np.isfinite(c) else -np.inf
    return names,lo,hi,logp

def mh(logp,x0,cov,n,rng):
    d=len(x0); Lc=np.linalg.cholesky(cov)
    x=x0.copy(); lp=logp(x); out=np.empty((n,d)); acc=0
    for i in range(n):
        y=x+Lc@rng.standard_normal(d); ly=logp(y)
        if np.log(rng.random())<ly-lp: x,lp=y,ly; acc+=1
        out[i]=x
    return out,acc/n

def pilot(logp,x0,scale,n,rng):
    d=len(x0); cov=np.diag(scale**2); x=x0.copy(); lp=logp(x); chain=[]
    for i in range(n):
        if i>=500 and i%250==0:
            cov=np.cov(np.array(chain[i//2:]).T).reshape(d,d)*2.38**2/d+1e-10*np.eye(d)
        Lc=np.linalg.cholesky(cov)
        y=x+Lc@rng.standard_normal(d); ly=logp(y)
        if np.log(rng.random())<ly-lp: x,lp=y,ly
        chain.append(x.copy())
    return np.array(chain)

def rhat(chains):   # chains: (m,n,d); split
    m,n,d=chains.shape; h=n//2
    sp=np.concatenate([chains[:,:h],chains[:,h:2*h]],axis=0)
    M,N=sp.shape[0],sp.shape[1]
    W=sp.var(axis=1,ddof=1).mean(axis=0); B=N*sp.mean(axis=1).var(axis=0,ddof=1)
    return np.sqrt(((N-1)/N*W+B/N)/W)

def tau_int(x):
    x=x-x.mean(); n=len(x); f=np.fft.rfft(x,2*n); ac=np.fft.irfft(f*np.conj(f))[:n]; ac/=ac[0]
    t=1.0
    for k in range(1,n):
        t+=2*ac[k]
        if k>=5*t: break
    return t

if __name__=='__main__':
    names_to_run=sys.argv[1:]
    for mname in names_to_run:
        t0=time.time(); names,lo,hi,logp=make_model(mname); d=len(names)
        rng=np.random.default_rng(12345)
        # dispersed starts: uniform in central 80% of prior box, must have finite logp
        starts=[]
        while len(starts)<4:
            x=lo+(hi-lo)*(0.1+0.8*rng.random(d))
            if mname=='wcdmM': x[2]=-19.3+0.3*(rng.random()-0.5)*2
            if np.isfinite(logp(x)): starts.append(x)
        scale=(hi-lo)*0.02
        pc=pilot(logp,starts[0],scale,6000,rng)
        cov=np.cov(pc[3000:].T).reshape(d,d)*2.38**2/d
        NBURN,NMAIN=2000,20000
        chains=[];accs=[]
        for s in starts:
            ch,a=mh(logp,s,cov,NBURN+NMAIN,rng); chains.append(ch); accs.append(a)
        chains=np.array(chains); post=chains[:,NBURN:]
        R=rhat(post); taus=[[tau_int(post[k,:,j]) for j in range(d)] for k in range(4)]
        ess=post.shape[0]*post.shape[1]/np.mean(taus,axis=0)
        np.savez(f'chains/chain_{mname}.npz',full=chains,post=post,names=names,lo=lo,hi=hi,R=R,acc=accs)
        print(mname,"| time",round(time.time()-t0),"s | acc",np.round(accs,3),"| Rhat",np.round(R,4),"| tau(mean)",np.round(np.mean(taus,axis=0),1),"| ESS",np.round(ess),flush=True)
        flat=post.reshape(-1,d)
        for j,nm in enumerate(names):
            q=np.percentile(flat[:,j],[16,50,84]); print("   ",nm,"=",round(q[1],4),"+",round(q[2]-q[1],4),"-",round(q[1]-q[0],4),flush=True)
