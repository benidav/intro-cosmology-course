import numpy as np, matplotlib, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt, corner
from scipy.optimize import minimize
from ppcore import *
plt.rcParams.update({'font.size':9,'axes.linewidth':0.8})
LV=(0.68,0.95)
L=Like(); out={}
def stats(x):
    q=np.percentile(x,[16,50,84]); return dict(med=q[1],up=q[2]-q[1],lo=q[1]-q[0])
ld=lambda n: np.load(f'chains/chain_{n}.npz',allow_pickle=True)
# ---------------- best fits (chi2 minima, marginalized over offset) ----------------
r1=minimize(lambda p:L.chi2_marg(p[0],None,-1.0),[0.33],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-8})
r2=minimize(lambda p:L.chi2_marg(p[0],None,p[1]),[0.3,-0.9],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-8})
r3=minimize(lambda p:L.chi2_marg(p[0],p[1],-1.0),[0.3,0.7],method='Nelder-Mead',options={'xatol':1e-6,'fatol':1e-8})
out['chi2min']=dict(flat=r1.fun,wcdm=r2.fun,curved=r3.fun,flat_Om=r1.x[0],wcdm_x=list(r2.x),curved_x=list(r3.x),N=len(L.m))
N=len(L.m); out['dBIC_wcdm_minus_flat']=(r2.fun-r1.fun)+1*np.log(N); out['dBIC_curved_minus_flat']=(r3.fun-r1.fun)+1*np.log(N)
out['dchi2_flat_minus_wcdm']=r1.fun-r2.fun; out['dchi2_flat_minus_curved']=r1.fun-r3.fun
# ---------------- Fig: Hubble diagram ----------------
Om0=r1.x[0]; off=L.best_offset(Om0,None,-1.0)
z=L.zhd; m=L.m; err=np.sqrt(np.diag(L.C))
mu0=mu_model(z,L.zhel,Om0,None,-1.0)+off
res=m-mu0
def res_model(Om,OL=None,w=-1.0):
    mu=mu_model(z,L.zhel,Om,OL,w); o=L.best_offset(Om,OL,w); return mu+o-mu0
order=np.argsort(z); zg=np.geomspace(0.0102,2.3,300)
def curve(Om,OL=None,w=-1.0):
    mu=mu_model(zg,zg,Om,OL,w); return mu  # for plotting curves (zhel~zhd)
fig,(a1,a2)=plt.subplots(2,1,figsize=(6.6,6.2),sharex=True,gridspec_kw={'height_ratios':[2.2,1.2],'hspace':0.05})
a1.errorbar(z,m,yerr=err,fmt='.',ms=2.2,lw=0.4,color='0.55',alpha=0.6,label=r"Pantheon+ ($z_{\rm HD}>0.01$, %d SNe)"%N)
a1.plot(zg,curve(Om0)+off,c='C0',lw=1.6,label=r"flat $\Lambda$CDM, $\Omega_m=%.3f$"%Om0)
a1.set_ylabel(r"$m_B^{\rm corr}$"); a1.set_xscale('log'); a1.legend(loc='lower right',fontsize=8)
# binned residuals
bins=np.geomspace(0.0102,2.3,22); idx=np.digitize(z,bins)
bz,br,be=[],[],[]
for k in range(1,len(bins)):
    s=idx==k
    if s.sum()>=3:
        w_=1/err[s]**2; bz.append(np.exp(np.mean(np.log(z[s])))); br.append(np.sum(w_*res[s])/w_.sum()); be.append(1/np.sqrt(w_.sum()))
a2.axhline(0,c='C0',lw=1.4)
a2.errorbar(bz,br,yerr=be,fmt='o',ms=3.5,color='k',capsize=2,lw=0.9,label="binned data")
def rel(Om,OL=None,w=-1.0):
    mu=mu_model(zg,zg,Om,OL,w); mu0g=mu_model(zg,zg,Om0,None,-1.0)
    # offsets chosen by GLS on the full sample
    return mu+L.best_offset(Om,OL,w)-(mu0g+off)
a2.plot(zg,rel(1.0,None,-1.0),c='C3',lw=1.2,ls='--',label=r"Einstein--de Sitter $\Omega_m=1$")
a2.plot(zg,rel(0.3,0.0),c='C2',lw=1.2,ls='-.',label=r"open, $\Omega_m=0.3,\ \Omega_\Lambda=0$")
a2.set_ylim(-0.45,0.35); a2.set_xlabel(r"$z_{\rm HD}$"); a2.set_ylabel(r"$\Delta m$ (vs best flat $\Lambda$CDM)")
a2.legend(fontsize=7.5,loc='lower left',ncol=1)
plt.savefig('./fig_pp_hubble.pdf',bbox_inches='tight'); plt.savefig('./fig_pp_hubble.png',dpi=130,bbox_inches='tight'); plt.close()
out['res_rms']=float(np.std(res)); 
# ---------------- Fig: flat wCDM corner ----------------
c2=ld('wcdm'); s2=c2['post'].reshape(-1,2)
out['flat']=stats(ld('flat')['post'].reshape(-1)); out['wcdm']={'Om':stats(s2[:,0]),'w':stats(s2[:,1]),'rho':float(np.corrcoef(s2.T)[0,1]),'P_w_lt_-1':float(np.mean(s2[:,1]<-1)),'Rhat':list(map(float,c2['R'])),'acc':list(map(float,c2['acc']))}
fig=corner.corner(s2,labels=[r"$\Omega_m$",r"$w$"],levels=LV,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt=".3f",bins=50,smooth=1.0,
                  plot_datapoints=False,color='C0',range=[(0,0.7),(-1.6,-0.4)],fig=plt.figure(figsize=(5.2,5.2)),title_kwargs={'fontsize':9},
                  hist_kwargs={'lw':1.3},contour_kwargs={'linewidths':1.3})
ax=np.array(fig.axes).reshape(2,2)
ax[1,0].axhline(-1,c='crimson',lw=1.0,ls='--'); ax[1,1].axvline(-1,c='crimson',lw=1.0,ls='--')
ax[1,0].plot([Om0],[-1],marker='*',c='crimson',ms=10,ls='')
plt.savefig('./fig_pp_wcdm.pdf',bbox_inches='tight'); plt.savefig('./fig_pp_wcdm.png',dpi=130,bbox_inches='tight'); plt.close()
# ---------------- Fig: trace plots ----------------
full=c2['full']; fig,axs=plt.subplots(2,1,figsize=(6.6,4.2),sharex=True)
for j,(nm,yl) in enumerate([("Om",r"$\Omega_m$"),("w",r"$w$")]):
    for k in range(4): axs[j].plot(np.arange(6000),full[k,:6000,j],lw=0.5,alpha=0.85)
    axs[j].axvline(2000,c='k',ls=':',lw=1); axs[j].set_ylabel(yl)
axs[1].set_xlabel("step"); axs[0].text(2060,axs[0].get_ylim()[1]*0.93,"burn-in discarded  $\\leftarrow$ | $\\rightarrow$ kept",fontsize=8,va='top')
plt.tight_layout(); plt.savefig('./fig_pp_trace.pdf'); plt.savefig('./fig_pp_trace.png',dpi=130); plt.close()
# ---------------- Fig: curved LCDM corner with derived quantities ----------------
c3=ld('curved'); s3=c3['post'].reshape(-1,2)
Ok=1-s3[:,0]-s3[:,1]; q0=0.5*s3[:,0]-s3[:,1]
S3=np.c_[s3,Ok,q0]
out['curved']={'Om':stats(s3[:,0]),'OL':stats(s3[:,1]),'Ok':stats(Ok),'q0':stats(q0),'P_q0_lt_0':float(np.mean(q0<0)),'P_Ok_gt_0':float(np.mean(Ok>0)),
  'rho_OmOL':float(np.corrcoef(s3.T)[0,1]),'Rhat':list(map(float,c3['R'])),'acc':list(map(float,c3['acc'])),
  'P_flat_band':float(np.mean(np.abs(Ok)<0.1)),'Om_OL_cov_eigs':list(map(float,np.linalg.eigvalsh(np.cov(s3.T))))}
fig=corner.corner(S3,labels=[r"$\Omega_m$",r"$\Omega_\Lambda$",r"$\Omega_k=1-\Omega_m-\Omega_\Lambda$",r"$q_0=\Omega_m/2-\Omega_\Lambda$"],levels=LV,
                  quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt=".2f",bins=45,smooth=1.0,plot_datapoints=False,color='C0',
                  fig=plt.figure(figsize=(8.2,8.2)),range=[(0.0,0.6),(0.0,0.95),(-0.45,0.75),(-0.72,0.12)],title_kwargs={'fontsize':8},hist_kwargs={'lw':1.3},contour_kwargs={'linewidths':1.3},label_kwargs={'fontsize':9})
ax=np.array(fig.axes).reshape(4,4)
xx=np.linspace(0,1.5,100)
ax[1,0].plot(xx,1-xx,c='crimson',ls='--',lw=1.1); ax[1,0].plot(xx,xx/2,c='k',ls=':',lw=1.1)
ax[1,0].text(0.98,0.02,"flat: $\\Omega_k=0$",color='crimson',fontsize=7.5,transform=ax[1,0].transAxes,ha='right',va='bottom')
ax[1,0].text(0.97,0.30,"$q_0=0$",fontsize=7.5,transform=ax[1,0].transAxes,ha='right',va='top',rotation=0)
ax[2,0].axhline(0,c='crimson',ls='--',lw=1.1); ax[3,0].axhline(0,c='k',ls=':',lw=1.1)
ax[2,2].axvline(0,c='crimson',ls='--',lw=1.1); ax[3,3].axvline(0,c='k',ls=':',lw=1.1)
plt.savefig('./fig_pp_curved.pdf',bbox_inches='tight'); plt.savefig('./fig_pp_curved.png',dpi=130,bbox_inches='tight'); plt.close()
# ---------------- Fig: nuisance offset M explicit + overlay of analytic marginalisation ----------------
c4=ld('wcdmM'); s4=c4['post'].reshape(-1,3)
out['wcdmM']={'Om':stats(s4[:,0]),'w':stats(s4[:,1]),'M':stats(s4[:,2]),'corr':np.corrcoef(s4.T).tolist(),'Rhat':list(map(float,c4['R'])),'acc':list(map(float,c4['acc']))}
fig=corner.corner(s4,labels=[r"$\Omega_m$",r"$w$",r"$\mathcal{M}$ (offset)"],levels=LV,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt=".3f",bins=45,smooth=1.0,
                  plot_datapoints=False,color='C0',fig=plt.figure(figsize=(6.8,6.8)),range=[(0,0.7),(-1.6,-0.4),(-19.39,-19.31)],
                  title_kwargs={'fontsize':8},hist_kwargs={'lw':1.3},contour_kwargs={'linewidths':1.3})
ax=np.array(fig.axes).reshape(3,3)
# overlay analytic marginalisation (red) on the (Om,w) panel and 1D marginals
corner.hist2d(s2[:,0],s2[:,1],ax=ax[1,0],levels=LV,bins=50,smooth=1.0,plot_datapoints=False,plot_density=False,color='crimson',
              contour_kwargs={'linewidths':1.1,'linestyles':'--'},range=[(0,0.7),(-1.6,-0.4)])
for j,rng_ in [(0,(0,0.7)),(1,(-1.6,-0.4))]:
    h,e=np.histogram(s2[:,j],bins=50,range=rng_,density=True); ctr=0.5*(e[1:]+e[:-1])
    a=ax[j,j]; sc=a.get_ylim()[1]/ (np.histogram(s4[:,j],bins=45,range=rng_,density=True)[0].max()*1.05)
    a.plot(ctr,h*sc/ (1.0) if False else h/ (np.histogram(s4[:,j],bins=45,range=rng_,density=True)[0].max()/ (a.get_ylim()[1]/1.05)) ,c='crimson',ls='--',lw=1.0)
plt.savefig('./fig_pp_nuisance.pdf',bbox_inches='tight'); plt.savefig('./fig_pp_nuisance.png',dpi=130,bbox_inches='tight'); plt.close()
json.dump(out,open('./results.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
