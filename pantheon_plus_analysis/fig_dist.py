import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dist import *
plt.rcParams.update({'font.size':9,'axes.linewidth':0.8})
STY={'flat $\\Lambda$CDM ($\\Omega_m=0.3$)':dict(c='C0',ls='-',lw=1.8),
     'Einstein--de Sitter':dict(c='C3',ls='--',lw=1.4),
     'open, $\\Omega_m=0.3$, $\\Omega_\\Lambda=0$':dict(c='C2',ls='-.',lw=1.4),
     'Milne (empty)':dict(c='k',ls=':',lw=1.4),
     'de Sitter (flat, $\\Lambda$ only)':dict(c='C1',ls=(0,(5,1,1,1,1,1)),lw=1.4)}
vec=lambda f,zs,Om,OL:np.array([f(z,Om,OL) for z in zs])
# ============ Fig A: the four distances ============
zs=np.linspace(0.005,3.0,400); Om,OL=0.3,0.7
dm=vec(DM,zs,Om,OL); dl=vec(dL,zs,Om,OL); da=vec(dA,zs,Om,OL); dh=vec(DHz,zs,Om,OL)
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,3.9))
a1.semilogy(zs,dm/1e3,c='C0',lw=1.8,label=r"$D_M$ (= $D_C$, flat)")
a1.semilogy(zs,dl/1e3,c='C3',lw=1.8,label=r"$d_L=(1+z)D_M$")
a1.semilogy(zs,da/1e3,c='C2',lw=1.8,label=r"$d_A=D_M/(1+z)$")
a1.semilogy(zs,dh/1e3,c='0.4',lw=1.4,ls='--',label=r"$D_H(z)=c/H(z)$")
zm=1.605; a1.plot([zm],[dA(zm,Om,OL)/1e3],'o',c='C2',ms=5); a1.annotate(r"max $d_A$ at $z\simeq1.6$",xy=(zm,dA(zm,Om,OL)/1e3),xytext=(1.75,0.28),fontsize=8,
          arrowprops=dict(arrowstyle='->',lw=0.8))
a1.set_xlabel(r"$z$"); a1.set_ylabel("distance [Gpc]  ($H_0=70$)"); a1.set_ylim(0.02,60); a1.legend(fontsize=8,loc='upper left'); a1.set_title(r"flat $\Lambda$CDM, $\Omega_m=0.3$",fontsize=9)
zl=np.geomspace(0.003,3.0,300); x=DH0*zl
r_L=vec(dL,zl,Om,OL)/x; r_M=vec(DM,zl,Om,OL)/x; r_A=vec(dA,zl,Om,OL)/x
q0=0.5*Om-OL
a2.semilogx(zl,r_L,c='C3',lw=1.8,label=r"$d_L$"); a2.semilogx(zl,r_M,c='C0',lw=1.8,label=r"$D_M$"); a2.semilogx(zl,r_A,c='C2',lw=1.8,label=r"$d_A$")
a2.semilogx(zl,1+0.5*(1-q0)*zl,c='C3',ls=':',lw=1.2,label=r"$1+\frac{1-q_0}{2}z$ (2nd order)")
a2.axhline(1,c='0.6',lw=0.8); a2.set_xlabel(r"$z$"); a2.set_ylabel("distance / (c z / H$_0$)"); a2.set_ylim(0,1.8)
a2.legend(fontsize=8,loc='upper left'); a2.set_title("all tend to the Hubble law as $z\\to0$",fontsize=9)
plt.tight_layout(); plt.savefig('fig_dist_four.pdf'); plt.savefig('fig_dist_four.png',dpi=130); plt.close()
# ============ Fig B: model zoo ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,3.9))
zs=np.linspace(0.01,5,300); zs2=np.geomspace(0.02,2.0,160)
dl_mil=vec(dL,zs2,0,0)
for nm,(Om_,OL_) in MODELS.items():
    a1.plot(zs,vec(dA,zs,Om_,OL_)/1e3,label=nm,**STY[nm])
    a2.plot(zs2,5*np.log10(vec(dL,zs2,Om_,OL_)/dl_mil),label=nm,**STY[nm])
for nm in ['flat $\\Lambda$CDM ($\\Omega_m=0.3$)','Einstein--de Sitter','open, $\\Omega_m=0.3$, $\\Omega_\\Lambda=0$']:
    Om_,OL_=MODELS[nm]; r=minimize_scalar(lambda z:-dA(z,Om_,OL_),bounds=(0.2,6),method='bounded')
    a1.plot([r.x],[-r.fun/1e3],'o',c=STY[nm]['c'],ms=4.5)
a1.set_xlabel(r"$z$"); a1.set_ylabel(r"$d_A$ [Gpc]"); a1.set_ylim(0,4.2); a1.legend(fontsize=7,loc='upper left',framealpha=0.9)
a2.set_xscale('log'); a2.axhline(0,c='k',lw=0.6); a2.set_xlabel(r"$z$"); a2.set_ylabel(r"$\mu-\mu_{\rm Milne}$ [mag]"); 
a2.set_title("more acceleration $\\Rightarrow$ fainter sources",fontsize=9)
a1.set_title("angular-diameter distance",fontsize=9)
plt.tight_layout(); plt.savefig('fig_dist_models.pdf'); plt.savefig('fig_dist_models.png',dpi=130); plt.close()
# ============ Fig C: angular size of rulers ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,3.9)); zs=np.geomspace(0.05,6,260)
three=['flat $\\Lambda$CDM ($\\Omega_m=0.3$)','Einstein--de Sitter','open, $\\Omega_m=0.3$, $\\Omega_\\Lambda=0$']
for nm in three:
    Om_,OL_=MODELS[nm]
    th=0.010/vec(dA,zs,Om_,OL_)*206265
    a1.plot(zs,th,label=nm,**STY[nm])
    r=minimize_scalar(lambda z:-dA(z,Om_,OL_),bounds=(0.2,6),method='bounded'); a1.plot([r.x],[0.010/(-r.fun)*206265],'o',c=STY[nm]['c'],ms=4.5)
    a2.plot(zs,np.degrees(147.0/vec(DM,zs,Om_,OL_)),label=nm,**STY[nm])
a1.set_xscale('log'); a1.set_yscale('log'); a2.set_xscale('log'); a2.set_yscale('log')
a1.set_xlabel(r"$z$"); a1.set_ylabel(r"$\theta=\ell/d_A$ [arcsec],  $\ell=10$ kpc (proper)"); a1.legend(fontsize=7.5)
a1.set_title("fixed proper ruler: minimum, then grows",fontsize=9)
a2.set_xlabel(r"$z$"); a2.set_ylabel(r"$\theta_{\rm BAO}=r_d/D_M$ [deg],  $r_d=147$ Mpc (comoving)"); a2.set_title("fixed comoving ruler: always decreasing",fontsize=9)
plt.tight_layout(); plt.savefig('fig_dist_angular.pdf'); plt.savefig('fig_dist_angular.png',dpi=130); plt.close()
# ============ Fig D: curvature ============
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.6,3.9)); x=np.linspace(0,3.2,300)
for Ok,c,ls,lab in [(0.5,'C2','-.',r"$\Omega_k=+0.5$ (open): $\sinh$"),(0,'C0','-',r"$\Omega_k=0$ (flat): linear"),(-0.5,'C3','--',r"$\Omega_k=-0.5$ (closed): $\sin$")]:
    if Ok==0: y=x
    elif Ok>0: y=np.sinh(np.sqrt(Ok)*x)/np.sqrt(Ok)
    else: y=np.sin(np.sqrt(-Ok)*x)/np.sqrt(-Ok)
    a1.plot(x,y,c=c,ls=ls,lw=1.7,label=lab)
a1.annotate("closed: $D_M$ peaks at\n$\\sqrt{|\\Omega_k|}D_C/D_H=\\pi/2$, then falls",xy=(np.pi/(2*np.sqrt(0.5)),1.414),xytext=(1.55,0.28),fontsize=7.5,arrowprops=dict(arrowstyle='->',lw=0.8))
a1.set_ylim(0,4); a1.set_xlabel(r"$D_C/D_H$"); a1.set_ylabel(r"$D_M/D_H$"); a1.legend(fontsize=8,loc='upper left'); a1.set_title("geometry: transverse vs. line-of-sight comoving distance",fontsize=9)
zs=np.linspace(0.02,3.0,120); ref=vec(DM,zs,0.3,0.7)
for Ok,c,ls in [(-0.10,'C3','--'),(-0.05,'C3',':'),(0.05,'C2',':'),(0.10,'C2','--')]:
    a2.plot(zs,100*(vec(DM,zs,0.3,0.7-Ok)/ref-1),c=c,ls=ls,lw=1.6,label=r"$\Omega_k=%+.2f$"%Ok)
a2.axhline(0,c='C0',lw=1.4); a2.set_xlabel(r"$z$"); a2.set_ylabel(r"$\Delta D_M/D_M$ vs flat [%]"); a2.legend(fontsize=8,ncol=2); a2.set_title(r"fixed $\Omega_m=0.3$, $\Omega_\Lambda=1-\Omega_m-\Omega_k$",fontsize=9)
plt.tight_layout(); plt.savefig('fig_dist_curv.pdf'); plt.savefig('fig_dist_curv.png',dpi=130); plt.close()
print("ok")
