import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
C=299792.458; H0=70.0; DH0=C/H0
def Efun(Om,OL,Or=0.0):
    Ok=1-Om-OL-Or
    return lambda z: np.sqrt(Or*(1+z)**4+Om*(1+z)**3+Ok*(1+z)**2+OL)
def DC(z,Om,OL):
    E=Efun(Om,OL); return DH0*quad(lambda x:1/E(x),0,z,epsabs=1e-12,epsrel=1e-12)[0]
def DM(z,Om,OL):
    Ok=1-Om-OL; x=DC(z,Om,OL)/DH0
    if abs(Ok)<1e-10: return DH0*x
    if Ok>0: return DH0*np.sinh(np.sqrt(Ok)*x)/np.sqrt(Ok)
    return DH0*np.sin(np.sqrt(-Ok)*x)/np.sqrt(-Ok)
def dL(z,Om,OL): return (1+z)*DM(z,Om,OL)
def dA(z,Om,OL): return DM(z,Om,OL)/(1+z)
def DHz(z,Om,OL): return DH0/Efun(Om,OL)(z)
MODELS={'flat $\\Lambda$CDM ($\\Omega_m=0.3$)':(0.3,0.7),'Einstein--de Sitter':(1.0,0.0),'open, $\\Omega_m=0.3$, $\\Omega_\\Lambda=0$':(0.3,0.0),
        'Milne (empty)':(0.0,0.0),'de Sitter (flat, $\\Lambda$ only)':(0.0,1.0)}
if __name__=='__main__':
    print("D_H(0) =",DH0)
    for z in (0.1,0.5,1,2,3):
        Om,OL=0.3,0.7
        print(z,"DM %.1f dL %.1f dA %.1f DH(z) %.1f"%(DM(z,Om,OL),dL(z,Om,OL),dA(z,Om,OL),DHz(z,Om,OL)))
    for nm,(Om,OL) in MODELS.items():
        r=minimize_scalar(lambda z:-dA(z,Om,OL),bounds=(0.2,6),method='bounded',options={'xatol':1e-8})
        print(nm,"dA max at z=%.3f  dA_max=%.1f Mpc"%(r.x,-r.fun))
    # z=1 d_L
    for nm,(Om,OL) in MODELS.items(): print("z=1 dL",nm,round(dL(1,Om,OL),0), " mu-mu_Milne=",round(5*np.log10(dL(1,Om,OL)/dL(1,0,0)),3))
    # low-z expansion test (flat LCDM 0.3: q0=-0.55)
    q0=0.5*0.3-0.7
    for z in (0.05,0.1,0.3):
        ex=dL(z,0.3,0.7); ap=DH0*(z+0.5*(1-q0)*z**2); print("lowz z=%.2f exact %.1f approx %.1f  rel err %.4f; Hubble-law only err %.4f"%(z,ex,ap,ap/ex-1,DH0*z/ex-1))
    # BAO numbers at z=0.5
    z=0.5; Om,OL=0.3,0.7; D_M=DM(z,Om,OL); D_Hz=DHz(z,Om,OL); DV=(z*D_M**2*D_Hz)**(1/3)
    print("z=0.5: DM %.1f DH %.1f DV %.1f F_AP %.4f  theta_BAO(147)=%.4f rad = %.2f deg ; dz=%.4f ; Hz=%.2f"%(D_M,D_Hz,DV,D_M/D_Hz,147/D_M,np.degrees(147/D_M),147/D_Hz,C/D_Hz))
    # EdS checks
    z=1; print("EdS DM(z=1) formula",2*DH0*(1-1/np.sqrt(2)),"numeric",DM(1,1,0)," dA max 8/27 DH=",8/27*DH0)
    # flat 2-z distance example for lensing (zl=0.5, zs=1.5)
    zl,zs=0.5,1.5; dls=(DM(zs,.3,.7)-DM(zl,.3,.7))/(1+zs); print("d_A(zl=0.5,zs=1.5)=%.1f ; dA(zl)=%.1f dA(zs)=%.1f"%(dls,dA(zl,.3,.7),dA(zs,.3,.7)))
    # angular size of 10 kpc at z=1, 2 ; minimum
    for z in (0.1,0.5,1.0,1.6,2.0,3.0): print("theta(10kpc) z=%.1f: %.3f arcsec"%(z,0.010/dA(z,.3,.7)*206265))
    # curvature effect: D_M ratio at z=1 for Ok=+-0.1 (Om=0.3)
    for Ok in (-0.1,0.1):
        OL=1-0.3-Ok; print("Ok=%+.1f z=1 DM/DM_flat=%.4f ; z=1089? skip"%(Ok,DM(1,0.3,OL)/DM(1,0.3,0.7)))
