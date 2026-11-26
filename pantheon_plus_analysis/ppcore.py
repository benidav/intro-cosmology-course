import numpy as np, pandas as pd
C_KMS=299792.458
H0FID=70.0
_raw=None

def load():
    d=pd.read_pickle('./d.pkl'); C=np.load('./C.npy')
    sel=(d.zHD.values>0.01)
    dd=d[sel].reset_index(drop=True); Cs=C[np.ix_(sel,sel)]
    Ci=np.linalg.inv(Cs)
    return dd,Cs,Ci

ZGRID=np.linspace(0.0,2.4,4801)   # h=5e-4

def chi_grid(Om,OL,w,Ork_free=True):
    """Return comoving-distance integral I(z)=int_0^z dz'/E on ZGRID, and E2 min; flat if OL is None"""
    z=ZGRID
    if OL is None:                 # flat wCDM: Ode=1-Om
        Ode=1-Om; E2=Om*(1+z)**3+Ode*(1+z)**(3*(1+w)); Ok=0.0
    else:                          # LCDM with curvature
        Ok=1-Om-OL; E2=Om*(1+z)**3+Ok*(1+z)**2+OL
    if E2.min()<=0: return None,None,Ok
    f=1/np.sqrt(E2)
    I=np.concatenate([[0],np.cumsum(0.5*(f[1:]+f[:-1])*np.diff(z))])
    return I,E2,Ok

def mu_model(zhd,zhel,Om,OL=None,w=-1.0):
    I,E2,Ok=chi_grid(Om,OL,w)
    if I is None: return None
    x=np.interp(zhd,ZGRID,I)
    if abs(Ok)<1e-8: dm=x
    elif Ok>0: dm=np.sinh(np.sqrt(Ok)*x)/np.sqrt(Ok)
    else: dm=np.sin(np.sqrt(-Ok)*x)/np.sqrt(-Ok)
    dl=(C_KMS/H0FID)*(1+zhel)*dm       # Mpc
    return 5*np.log10(dl)+25

class Like:
    def __init__(self):
        self.d,self.C,self.Ci=load()
        self.m=self.d.m_b_corr.values; self.zhd=self.d.zHD.values; self.zhel=self.d.zHEL.values
        self.one=np.ones(len(self.m)); self.Cii=self.Ci@self.one; self.S=self.one@self.Cii
    def chi2_marg(self,Om,OL=None,w=-1.0):
        mu=mu_model(self.zhd,self.zhel,Om,OL,w)
        if mu is None or not np.all(np.isfinite(mu)): return np.inf
        r=self.m-mu; v=self.Ci@r; A=r@v; B=v.sum()
        return A-B*B/self.S
    def chi2_M(self,Om,OL,w,M):
        mu=mu_model(self.zhd,self.zhel,Om,OL,w)
        if mu is None or not np.all(np.isfinite(mu)): return np.inf
        r=self.m-M-mu; return r@(self.Ci@r)
    def best_offset(self,Om,OL=None,w=-1.0):
        mu=mu_model(self.zhd,self.zhel,Om,OL,w); r=self.m-mu
        return (self.Cii@r)/self.S
