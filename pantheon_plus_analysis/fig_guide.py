import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, corner
plt.rcParams.update({'font.size':9,'axes.linewidth':0.8})
rng=np.random.default_rng(7)
LV=(0.68,0.95)
# ---------- Fig 1: gallery of 6 cases ----------
N=30000
def gauss(rho): 
    c=np.array([[1,rho],[rho,1]]); return rng.multivariate_normal([0,0],c,N)
cases=[]
cases.append(("(a) uncorrelated",gauss(0.0)))
cases.append(("(b) positive correlation",gauss(0.85)))
cases.append(("(c) negative correlation",gauss(-0.85)))
x=rng.standard_normal(N); y=0.55*(x**2-1)+0.22*rng.standard_normal(N)
cases.append(("(d) banana: dependent, but "+r"$\rho\approx0$",np.c_[x,y]))
# (e) degenerate ridge: only a combination is measured; theta_1 fills its whole flat prior range
xs=rng.random(N); ys=-1.5*xs-0.2+0.05*rng.standard_normal(N)
ridge=np.c_[xs,ys]
cases.append((r"(e) degeneracy: $\theta_1$ unconstrained",ridge))
b=np.r_[rng.multivariate_normal([-1.5,-1.2],[[.15,0],[0,.15]],N//2),rng.multivariate_normal([1.5,1.2],[[.15,0],[0,.15]],N//2)]
cases.append(("(f) bimodal",b))
fig,axs=plt.subplots(2,3,figsize=(9.2,6.0))
for ax,(t,s) in zip(axs.ravel(),cases):
    corner.hist2d(s[:,0],s[:,1],ax=ax,levels=LV,bins=60,plot_datapoints=True,plot_density=False,
                  color='C0',data_kwargs={'alpha':0.04,'ms':1.2},contour_kwargs={'linewidths':1.2},fill_contours=False,smooth=1.0)
    r=np.corrcoef(s[:,0],s[:,1])[0,1]
    ax.set_title(t,fontsize=9); ax.text(0.04,0.93,r"$\rho=%+.2f$"%r,transform=ax.transAxes,va='top',fontsize=10,
                                     bbox=dict(fc='w',ec='0.6',boxstyle='round,pad=0.25'))
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
axs[1,1].set_xlim(-0.05,1.05); axs[1,1].axvline(0,c='0.6',lw=0.8,ls=':'); axs[1,1].axvline(1,c='0.6',lw=0.8,ls=':')
axs[1,1].text(0.5,0.06,"dotted = prior edges",transform=axs[1,1].transAxes,ha='center',fontsize=7.5,color='0.3')
plt.tight_layout(); plt.savefig('./fig_corner_gallery.pdf'); plt.savefig('./fig_corner_gallery.png',dpi=130); plt.close()

# ---------- Fig 2: anatomy of a 3-parameter corner (synthetic) ----------
cov=np.array([[1.0,-0.75,0.30],[-0.75,1.0,0.0],[0.30,0.0,1.0]])
s=rng.multivariate_normal([0,0,0],cov,60000)
lab=[r"$\theta_1$",r"$\theta_2$",r"$\theta_3$"]
fig=corner.corner(s,labels=lab,levels=LV,quantiles=[0.16,0.5,0.84],show_titles=True,title_fmt=".2f",
                  bins=50,smooth=1.0,plot_datapoints=False,color='C0',title_kwargs={'fontsize':8},
                  contour_kwargs={'linewidths':1.2},hist_kwargs={'lw':1.2},fig=plt.figure(figsize=(7.4,7.4)))
axs=np.array(fig.axes).reshape(3,3)
ov=fig.add_axes([0,0,1,1],facecolor='none',zorder=100); ov.axis('off'); ov.set_xlim(0,1); ov.set_ylim(0,1)
def fpt(ax,fx,fy):
    return fig.transFigure.inverted().transform(ax.transAxes.transform((fx,fy)))
def note(ax,txt,fxy,tpos):
    p=fpt(ax,*fxy)
    ov.annotate(txt,xy=p,xytext=tpos,fontsize=8,color='crimson',ha='left',va='center',
                arrowprops=dict(arrowstyle='->',lw=0.9,color='crimson'))
note(axs[0,0],"diagonal: 1D marginal posterior;\ndashed = median, 16%, 84%",(0.80,0.70),(0.50,0.90))
note(axs[1,0],"off-diagonal: joint 2D posterior;\ncontours enclose 68% and 95%",(0.80,0.55),(0.66,0.74))
note(axs[1,0],"tilted ellipse = correlation\n(here negative)",(0.30,0.72),(0.66,0.61))
note(axs[2,1],"round blob = no correlation",(0.75,0.65),(0.66,0.47))
plt.savefig('./fig_corner_anatomy.pdf'); plt.savefig('./fig_corner_anatomy.png',dpi=130); plt.close()
print("guide figs done")
