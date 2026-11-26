"""New figures for the lecture notes.

fig_rotcurve_nfw.pdf : synthetic galaxy rotation curve, fitted with
                       baryons + NFW, baryons + Burkert, and MOND (no DM).
fig_desi_bao.pdf     : DESI DR2 BAO distances relative to Planck LCDM,
                       and w(z) for the DESI DR2 w0wa fits.
fig_jeans.pdf        : linear growth of delta_k above / below the Jeans scale.

Run:  python3 scripts/make_new_figs.py   (from lecture_notes/)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad, solve_ivp

plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
G = 4.30091e-6          # kpc (km/s)^2 / Msun
OUT = "figs/"

# ----------------------------------------------------------------------
# 1. Rotation curve: exponential disk + halo
# ----------------------------------------------------------------------
from scipy.special import i0, i1, k0, k1


def v_disk(r, Md=5e10, Rd=3.0):
    """Freeman exponential disk."""
    y = r / (2 * Rd)
    Sig0 = Md / (2 * np.pi * Rd**2)
    return np.sqrt(4 * np.pi * G * Sig0 * Rd * y**2 * (i0(y) * k0(y) - i1(y) * k1(y)))


def m_nfw(r, rho_s, r_s):
    x = r / r_s
    return 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))


def m_bur(r, rho_0, r_0):
    x = r / r_0
    return np.pi * rho_0 * r_0**3 * (np.log((1 + x**2) * (1 + x)**2) - 2 * np.arctan(x))


def v_nfw_tot(r, lrho, r_s):
    return np.sqrt(v_disk(r)**2 + G * m_nfw(r, 10**lrho, r_s) / r)


def v_bur_tot(r, lrho, r_0):
    return np.sqrt(v_disk(r)**2 + G * m_bur(r, 10**lrho, r_0) / r)


def v_mond(r, a0=1.2e-10):
    # a0 in m/s^2 -> (km/s)^2/kpc : 1 (km/s)^2/kpc = 3.241e-14 m/s^2
    a0k = a0 / 3.241e-14
    gN = v_disk(r)**2 / r
    g = gN / 2 + np.sqrt(gN**2 / 4 + gN * a0k)      # "simple" interpolating fn
    return np.sqrt(g * r)


rng = np.random.default_rng(3)
r = np.linspace(1.0, 30.0, 22)
v_true = v_nfw_tot(r, np.log10(8e6), 18.0)           # truth: NFW, rho_s in Msun/kpc^3
err = 4.0 + 0.03 * v_true
v_obs = v_true + err * rng.normal(size=r.size)

pN, cN = curve_fit(v_nfw_tot, r, v_obs, p0=[7, 15], sigma=err, absolute_sigma=True)
pB, cB = curve_fit(v_bur_tot, r, v_obs, p0=[7.5, 5], sigma=err, absolute_sigma=True)
chi = lambda f, p: np.sum(((v_obs - f(r, *p)) / err)**2)
chiN, chiB = chi(v_nfw_tot, pN), chi(v_bur_tot, pB)
chiM = np.sum(((v_obs - v_mond(r)) / err)**2)
print(f"NFW: log rho_s={pN[0]:.2f}, r_s={pN[1]:.1f} kpc, chi2={chiN:.1f}")
print(f"Burkert: log rho_0={pB[0]:.2f}, r_0={pB[1]:.1f} kpc, chi2={chiB:.1f}")
print(f"MOND (a0 fixed, no free par): chi2={chiM:.1f}   dof={r.size}")

rr = np.linspace(0.3, 32, 300)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
a = ax[0]
a.errorbar(r, v_obs, err, fmt="o", color="k", ms=4, lw=1, label="data (synthetic)")
a.plot(rr, v_disk(rr), "C2--", label="stars+gas (Newton)")
a.plot(rr, np.sqrt(G * m_nfw(rr, 10**pN[0], pN[1]) / rr), "C0:", label="NFW halo")
a.plot(rr, v_nfw_tot(rr, *pN), "C0", lw=2, label=rf"disk+NFW, $\chi^2$={chiN:.0f}")
a.plot(rr, v_bur_tot(rr, *pB), "C1", lw=1.6, label=rf"disk+Burkert, $\chi^2$={chiB:.0f}")
a.plot(rr, v_mond(rr), "C3-.", lw=1.6, label=rf"MOND, $\chi^2$={chiM:.0f}")
rk = rr[rr > 8]
a.plot(rk, v_disk(8.0) * np.sqrt(8.0 / rk), color="gray", lw=1, ls=(0, (1, 2)))
a.text(26, v_disk(8.0) * np.sqrt(8 / 26) + 4, r"Kepler $\propto r^{-1/2}$", color="gray", fontsize=9)
a.set_xlabel("r [kpc]"); a.set_ylabel(r"$v_c$ [km/s]"); a.set_ylim(0, 260)
a.legend(fontsize=8.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
a.set_ylim(-60, 260); a.axhline(0, color="k", lw=0.5)

# density profiles
b = ax[1]
x = np.logspace(-2, 1.3, 300)
rho_nfw = 1 / (x * (1 + x)**2)
rho_bur = 1 / ((1 + x) * (1 + x**2))
rho_iso = 1 / (1 + x**2)
b.loglog(x, rho_nfw, "C0", lw=2, label=r"NFW: cusp $\rho\propto r^{-1}$")
b.loglog(x, rho_bur, "C1", lw=1.6, label=r"Burkert: core")
b.loglog(x, rho_iso, "C4--", lw=1.4, label=r"pseudo-isothermal: core, $r^{-2}$")
x1 = x[x < 0.1]; x3 = x[x > 3]
b.loglog(x1, 0.5 * x1**-1, color="gray", lw=1, ls=":")
b.loglog(x3, 2.5 * x3**-3, color="gray", lw=1, ls=":")
b.text(0.012, 15, r"$r^{-1}$", color="gray"); b.text(6, 0.03, r"$r^{-3}$", color="gray")
b.set_xlabel(r"$r/r_s$"); b.set_ylabel(r"$\rho/\rho_s$")
b.legend(fontsize=9, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "fig_rotcurve_nfw.pdf"); plt.close(fig)

# ----------------------------------------------------------------------
# 2. DESI DR2 BAO (arXiv:2503.14738, Table IV) vs Planck LCDM
# ----------------------------------------------------------------------
c = 299792.458
data = [  # name, z, DV, sDV, DM, sDM, DH, sDH
    ("BGS", 0.295, 7.942, 0.075, None, None, None, None),
    ("LRG1", 0.510, 12.720, 0.099, 13.588, 0.167, 21.863, 0.425),
    ("LRG2", 0.706, 16.050, 0.110, 17.351, 0.177, 19.455, 0.330),
    ("LRG3+ELG1", 0.934, 19.721, 0.091, 21.576, 0.152, 17.641, 0.193),
    ("ELG2", 1.321, 24.252, 0.174, 27.601, 0.318, 14.176, 0.221),
    ("QSO", 1.484, 26.055, 0.398, 30.512, 0.760, 12.817, 0.516),
    ("Lya", 2.330, 31.267, 0.256, 38.988, 0.531, 8.632, 0.101),
]


def dists(z, Om, H0rd, w0=-1, wa=0):
    E = lambda zz: np.sqrt(Om * (1 + zz)**3 + (1 - Om) * (1 + zz)**(3 * (1 + w0 + wa))
                           * np.exp(-3 * wa * zz / (1 + zz)))
    DM = c / H0rd * quad(lambda zz: 1 / E(zz), 0, z)[0]
    DH = c / H0rd / E(z)
    DV = (z * DM**2 * DH)**(1 / 3)
    return DM, DH, DV


planck = dict(Om=0.3153, H0rd=67.36 * 147.09)
desi = dict(Om=0.2975, H0rd=101.54 * 100)
zz = np.linspace(0.1, 2.5, 120)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
a = ax[0]
for j, (lab, mk, col) in enumerate([("DM", "o", "C0"), ("DH", "s", "C1"), ("DV", "^", "C2")]):
    ref = np.array([dists(z, **planck)[j] for z in zz])
    alt = np.array([dists(z, **desi)[j] for z in zz])
    a.plot(zz, alt / ref, color=col, lw=1.2)
    for d in data:
        idx = {"DV": (2, 3), "DM": (4, 5), "DH": (6, 7)}[lab]
        val, sig = d[idx[0]], d[idx[1]]
        if val is None or (lab == "DV" and d[0] not in ("BGS", "QSO")):
            continue
        refp = dists(d[1], **planck)[j]
        a.errorbar(d[1] + 0.02 * (j - 1), val / refp, sig / refp, fmt=mk, color=col, ms=5, lw=1)
a.axhline(1, color="k", lw=0.8)
a.plot([], [], "o", color="C0", label=r"$D_M/r_d$"); a.plot([], [], "s", color="C1", label=r"$D_H/r_d$")
a.plot([], [], "^", color="C2", label=r"$D_V/r_d$")
a.plot([], [], color="gray", label=r"DESI best-fit $\Lambda$CDM ($\Omega_m=0.298$)")
a.set_xlabel("z"); a.set_ylabel("DESI DR2 / Planck $\\Lambda$CDM")
a.set_ylim(0.93, 1.07); a.legend(fontsize=9, frameon=False, loc="upper left", ncol=2)

b = ax[1]
fits = [("DESI+CMB", -0.42, -1.75, "C3"), ("+Pantheon+", -0.838, -0.62, "C0"),
        ("+Union3", -0.667, -1.09, "C1"), ("+DESY5", -0.752, -0.86, "C2")]
zz2 = np.linspace(0, 2.5, 200); aa = 1 / (1 + zz2)
for lab, w0, wa, col in fits:
    b.plot(zz2, w0 + wa * (1 - aa), color=col, lw=1.8, label=rf"{lab}: $w_0={w0}$, $w_a={wa}$")
b.axhline(-1, color="k", lw=1, ls="--"); b.text(2.05, -0.97, r"$\Lambda$", fontsize=12)
b.fill_between(zz2, -3, -1, color="gray", alpha=0.08); b.text(1.4, -2.2, "phantom $w<-1$", color="gray")
b.set_ylim(-2.4, -0.2); b.set_xlabel("z"); b.set_ylabel(r"$w(z)=w_0+w_a\,z/(1+z)$")
b.legend(fontsize=8.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "fig_desi_bao.pdf"); plt.close(fig)

# ----------------------------------------------------------------------
# 3. Jeans: delta_k(a) in EdS for k/k_J = 0.5, 1, 2, 4  (k_J fixed in physical units)
# ----------------------------------------------------------------------
# EdS, time in units where H = 2/(3t); use ln a as variable:
# delta'' + (1/2) delta' - (3/2)(1 - (k/kJ)^2 a^{-1}) delta = 0   for c_s^2 ∝ 1/a (gas after decoupling: T∝a^-2)
# keep it simple: constant comoving Jeans ratio q = k/k_J
def rhs(lna, y, q):
    d, dp = y
    return [dp, -0.5 * dp + 1.5 * (1 - q**2) * d]


fig, a = plt.subplots(figsize=(5.8, 4.0))
lna = np.linspace(0, np.log(1000), 600)
for q, col in [(0.0, "k"), (0.7, "C0"), (1.0, "C2"), (2.0, "C1"), (4.0, "C3")]:
    sol = solve_ivp(rhs, (0, lna[-1]), [1.0, 1.0 if q < 1 else 0.0], t_eval=lna, args=(q,), rtol=1e-8)
    lab = r"$k\ll k_J$ (CDM)" if q == 0 else rf"$k/k_J={q}$"
    a.plot(np.exp(sol.t), sol.y[0], color=col, lw=1.6, label=lab)
a.set_xscale("log"); a.set_yscale("symlog", linthresh=1)
a.set_xlabel(r"$a/a_i$"); a.set_ylabel(r"$\delta_k/\delta_{k,i}$")
a.legend(fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig_jeans.pdf"); plt.close(fig)
print("done")
