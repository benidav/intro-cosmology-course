# Pantheon+ MCMC pipeline (lecture 2)

Script version of the analysis in `notebooks/02_mcmc_pantheon.ipynb`, using a hand-written Metropolis–Hastings sampler instead of `emcee`.

```bash
sh download_data.sh                           # official Pantheon+ catalogue + STAT+SYS covariance (~34 MB) -> d.pkl, C.npy
python3 run_mcmc.py flat wcdm curved wcdmM    # 4 chains per model; prints R-hat, tau, ESS  (~10 min, single CPU)
python3 fig_guide.py                          # synthetic "how to read a corner plot" figures
python3 fig_results.py                        # Pantheon+ figures + results.json
python3 fig_dist.py                           # distance figures (D_H, D_M, d_A, d_L)
```

- Cut: z_HD > 0.01 (1590 light curves). μ uses d_L = (1+z_hel) D_M(z_HD) with H0 = 70 fixed.
- The offset M (absolute magnitude + H0) is marginalised analytically, except in the `wcdmM` run where it is sampled.
- `figs/` holds the figures from the reference run; `results.json` holds the reference numbers.

Data: Scolnic et al. 2022, ApJ 938, 113; Brout et al. 2022, ApJ 938, 110.
