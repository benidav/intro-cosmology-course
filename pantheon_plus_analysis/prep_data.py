import numpy as np, pandas as pd
raw=np.loadtxt('cov_raw.txt'); n=int(raw[0]); C=raw[1:].reshape(n,n)
d=pd.read_csv('pp.dat',sep=r'\s+')
assert len(d)==n==1701
np.save('C.npy',C); d.to_pickle('d.pkl'); print("saved C.npy, d.pkl")
