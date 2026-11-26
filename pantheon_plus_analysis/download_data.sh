#!/bin/sh
# Downloads the public Pantheon+ catalogue and the STAT+SYS covariance matrix (official data release)
B="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
curl -L -o pp.dat      "$B/Pantheon%2BSH0ES.dat"
curl -L -o cov_raw.txt "$B/Pantheon%2BSH0ES_STAT%2BSYS.cov"
python3 prep_data.py
