#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:44:12 2025

@author: jgonzalez
"""

import numpy as np
from SRC.utils import DT_loguniform_dist

if __name__ == "__main__":
    
    # EXPERIMENT 1
    # Generate random samples following a log-normal distribution
    n_samples = 200
    exp1_samples = DT_loguniform_dist(n_samples, low_bound=10**-4, up_bound=15)
    np.savetxt('./OpenFOAM/experiment1/exp1_randomDT.dat', exp1_samples, fmt='%.9f')
    
    # EXPERIMENT 2
    n_samples = 200
    exp2_samples = DT_loguniform_dist(n_samples, low_bound=10**-4, up_bound=10)
    np.savetxt('./OpenFOAM/experiment2/exp2_randomDT.dat', exp2_samples, fmt='%.9f')
    
    # EXPERIMENT 3
    n_samples = 500
    exp3_samples_DT1 = DT_loguniform_dist(n_samples, low_bound=10**-3, up_bound=10)
    exp3_samples_DT2 = DT_loguniform_dist(n_samples, low_bound=10**-3, up_bound=10)
    exp3_samples_V = np.random.uniform(1,50,n_samples)
    np.random.shuffle(exp3_samples_DT1)
    np.random.shuffle(exp3_samples_DT2)
    np.savetxt('./OpenFOAM/experiment3/exp3_randomDT13.dat', exp3_samples_DT1, fmt='%.9f')
    np.savetxt('./OpenFOAM/experiment3/exp3_randomDT23.dat', exp3_samples_DT2, fmt='%.9f')
    np.savetxt('./OpenFOAM/experiment3/exp3_randomV3.dat', exp3_samples_V, fmt='%.9f')