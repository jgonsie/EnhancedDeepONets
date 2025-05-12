#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:04:42 2025

@author: jgonzalez
"""

import subprocess
from multiprocessing import Pool
from functools import partial

def run(loss, LS, var):
    # cmd = f'python3 parallel_main.py --loss {loss} --LS {LS} --seed {var}'
    cmd = f'python3 parallel_main_npoints.py --loss {loss} --LS {LS} --npoints {var}'
    subprocess.run(cmd, shell=True)
    
if __name__ == "__main__":
    loss = 'H1'
    LS = False
    npoints = [6, 7, 8, 9]
    # seeds = [42, 120, 4, 882, 1134]
    
    run_with_fixed_args = partial(run, loss, LS)
    
    with Pool(processes=len(npoints)) as pool:
        pool.map(run_with_fixed_args, npoints)