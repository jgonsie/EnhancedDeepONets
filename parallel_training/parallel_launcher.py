#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:04:42 2025

@author: jgonzalez
"""

import subprocess
from multiprocessing import Pool
from functools import partial

def run1(loss, LS, var):
    # cmd = f'python3 parallel_main_npoints.py --loss {loss} --LS {LS} --npoints {var}'
    # cmd = f'python3 parallel_main_regularizer.py --loss {loss} --LS {LS} --regularizer {var}'
    # cmd = f'python3 parallel_main_initializer.py --loss {loss} --LS {LS} --seed {var}'
    # cmd = f'python3 parallel_main_regulinit.py --loss {loss} --regularizer {LS} --seed {var}'
    cmd = f'python3 parallel_main_npointsinit_exp3.py --loss {loss} --npoints {LS} --seed {var}'
    
    subprocess.run(cmd, shell=True)
    
def run2(loss, LS, var):
    # cmd = f'python3 parallel_main_npoints.py --loss {loss} --LS {LS} --npoints {var}'
    # cmd = f'python3 parallel_main_regularizer.py --loss {loss} --LS {LS} --regularizer {var}'
    # cmd = f'python3 parallel_main_initializer.py --loss {loss} --LS {LS} --seed {var}'
    # cmd = f'python3 parallel_main_regulinit_exp2.py --loss {loss} --regularizer {LS} --seed {var}'
    # cmd = f'python3 parallel_main_npointsinit_exp2.py --loss {loss} --npoints {LS} --seed {var}'
    cmd = f'python3 parallel_main_regulinit_exp3.py --loss {loss} --regularizer {LS} --seed {var}'
    
    subprocess.run(cmd, shell=True)
    
if __name__ == "__main__":
    # loss = 'H1+der'
    # LS = True
    # # varss = [3, 4, 5, 6]
    # # varss = [10**-2,10**-1, 10**0, 10**1]
    # # varss = [0.05, 0.5, 5]
    # varss = [42, 120, 4, 882]
    
    # run_with_fixed_args = partial(run, loss, LS)
    
    # with Pool(processes=len(varss)) as pool:
    #     pool.map(run_with_fixed_args, varss)
        
        
    # for r in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:    
    #     loss = 'H1+der'
    #     LS = r
    #     varss = [1234, 66, 353, 101]
    #     run_with_fixed_args = partial(run, loss, LS)
            
    #     with Pool(processes=len(varss)) as pool:
    #         pool.map(run_with_fixed_args, varss)
            
    #     loss = 'H1+der'
    #     LS = r
    #     varss = [42, 120, 4, 882]
    #     run_with_fixed_args = partial(run, loss, LS)
            
    #     with Pool(processes=len(varss)) as pool:
    #         pool.map(run_with_fixed_args, varss)
        
    # for r in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:    
    #     loss = 'H1'
    #     LS = r
    #     varss = [1234, 66, 353, 101]
    #     run_with_fixed_args = partial(run, loss, LS)
            
    #     with Pool(processes=len(varss)) as pool:
    #         pool.map(run_with_fixed_args, varss)
            
    #     loss = 'H1'
    #     LS = r
    #     varss = [42, 120, 4, 882]
    #     run_with_fixed_args = partial(run, loss, LS)
            
    #     with Pool(processes=len(varss)) as pool:
    #         pool.map(run_with_fixed_args, varss)    
        
    # for r in [4]:    
        # loss = 'H1+der'
        # LS = r
        # varss = [1234, 66, 353, 101]
        # run_with_fixed_args = partial(run1, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
            
        # loss = 'H1+der'
        # LS = r
        # varss = [42, 120, 4, 882]
        # run_with_fixed_args = partial(run1, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
           
        # loss = 'H1'
        # LS = r
        # varss = [1234, 66, 353, 101]
        # run_with_fixed_args = partial(run1, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
            
        # loss = 'H1'
        # LS = r
        # varss = [42, 120, 4, 882]
        # run_with_fixed_args = partial(run1, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss) 
            
        # loss = 'H1+der'
        # LS = 10.0
        # varss = [1234, 66, 353, 101]
        # run_with_fixed_args = partial(run2, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
            
        # loss = 'H1'
        # LS = 5.0
        # varss = [1234, 66, 353, 101]
        # run_with_fixed_args = partial(run2, loss, LS)
                
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)

        # loss = 'H1+der'
        # LS = 10.0
        # varss = [42, 120, 4, 882]
        # run_with_fixed_args = partial(run2, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
            
        # loss = 'H1'
        # LS = 5.0
        # varss = [42, 120, 4, 882]
        # run_with_fixed_args = partial(run2, loss, LS)
                
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
        
        # loss = 'H1+der'
        # LS = 20.0
        # varss = [1234, 66, 353, 101]
        # run_with_fixed_args = partial(run2, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)


        # loss = 'H1+der'
        # LS = 20.0
        # varss = [42, 120, 4, 882]
        # run_with_fixed_args = partial(run2, loss, LS)
            
        # with Pool(processes=len(varss)) as pool:
        #     pool.map(run_with_fixed_args, varss)
    

    
    loss = 'H1'
    LS = 200
    varss = [120]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 200
    varss = [4]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
    
    loss = 'H1'
    LS = 200
    varss = [882]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 200
    varss = [666]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 200
    varss = [965]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)


    loss = 'H1+der'
    LS = 200
    varss = [120]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 200
    varss = [4]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 200
    varss = [4]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 200
    varss = [882]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 200
    varss = [666]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 200
    varss = [965]
    run_with_fixed_args = partial(run1, loss, LS)
    
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    # loss = 'H1'
    # LS = 200
    # varss = [42]
    # run_with_fixed_args = partial(run1, loss, LS)
    
    # with Pool(processes=len(varss)) as pool:
    #     pool.map(run_with_fixed_args, varss)
    
    # loss = 'H1+der'
    # LS = 200
    # varss = [42]
    # run_with_fixed_args = partial(run1, loss, LS)
    
    # with Pool(processes=len(varss)) as pool:
    #     pool.map(run_with_fixed_args, varss)
    
    
    loss = 'H1'
    LS = 10.0
    varss = [120]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 10.0
    varss = [4]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 10.0
    varss = [882]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1'
    LS = 10.0
    varss = [666]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)

    loss = 'H1'
    LS = 10.0
    varss = [965]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 10.0
    varss = [120]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 10.0
    varss = [4]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 10.0
    varss = [882]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)
        
    loss = 'H1+der'
    LS = 10.0
    varss = [666]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)

    loss = 'H1+der'
    LS = 10.0
    varss = [965]
    run_with_fixed_args = partial(run2, loss, LS)
        
    with Pool(processes=len(varss)) as pool:
        pool.map(run_with_fixed_args, varss)

