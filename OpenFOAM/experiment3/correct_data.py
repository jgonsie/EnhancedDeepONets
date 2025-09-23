# -*- coding: utf-8 -*-

import os
import numpy as np

def get_failed_simulations(directory, filetocheck = 'jacMu1(T)'):
    
    failed_DT1 = []
    failed_DT2 = []
    failed_V = []
    failed_dirs = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a folder
        if os.path.isdir(item_path):
            # Create a list with the files inside the folder (not subdirectories)
            files = [f for f in os.listdir(item_path)
                     if os.path.isfile(os.path.join(item_path,f))]
            
            if filetocheck not in files:
                failed_DT1.append(float(item.split('-')[0].split('_')[1]))
                failed_DT2.append(float(item.split('-')[1].split('_')[1]))
                failed_V.append(float(item.split('-')[2].split('_')[1]))
                failed_dirs.append(item)
            
    return np.array(failed_dirs), np.array(failed_DT1), np.array(failed_DT2), np.array(failed_V)

if __name__ == "__main__":
    # Get the names of the failed simulations
    failed_dirs, failed_DT1, failed_DT2, failed_V = get_failed_simulations('training_data')
    
    # Save the failed dirs into a file
    np.savetxt('exp3_corr_dirs.dat', failed_dirs, fmt='%s')
    
    # Perturbate failed simulation
    failed_sims_corr = failed_DT1 + failed_DT1*0.001
    np.savetxt('exp3_corr_randomDT1.dat', failed_sims_corr, fmt='%.9f')
    np.savetxt('exp3_corr_randomDT2.dat', failed_DT2, fmt='%.9f')
    np.savetxt('exp3_corr_randomV.dat', failed_V, fmt='%.9f')
    
    # Correct original file
    original_DT = np.loadtxt('exp3_randomDT1.dat')
    for i in range(len(failed_DT1)):  
        index = np.where(original_DT == failed_DT1[i])[0]
        original_DT[index] = failed_sims_corr[i]
    np.savetxt('exp3_randomDT1.dat', original_DT, fmt='%.9f')