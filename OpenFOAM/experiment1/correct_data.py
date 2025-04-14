# -*- coding: utf-8 -*-

import os
import numpy as np

def get_failed_simulations(directory, filetocheck = 'jacMu(T)'):
    
    failed_sims = []
    failed_dirs = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a folder
        if os.path.isdir(item_path):
            # Create a list with the files inside the folder (not subdirectories)
            files = [f for f in os.listdir(item_path)
                     if os.path.isfile(os.path.join(item_path,f))]
            
            if filetocheck not in files:
                failed_sims.append(float(item.split('-')[0].split('_')[1]))
                failed_dirs.append(item)
            
    return np.sort(np.array(failed_dirs)), np.sort(np.array(failed_sims))

if __name__ == "__main__":
    # Get the names of the failed simulations
    failed_dirs, failed_sims = get_failed_simulations('training_data')
    
    # Save the failed dirs into a file
    np.savetxt('exp1_corr_dirs.dat', failed_dirs, fmt='%s')
    
    # Perturbate failed simulation
    failed_sims_corr = failed_sims + 1e-9
    np.savetxt('exp1_corr_randomDT.dat', failed_sims_corr, fmt='%.9f')
    
    # Correct original file
    original_DT = np.loadtxt('exp1_randomDT.dat')
    for i in range(len(failed_sims)):  
        index = np.where(original_DT == failed_sims[i])[0]
        original_DT[index] = failed_sims_corr[i]
    np.savetxt('exp1_randomDT.dat', original_DT, fmt='%.9f')