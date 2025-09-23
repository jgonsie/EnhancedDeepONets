#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:13:40 2025

@author: jgonzalez
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
import pandas as pd

from SRC.utils import fix_random, fix_precission
from SRC.utils import prepare_raw_data_items_esp, prepare_raw_data_items, generate_loss_weights
from SRC.models import DeepONet, NeuralOperatorModel
from SRC import foamRW as fRW
from SRC.grids2D import Grid
from SRC import postprocessing as pp

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.ticker import FormatStrFormatter
plt.rc('axes', axisbelow=True)
# matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

width = 6.6

if __name__ == "__main__":
    
    fix_random(1234)
    dtype = fix_precission(double_precission = True)

    # Create grid
    grid = Grid(size_x = 50, size_y = 50, step_size = 1/50)

    # Data for training and validation
    data_route = '../OpenFOAM/experiment3/training_data/'
    training_data = fRW.upload_training_data(data_route, experiment='exp3', dtype=dtype)
    

    
#%% Comparative data

    def compute_errors(result, y_val):
        err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
        grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
        
        return tf.reduce_mean(err_u).numpy(), tf.reduce_mean(err_gradu).numpy()

    seeds =  [4, 42, 66, 101, 120, 353, 666, 882, 965, 1234]
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=200)
    
    x_test = x_val
    y_test = y_val
    
    # Create learning model
    net = DeepONet(layers_branch=[120,120,120,120,100], layers_trunk=[100,100,100,100], 
                   experiment='exp3', dimension='2D', seed=42, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model   
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 
        
    err_u_van_GD = []
    err_u_van_GD_LS = []
    err_u_enh_GD = []
    err_u_enh_GD_LS = []
    err_gradu_van_GD = []
    err_gradu_van_GD_LS = []
    err_gradu_enh_GD = []
    err_gradu_enh_GD_LS = []

    for s in seeds:
        #Load weights and history
        sim_name = f'H1_LS-False_seed-{s}_200'
        model.load_weights(f'../results/exp3/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model.predict(x_test, batch_size=36)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_van_GD.append(err_u)
        err_gradu_van_GD.append(err_gradu)
        print('Done')
        #Load weights and history
        sim_name = f'H1_LS-True_reg-10.0_seed-{s}_200'
        model.load_weights(f'../results/exp3/2.regularizer_init/{sim_name}_best.weights.h5')
        
        result = model.predict(x_test, batch_size=36)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_van_GD_LS.append(err_u)
        err_gradu_van_GD_LS.append(err_gradu)
        print('Done')
        #Load weights and history
        sim_name = f'H1+der_LS-False_seed-{s}_200'
        model.load_weights(f'../results/exp3/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model.predict(x_test, batch_size=36)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_enh_GD.append(err_u)
        err_gradu_enh_GD.append(err_gradu)
        print('Done')
        #Load weights and history
        sim_name = f'H1+der_LS-True_reg-10.0_seed-{s}_200'
        model.load_weights(f'../results/exp3/2.regularizer_init/{sim_name}_best.weights.h5')
        
        result = model.predict(x_test, batch_size=36)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_enh_GD_LS.append(err_u)
        err_gradu_enh_GD_LS.append(err_gradu)
        print('Done')
        
    mean_u_van_GD = np.mean(np.array(err_u_van_GD))
    mean_u_van_GD_LS = np.mean(np.array(err_u_van_GD_LS))
    mean_u_enh_GD = np.mean(np.array(err_u_enh_GD))
    mean_u_enh_GD_LS = np.mean(np.array(err_u_enh_GD_LS))
    
    std_u_van_GD = np.std(np.array(err_u_van_GD), ddof=1)
    std_u_van_GD_LS = np.std(np.array(err_u_van_GD_LS), ddof=1)
    std_u_enh_GD = np.std(np.array(err_u_enh_GD), ddof=1)
    std_u_enh_GD_LS = np.std(np.array(err_u_enh_GD_LS), ddof=1)
    
    mean_gradu_van_GD = np.mean(np.array(err_gradu_van_GD))
    mean_gradu_van_GD_LS = np.mean(np.array(err_gradu_van_GD_LS))
    mean_gradu_enh_GD = np.mean(np.array(err_gradu_enh_GD))
    mean_gradu_enh_GD_LS = np.mean(np.array(err_gradu_enh_GD_LS))
    
    std_gradu_van_GD = np.std(np.array(err_gradu_van_GD), ddof=1)
    std_gradu_van_GD_LS = np.std(np.array(err_gradu_van_GD_LS), ddof=1)
    std_gradu_enh_GD = np.std(np.array(err_gradu_enh_GD), ddof=1)
    std_gradu_enh_GD_LS = np.std(np.array(err_gradu_enh_GD_LS), ddof=1)
    
    print(f'u VAN-GD: {mean_u_van_GD:.2f} +- {std_u_van_GD:.2f}')
    print(f'gradu VAN-GD: {mean_gradu_van_GD:.2f} +- {std_gradu_van_GD:.2f}')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(f'u ENH-GD: {mean_u_enh_GD:.2f} +- {std_u_enh_GD:.2f}')
    print(f'gradu ENH-GD: {mean_gradu_enh_GD:.2f} +- {std_gradu_enh_GD:.2f}')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(f'u VAN-GD-LS: {mean_u_van_GD_LS:.2f} +- {std_u_van_GD_LS:.2f}')
    print(f'gradu VAN-GD_LS: {mean_gradu_van_GD_LS:.2f} +- {std_gradu_van_GD_LS:.2f}')
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(f'u ENH-GD_LS: {mean_u_enh_GD_LS:.2f} +- {std_u_enh_GD_LS:.2f}')
    print(f'gradu ENH-GD_LS: {mean_gradu_enh_GD_LS:.2f} +- {std_gradu_enh_GD_LS:.2f}')
    
    
#%%
    def plotter3(x_train, x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS):
        
        def compute_errors(result, y_val):
            err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
            grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
            grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
            err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
            
            return err_u.numpy(), err_gradu.numpy()
        
        err_u_VANGD, err_gradu_VANGD = compute_errors(result_VANGD, y_val)
        err_u_ENHGD, err_gradu_ENHGD = compute_errors(result_ENHGD, y_val)
        err_u_VANLS, err_gradu_VANLS = compute_errors(result_VANLS, y_val)
        err_u_ENHLS, err_gradu_ENHLS = compute_errors(result_ENHLS, y_val)

        fig, ax = plt.subplots(2, 2, figsize=(width, width * 0.65))  
        
        ax[0][0].hist(err_u_VANGD[err_u_VANGD<75], bins=20, alpha=0.3, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        ax[0][1].hist(err_gradu_VANGD[err_gradu_VANGD<75], bins=20, alpha=0.3, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        ax[0][0].hist(err_u_VANLS[err_u_VANLS<75], bins=20, alpha=0.3, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
        ax[0][1].hist(err_gradu_VANLS[err_gradu_VANLS<75], bins=20, alpha=0.3, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
    
        ax[1][0].hist(err_u_ENHGD[err_u_ENHGD<75], bins=20, alpha=0.3, label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
        ax[1][0].hist(err_u_ENHLS[err_u_ENHLS<75], bins=20, alpha=0.3, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')
        ax[1][1].hist(err_gradu_ENHGD[err_gradu_ENHGD<75], bins=20, alpha=0.3, label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
        ax[1][1].hist(err_gradu_ENHLS[err_gradu_ENHLS<75], bins=20, alpha=0.3, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')


        ax[1][0].set_xlabel(r'$u_{\theta}$ rel. $\ell_2$-error (\%)')
        ax[1][1].set_xlabel(r'$\nabla u_{\theta}$ rel. $\ell_2$-error (\%)')
        ax[0][0].set_ylabel(r'\# Occurences')
        ax[1][1].set_ylabel(r'\# Occurences')
        ax[0][1].set_ylabel(r'\# Occurences')
        ax[1][0].set_ylabel(r'\# Occurences')

        ax[0][0].set_xticks([0,25,50,75])
        ax[0][1].set_xticks([0,25,50,75])
        ax[1][0].set_xticks([0,25,50,75])
        ax[1][1].set_xticks([0,25,50,75])

        ax[0][0].legend()
        ax[1][0].legend()
        ax[0][1].legend()
        ax[1][1].legend()
         
        fig.tight_layout()
        
        return fig
    
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=200)
    
    # Create learning model
    net = DeepONet(layers_branch=[120,120,120,120,100], layers_trunk=[100,100,100,100], 
                   experiment='exp3', dimension='2D', seed=42, dtypeid=dtype) 
    loss_weights = generate_loss_weights(y_train)
    
    # Select model   
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 
    
    #Load weights and history
    sim_name = 'H1_LS-False_seed-42_200'
    model.load_weights(f'../results/exp3/1.training_points_init/{sim_name}_best.weights.h5')
    result_VANGD = model.predict(x_val, batch_size=36)
    
    sim_name = 'H1+der_LS-False_seed-42_200'
    model.load_weights(f'../results/exp3/1.training_points_init/{sim_name}_best.weights.h5')
    result_ENHGD = model.predict(x_val, batch_size=36)

    sim_name = 'H1_LS-True_reg-10.0_seed-42_200'
    model.load_weights(f'../results/exp3/2.regularizer_init/{sim_name}_best.weights.h5')
    result_VANLS = model.predict(x_val, batch_size=36)
    
    sim_name = 'H1+der_LS-True_reg-10.0_seed-42_200'
    model.load_weights(f'../results/exp3/2.regularizer_init/{sim_name}_best.weights.h5')
    result_ENHLS = model.predict(x_val, batch_size=36)
        
    fig301 = plotter3(x_train, x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS)
    # fig301.savefig('../results/exp3/figures/plot301.pdf', dpi=300, bbox_inches='tight')
