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
    data_route = '../OpenFOAM/experiment2/training_data/'
    training_data = fRW.upload_training_data(data_route, experiment='exp2', dtype=dtype)
    


    #%% PLOTS 1.2 (training points and VAN vs ENH)
    
    def compute_errors_L2(result, y_val):
        err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
        grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
        
        return tf.reduce_mean(err_u).numpy(), tf.reduce_mean(err_gradu).numpy()

    def compute_mean_and_std(error_array):
        mean = np.mean(error_array)
        std = np.std(error_array, ddof=1)
        
        return mean, std
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=5)
    # Create learning model
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment = 'exp2', dimension='2D', seed=420, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model   
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 
    
    npoints = [4]
    seeds = [1234, 66, 353, 42, 120, 4,  101, 882] 
    
    mean_err_van_u = []
    mean_err_van_gradu = []
    mean_err_enh_u = []
    mean_err_enh_gradu = []
    std_err_van_u = []
    std_err_van_gradu = []
    std_err_enh_u = []
    std_err_enh_gradu = []
    
    for p in npoints:
        x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=p)
        
        van_errs_u = []
        van_errs_gradu = []
        enh_errs_u = []
        enh_errs_gradu = []

        for s in seeds:
            #Load weights and history
            sim_name = f'H1_LS-False_seed-{s}_{p}'
            model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
            result = model(x_val)
            err_u, err_gradu = compute_errors_L2(result, y_val) 
            van_errs_u.append(tf.reduce_mean(err_u).numpy())
            van_errs_gradu.append(tf.reduce_mean(err_gradu).numpy())

            #Load weights and history
            sim_name = f'H1+der_LS-False_seed-{s}_{p}'
            model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
            result = model(x_val)
            err_u, err_gradu = compute_errors_L2(result, y_val) 
            enh_errs_u.append(tf.reduce_mean(err_u).numpy())
            enh_errs_gradu.append(tf.reduce_mean(err_gradu).numpy()) 
            
        mean_van_u, std_van_u = compute_mean_and_std(van_errs_u)
        mean_van_gradu, std_van_gradu = compute_mean_and_std(van_errs_gradu) 
        mean_enh_u, std_enh_u = compute_mean_and_std(enh_errs_u)
        mean_enh_gradu, std_enh_gradu = compute_mean_and_std(enh_errs_gradu)
        mean_err_van_u.append(mean_van_u)
        mean_err_van_gradu.append(mean_van_gradu)
        mean_err_enh_u.append(mean_enh_u)
        mean_err_enh_gradu.append(mean_enh_gradu)
        std_err_van_u.append(std_van_u)
        std_err_van_gradu.append(std_van_gradu)
        std_err_enh_u.append(std_enh_u)
        std_err_enh_gradu.append(std_enh_gradu)
    
    fig121, ax = plt.subplots(1, 2, figsize=(width, width * 0.35))  
    ax[0].plot(npoints, mean_err_van_u, linestyle='--', color='tab:blue')
    ax[0].errorbar(npoints, mean_err_van_u, std_err_van_u, fmt='D', color='tab:blue',
                   ecolor='blue',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
    
    ax[0].plot(npoints, mean_err_enh_u, linestyle='--', color='tab:red')
    ax[0].errorbar(npoints, mean_err_enh_u, std_err_enh_u, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
    ax[0].set_xlabel(r'$\#$ training points')
    ax[0].set_ylabel(r'$u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[0].set_xticks([3,4,5,6,7,8,9])
    ax[0].set_yticks([0,30,60,90,120,150,180])
    ax[0].grid(True, which="both", linestyle=':', linewidth=0.5)
    ax[0].legend()
    
    ax[1].plot(npoints, mean_err_van_gradu, linestyle='--', color='tab:blue')
    ax[1].errorbar(npoints, mean_err_van_gradu, std_err_van_u, fmt='D', color='tab:blue',
                   ecolor='blue',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
    
    ax[1].plot(npoints, mean_err_enh_gradu, linestyle='--', color='tab:red')
    ax[1].errorbar(npoints, mean_err_enh_gradu, std_err_enh_gradu, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
    ax[1].set_xlabel(r'$\#$ training points')
    ax[1].set_ylabel(r'$\nabla u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[1].set_xticks([3,4,5,6,7,8,9])
    ax[1].set_yticks([0,30,60,90,120,150])
    ax[1].grid(True, which="both", linestyle=':', linewidth=0.5)
    ax[1].legend()
    plt.tight_layout()
    
    fig121.savefig('../results/exp1/figures/plot121.pdf', dpi=300, bbox_inches='tight')
    
#%% PLOTS 1.3 (GD vs GD-LS)   

    def compute_errors_L2(result, y_val):
        err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
        grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
        
        return tf.reduce_mean(err_u).numpy(), tf.reduce_mean(err_gradu).numpy()

    def compute_errors_H1(result, y_val):
        mod_grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        mod_grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        num = tf.square(result['T']-y_val['T']) + tf.square(mod_grad_ut-mod_grad_uh)
        den = tf.square(y_val['T']) + tf.square(mod_grad_uh)
        H1_err_rel = tf.sqrt(tf.reduce_sum(num,axis=1)) / tf.sqrt(tf.reduce_sum(den,axis=1)) * 100
            
        return tf.reduce_mean(H1_err_rel).numpy()
    
    
    tr_points = 4
    van_GD_histories = []
    van_GDLS_histories = []
    enh_GD_histories = []
    enh_GDLS_histories = []
    
    for s in seeds:
        sim_name = f'H1_LS-False_seed-{s}_{tr_points}'
        van_GD_histories.append(pd.read_csv(f'../results/exp2/1.training_points_init/{sim_name}_history.csv'))
        sim_name = f'H1+der_LS-False_seed-{s}_{tr_points}'
        enh_GD_histories.append(pd.read_csv(f'../results/exp2/1.training_points_init/{sim_name}_history.csv'))
        sim_name = f'H1_LS-True_reg-5.0_seed-{s}_{tr_points}'
        van_GDLS_histories.append(pd.read_csv(f'../results/exp2/2.regularizer_init/{sim_name}_history.csv'))
        sim_name = f'H1+der_LS-True_reg-5.0_seed-{s}_{tr_points}'
        enh_GDLS_histories.append(pd.read_csv(f'../results/exp2/2.regularizer_init/{sim_name}_history.csv'))


    van_GD_tr = np.array([i['loss'] for i in van_GD_histories])
    van_GD_val = np.array([i['val_loss'] for i in van_GD_histories])
    enh_GD_tr = np.array([i['loss'] for i in enh_GD_histories])
    enh_GD_val = np.array([i['val_loss'] for i in enh_GD_histories])
    van_GDLS_tr = np.array([i['loss'] for i in van_GDLS_histories])
    van_GDLS_val = np.array([i['val_loss'] for i in van_GDLS_histories])
    enh_GDLS_tr = np.array([i['loss'] for i in enh_GDLS_histories])
    enh_GDLS_val = np.array([i['val_loss'] for i in enh_GDLS_histories])
    
    mean_van_GD_tr = np.mean(van_GD_tr, axis=0)
    mean_van_GD_val = np.mean(van_GD_val, axis=0)
    mean_enh_GD_tr = np.mean(enh_GD_tr, axis=0)
    mean_enh_GD_val = np.mean(enh_GD_val, axis=0)
    std_van_GD_tr = np.std(van_GD_tr, axis=0, ddof=1)
    std_van_GD_val = np.std(van_GD_val, axis=0, ddof=1)
    std_enh_GD_tr = np.std(enh_GD_tr, axis=0, ddof=1)
    std_enh_GD_val = np.std(enh_GD_val, axis=0, ddof=1)
    mean_van_GDLS_tr = np.mean(van_GDLS_tr, axis=0)
    mean_van_GDLS_val = np.mean(van_GDLS_val, axis=0)
    mean_enh_GDLS_tr = np.mean(enh_GDLS_tr, axis=0)
    mean_enh_GDLS_val = np.mean(enh_GDLS_val, axis=0)
    std_van_GDLS_tr = np.std(van_GDLS_tr, axis=0, ddof=1)
    std_van_GDLS_val = np.std(van_GDLS_val, axis=0, ddof=1)
    std_enh_GDLS_tr = np.std(enh_GDLS_tr, axis=0, ddof=1)
    std_enh_GDLS_val = np.std(enh_GDLS_val, axis=0, ddof=1)
    
    fig131, ax = plt.subplots(2, 2, figsize=(width, width * 0.62))  
    ax[0][0].plot(mean_van_GD_tr, color='blue', zorder=2, label=r'$\mathtt{VAN}$-$\mathtt{GD}$: train')
    ax[0][0].plot(mean_van_GD_val, color='red', zorder=2, label=r'$\mathtt{VAN}$-$\mathtt{GD}$: val')
    ax[0][0].fill_between(np.arange(10000),mean_van_GD_tr+std_van_GD_tr, 
                    mean_van_GD_tr-std_van_GD_tr, color='b', alpha=0.1)
    ax[0][0].fill_between(np.arange(10000),mean_van_GD_val+std_van_GD_val, 
                    mean_van_GD_val-std_van_GD_val, color='r', alpha=0.1)
    ax[0][0].set_yscale("log")
    ax[0][0].grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel(r'Loss')
    ax[0][0].legend()
    
    ax[0][1].plot(mean_enh_GD_tr, color='blue', zorder=2, label=r'$\mathtt{ENH}$-$\mathtt{GD}$: train')
    ax[0][1].plot(mean_enh_GD_val, color='red', zorder=2, label=r'$\mathtt{ENH}$-$\mathtt{GD}$: val')
    ax[0][1].fill_between(np.arange(10000),mean_enh_GD_tr+std_enh_GD_tr, 
                    mean_enh_GD_tr-std_enh_GD_tr, color='b', alpha=0.1)
    ax[0][1].fill_between(np.arange(10000),mean_enh_GD_val+std_enh_GD_val, 
                    mean_enh_GD_val-std_enh_GD_val, color='r', alpha=0.1)
    ax[0][1].set_yscale("log")
    ax[0][1].grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel(r'Loss')
    ax[0][1].legend()
    
    lim=10000
    ax[1][0].plot(mean_van_GDLS_tr[:lim], color='blue', zorder=2, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$: train')
    ax[1][0].plot(mean_van_GDLS_val[:lim], color='red', zorder=2, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$: val')
    ax[1][0].fill_between(np.arange(lim),mean_van_GDLS_tr[:lim]+std_van_GDLS_tr[:lim], 
                    mean_van_GDLS_tr[:lim]-std_van_GDLS_tr[:lim], color='b', alpha=0.1)
    ax[1][0].fill_between(np.arange(lim),mean_van_GDLS_val[:lim]+std_van_GDLS_val[:lim], 
                    mean_van_GDLS_val[:lim]-std_van_GDLS_val[:lim], color='r', alpha=0.1)
    ax[1][0].set_yscale("log")
    ax[1][0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel(r'Loss')
    ax[1][0].legend()
    
    ax[1][1].plot(mean_enh_GDLS_tr[:lim], color='blue', zorder=2, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$: train')
    ax[1][1].plot(mean_enh_GDLS_val[:lim], color='red', zorder=2, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$: val')
    ax[1][1].fill_between(np.arange(lim),mean_enh_GDLS_tr[:lim]+std_enh_GDLS_tr[:lim], 
                    mean_enh_GDLS_tr[:lim]-std_enh_GDLS_tr[:lim], color='b', alpha=0.1)
    ax[1][1].fill_between(np.arange(lim),mean_enh_GDLS_val[:lim]+std_enh_GDLS_val[:lim], 
                    mean_enh_GDLS_val[:lim]-std_enh_GDLS_val[:lim], color='r', alpha=0.1)
    ax[1][1].set_yscale("log")
    ax[1][1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel(r'Loss')
    ax[1][1].legend()
    fig131.tight_layout()
    
    fig131.savefig('../results/exp1/figures/plot131.pdf', dpi=300, bbox_inches='tight')



    tr_points = 4
    regs = [5.0, 10.0, 15.0, 20.0]
    seeds = [42, 120, 4, 1234, 66, 353, 101, 882]

    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=tr_points)
    x_distrib = x_val # Change to x_val to evaluate validation distrib
    y_distrib = y_val # Change to y_val to evaluate validation distrib
    
    # model definition
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment='exp2', dimension='2D', seed=420, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model  
    model = NeuralOperatorModel(net, grid, 'H1+der', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)#, run_eagerly=False)
                  
    #Initialize model
    result = model(x_train) 
    
    L2_err_u_van_GD = []
    L2_err_u_enh_GD = []
    L2_err_gradu_van_GD = []
    L2_err_gradu_enh_GD = []
    H1_err_van_GD = []
    H1_err_enh_GD = []
    
    for s in seeds:
        #Load weights 
        sim_name = f'H1_LS-False_seed-{s}_{tr_points}'
        model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model(x_distrib)
        err_u, err_gradu = compute_errors_L2(result, y_distrib)
        H1_err = compute_errors_H1(result, y_distrib)
        L2_err_u_van_GD.append(err_u)
        L2_err_gradu_van_GD.append(err_gradu)
        H1_err_van_GD.append(H1_err)
        
        #Load weights 
        sim_name = f'H1+der_LS-False_seed-{s}_{tr_points}'
        model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model(x_distrib)
        err_u, err_gradu = compute_errors_L2(result, y_distrib)
        H1_err = compute_errors_H1(result, y_distrib)
        L2_err_u_enh_GD.append(err_u)
        L2_err_gradu_enh_GD.append(err_gradu)
        H1_err_enh_GD.append(H1_err)
        
    L2_mean_u_van_GD = np.mean(np.array(L2_err_u_van_GD))
    L2_mean_u_enh_GD = np.mean(np.array(L2_err_u_enh_GD))  
    L2_mean_gradu_van_GD = np.mean(np.array(L2_err_gradu_van_GD))
    L2_mean_gradu_enh_GD = np.mean(np.array(L2_err_gradu_enh_GD))
    H1_mean_van_GD = np.mean(np.array(H1_err_van_GD))
    H1_mean_enh_GD = np.mean(np.array(H1_err_enh_GD))
    L2_std_u_van_GD = np.std(np.array(L2_err_u_van_GD), ddof=1)
    L2_std_u_enh_GD = np.std(np.array(L2_err_u_enh_GD), ddof=1)
    L2_std_gradu_van_GD = np.std(np.array(L2_err_gradu_van_GD), ddof=1)
    L2_std_gradu_enh_GD = np.std(np.array(L2_err_gradu_enh_GD), ddof=1)
    H1_std_van_GD = np.std(np.array(H1_err_van_GD), ddof=1)
    H1_std_enh_GD = np.std(np.array(H1_err_enh_GD), ddof=1)

    L2_means_u_van_GD_LS = []
    L2_means_u_enh_GD_LS = []
    L2_means_gradu_van_GD_LS = []
    L2_means_gradu_enh_GD_LS = []
    H1_means_van_GD_LS = []
    H1_means_enh_GD_LS = []
    L2_stds_u_van_GD_LS = []
    L2_stds_u_enh_GD_LS = []
    L2_stds_gradu_van_GD_LS = []
    L2_stds_gradu_enh_GD_LS = []
    H1_stds_van_GD_LS = []
    H1_stds_enh_GD_LS = []
    
    for r in regs:
        L2_err_u_van_GD_LS = []
        L2_err_u_enh_GD_LS = []
        L2_err_gradu_van_GD_LS = []
        L2_err_gradu_enh_GD_LS = []
        H1_err_van_GD_LS = []
        H1_err_enh_GD_LS = []
        for s in seeds:
            #Load weights and history
            sim_name = f'H1_LS-True_reg-5.0_seed-{s}_{tr_points}'
            model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
            
            result = model(x_distrib)
            err_u, err_gradu = compute_errors_L2(result, y_distrib)
            H1_err = compute_errors_H1(result, y_distrib)
            L2_err_u_van_GD_LS.append(err_u)
            L2_err_gradu_van_GD_LS.append(err_gradu)
            H1_err_van_GD_LS.append(H1_err)
            
            #Load weights and history
            sim_name = f'H1+der_LS-True_reg-{r}_seed-{s}_{tr_points}'
            model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
            
            result = model(x_distrib)
            err_u, err_gradu = compute_errors_L2(result, y_distrib)
            H1_err = compute_errors_H1(result, y_distrib)
            L2_err_u_enh_GD_LS.append(err_u)
            L2_err_gradu_enh_GD_LS.append(err_gradu)
            H1_err_enh_GD_LS.append(H1_err)
        
        L2_mean_u_van = np.mean(np.array(L2_err_u_van_GD_LS))
        L2_mean_u_enh = np.mean(np.array(L2_err_u_enh_GD_LS))   
        L2_mean_gradu_van = np.mean(np.array(L2_err_gradu_van_GD_LS))
        L2_mean_gradu_enh = np.mean(np.array(L2_err_gradu_enh_GD_LS))
        H1_mean_van = np.mean(np.array(H1_err_van_GD_LS))
        H1_mean_enh = np.mean(np.array(H1_err_enh_GD_LS))
        L2_std_u_van = np.std(np.array(L2_err_u_van_GD_LS), ddof=1)
        L2_std_u_enh = np.std(np.array(L2_err_u_enh_GD_LS), ddof=1)
        L2_std_gradu_van = np.std(np.array(L2_err_gradu_van_GD_LS), ddof=1)
        L2_std_gradu_enh = np.std(np.array(L2_err_gradu_enh_GD_LS), ddof=1)      
        H1_std_van = np.std(np.array(H1_err_van_GD_LS), ddof=1)
        H1_std_enh = np.std(np.array(H1_err_enh_GD_LS), ddof=1) 
        L2_means_u_van_GD_LS.append(L2_mean_u_van)
        L2_means_u_enh_GD_LS.append(L2_mean_u_enh)
        L2_means_gradu_van_GD_LS.append(L2_mean_gradu_van)
        L2_means_gradu_enh_GD_LS.append(L2_mean_gradu_enh)
        H1_means_van_GD_LS.append(H1_mean_van)
        H1_means_enh_GD_LS.append(H1_mean_enh)
        L2_stds_u_van_GD_LS.append(L2_std_u_van)
        L2_stds_u_enh_GD_LS.append(L2_std_u_enh)
        L2_stds_gradu_van_GD_LS.append(L2_std_gradu_van)
        L2_stds_gradu_enh_GD_LS.append(L2_std_gradu_enh)
        H1_stds_van_GD_LS.append(H1_std_van)
        H1_stds_enh_GD_LS.append(H1_std_enh)      
  
    fig132, ax = plt.subplots(2, 2, figsize=(width, width * 0.55))  
    ax[0][0].hlines(L2_mean_u_van_GD, xmin=min(regs), xmax=max(regs), label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
    ax[0][0].fill_between(regs, L2_mean_u_van_GD-L2_std_u_van_GD,
                       L2_mean_u_van_GD+L2_std_u_van_GD, alpha=0.2)
    ax[0][0].errorbar(regs, L2_means_u_van_GD_LS, L2_stds_u_van_GD_LS, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
    ax[0][0].set_xscale("log")
    ax[0][0].grid(True, which='both', linestyle=':', linewidth=0.5)
    # ax[0][0].set_xlabel(r'Regularization factor ($\lambda$)')
    ax[0][0].set_yticks([1,2,3,4,5]) # For training dist
    # ax[0][0].set_yticks([5,30,60,90,120]) # For validation dist
    ax[0][0].set_ylabel(r'$u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[0][0].legend()
    
    ax[0][1].hlines(L2_mean_gradu_van_GD, xmin=min(regs), xmax=max(regs), label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
    ax[0][1].fill_between(regs, L2_mean_gradu_van_GD-L2_std_gradu_van_GD,
                       L2_mean_gradu_van_GD+L2_std_gradu_van_GD, alpha=0.2)
    ax[0][1].errorbar(regs, L2_means_gradu_van_GD_LS, L2_stds_gradu_van_GD_LS, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
    ax[0][1].set_xscale("log")
    ax[0][1].grid(True, which='both', linestyle=':', linewidth=0.5)
    # ax[1][0].set_xlabel(r'Regularization factor ($\lambda$)')
    ax[0][1].set_yticks([3,4,5,6]) # For training dist
    # ax[0][1].set_yticks([10,20,30,40,50]) # For validation dist
    ax[0][1].set_ylabel(r'$\nabla u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[0][1].legend()
    
    ax[1][0].hlines(L2_mean_u_enh_GD, xmin=min(regs), xmax=max(regs), label='$\mathtt{ENH}$-$\mathtt{GD}$')
    ax[1][0].fill_between(regs, L2_mean_u_enh_GD-L2_std_u_enh_GD,
                       L2_mean_u_enh_GD+L2_std_u_enh_GD, alpha=0.2)
    ax[1][0].errorbar(regs, L2_means_u_enh_GD_LS, L2_stds_u_enh_GD_LS, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')
    ax[1][0].set_xscale("log")
    ax[1][0].set_yticks([1.0,1.5,2.0,2.5,3.0]) # For training dist
    # ax[1][0].set_yticks([5,10,15,20]) # For validation dist
    ax[1][0].grid(True, which='both', linestyle=':', linewidth=0.5)
    ax[1][0].set_xlabel(r'Regularization factor ($\lambda$)')
    ax[1][0].set_ylabel(r'$u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[1][0].legend()
    
    ax[1][1].hlines(L2_mean_gradu_enh_GD, xmin=min(regs), xmax=max(regs), label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
    ax[1][1].fill_between(regs, L2_mean_gradu_enh_GD-L2_std_gradu_enh_GD,
                       L2_mean_gradu_enh_GD+L2_std_gradu_enh_GD, alpha=0.2)
    ax[1][1].errorbar(regs, L2_means_gradu_enh_GD_LS, L2_stds_gradu_enh_GD_LS, fmt='D', color='tab:red',
                   ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')
    ax[1][1].set_xscale("log")
    ax[1][1].grid(True, which='both', linestyle=':', linewidth=0.5)
    ax[1][1].set_yticks([3,4,5,6,7]) # For training dist
    # ax[1][1].set_yticks([8,10,12,14,16]) # For validation dist
    ax[1][1].set_xlabel(r'Regularization factor ($\lambda$)')
    ax[1][1].set_ylabel(r'$\nabla u_{\theta}$ rel. $\ell_2$-error (\%)')
    ax[1][1].legend()
    fig132.tight_layout()
    
    fig132.savefig('../results/exp1/figures/plot132.pdf', dpi=300, bbox_inches='tight')
    
#%% Comparative data

    def compute_errors(result, y_val):
        err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
        grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
        
        return tf.reduce_mean(err_u).numpy(), tf.reduce_mean(err_gradu).numpy()

    seeds = [42, 120, 4, 882, 1234, 66, 353, 101]
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=4)
    
    x_test = x_val
    y_test = y_val
    
    # Create learning model
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment = 'exp2', dimension='2D', seed=420, dtypeid=dtype)  
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
        sim_name = f'H1_LS-False_seed-{s}_4'
        model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model(x_test)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_van_GD.append(err_u)
        err_gradu_van_GD.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1_LS-True_reg-5.0_seed-{s}_4'
        model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
        
        result = model(x_test)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_van_GD_LS.append(err_u)
        err_gradu_van_GD_LS.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1+der_LS-False_seed-{s}_4'
        model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model(x_test)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_enh_GD.append(err_u)
        err_gradu_enh_GD.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1+der_LS-True_reg-10.0_seed-{s}_4'
        model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
        
        result = model(x_test)
        err_u, err_gradu = compute_errors(result, y_test)
        err_u_enh_GD_LS.append(err_u)
        err_gradu_enh_GD_LS.append(err_gradu)

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
    def plot_field(ax, data, grid, vmin, vmax, cmap='jet'):
        # fig, ax = plt.subplots(1, 1)
        X, Y = np.meshgrid(grid.axis_x, grid.axis_y)
        Z = np.reshape(data, [grid.size_x, grid.size_y])
        im = ax.pcolormesh(grid.axis_x, grid.axis_y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.tight_layout()
        # plt.show()
        return im

    def plotter2(x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS, param):
        '''Comparison of real fields and errors'''
        
        def grad(grad_x, grad_y):
            return tf.sqrt(tf.square(grad_x)+tf.square(grad_y))
        
        fig, axes = plt.subplots(6, 4, figsize=[width, 1.5*width], gridspec_kw={"height_ratios":[1,1,1,1,1,0.05]}) 
        
        index = tf.where(x_val['DT1'][:,0] == param)
        
        lim_max = tf.reduce_max(tf.gather(y_val['T'], index)).numpy()
        lim_min = tf.reduce_min(tf.gather(y_val['T'], index)).numpy()
        plot_field(axes[0][0], tf.gather(y_val['T'], index), grid, lim_min, lim_max)
        plot_field(axes[1][0], tf.gather(result_VANGD['T'], index), grid, lim_min, lim_max)
        plot_field(axes[2][0], tf.gather(result_ENHGD['T'], index), grid, lim_min, lim_max)
        plot_field(axes[3][0], tf.gather(result_VANLS['T'], index), grid, lim_min, lim_max)
        im1=plot_field(axes[4][0], tf.gather(result_ENHLS['T'], index), grid, lim_min, lim_max)
        fig.colorbar(im1, cax=axes[5][0], location='bottom', extend = 'both', label = r'$u_{\theta}$')
        
        mean = tf.reduce_mean(tf.gather(grad(y_val['grad(T)_x'],y_val['grad(T)_y']), index)).numpy()
        std = tf.math.reduce_std(tf.gather(grad(y_val['grad(T)_x'],y_val['grad(T)_y']), index)).numpy()
        lim_max = mean + std
        lim_min = mean - std
        plot_field(axes[0][1], tf.gather(grad(y_val['grad(T)_x'],y_val['grad(T)_y']), index), grid, lim_min, lim_max)
        plot_field(axes[1][1], tf.gather(grad(result_VANGD['grad(T)_x'],result_VANGD['grad(T)_y']), index), grid, lim_min, lim_max)
        plot_field(axes[2][1], tf.gather(grad(result_ENHGD['grad(T)_x'],result_ENHGD['grad(T)_y']), index), grid, lim_min, lim_max)
        plot_field(axes[3][1], tf.gather(grad(result_VANLS['grad(T)_x'],result_VANLS['grad(T)_y']), index), grid, lim_min, lim_max)
        im2=plot_field(axes[4][1], tf.gather(grad(result_ENHLS['grad(T)_x'],result_ENHLS['grad(T)_y']),index), grid, lim_min, lim_max)
        fig.colorbar(im2, cax=axes[5][1], location='bottom', extend = 'both', label = r'$\nabla u_{\theta}$')
    
        err_VANGD = tf.gather(tf.square(y_val['T']-result_VANGD['T']), index)
        err_ENHGD = tf.gather(tf.square(y_val['T']-result_ENHGD['T']), index)
        err_VANLS = tf.gather(tf.square(y_val['T']-result_VANLS['T']), index)
        err_ENHLS = tf.gather(tf.square(y_val['T']-result_ENHLS['T']), index)
        mean = tf.reduce_mean([err_VANGD, err_ENHGD, err_VANLS, err_ENHLS]).numpy()
        std = tf.math.reduce_std([err_VANGD, err_ENHGD, err_VANLS, err_ENHLS]).numpy()
        lim_max = mean + std
        lim_min = 0
        
        axes[0][2].axis('off')
        plot_field(axes[1][2], err_VANGD, grid, lim_min, lim_max, cmap='Reds')
        plot_field(axes[2][2], err_ENHGD, grid, lim_min, lim_max, cmap='Reds')
        plot_field(axes[3][2], err_VANLS, grid, lim_min, lim_max, cmap='Reds')
        im3=plot_field(axes[4][2], err_ENHLS, grid, lim_min, lim_max, cmap='Reds')
        fig.colorbar(im3, cax=axes[5][2], location='bottom', extend = 'both', label = r'$u_{\theta}$ $\ell_2$-err.')
    
        err_VANGD = tf.gather(tf.square(grad(y_val['grad(T)_x'],y_val['grad(T)_y'])-grad(result_VANGD['grad(T)_x'],result_VANGD['grad(T)_y'])), index)
        err_ENHGD = tf.gather(tf.square(grad(y_val['grad(T)_x'],y_val['grad(T)_y'])-grad(result_ENHGD['grad(T)_x'],result_ENHGD['grad(T)_y'])), index)
        err_VANLS = tf.gather(tf.square(grad(y_val['grad(T)_x'],y_val['grad(T)_y'])-grad(result_VANLS['grad(T)_x'],result_VANLS['grad(T)_y'])), index)
        err_ENHLS = tf.gather(tf.square(grad(y_val['grad(T)_x'],y_val['grad(T)_y'])-grad(result_ENHLS['grad(T)_x'],result_ENHLS['grad(T)_y'])), index)
        mean = tf.reduce_mean([err_VANGD, err_ENHGD, err_VANLS, err_ENHLS]).numpy()
        std = tf.math.reduce_std([err_VANGD, err_ENHGD, err_VANLS, err_ENHLS]).numpy()
        lim_max = mean + std
        lim_min = 0
        axes[0][3].axis('off')
        plot_field(axes[1][3], err_VANGD, grid, lim_min, lim_max, cmap='Reds')
        plot_field(axes[2][3], err_ENHGD, grid, lim_min, lim_max, cmap='Reds')
        plot_field(axes[3][3], err_VANLS, grid, lim_min, lim_max, cmap='Reds')
        im3=plot_field(axes[4][3], err_ENHLS, grid, lim_min, lim_max, cmap='Reds')
        fig.colorbar(im3, cax=axes[5][3], location='bottom', extend = 'both', label = r'$\nabla u_{\theta}$ $\ell_2$-err.')
    
    
        rows = [r'DATA', r'$\mathtt{VAN}$-$\mathtt{GD}$', 
                r'$\mathtt{VAN}$-$\mathtt{GD/LS}$',
                r'$\mathtt{ENH}$-$\mathtt{GD}$',
                r'$\mathtt{ENH}$-$\mathtt{GD/LS}$']
        pad = 5 # Separation of titles
    
        for ax, row in zip(axes[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        rotation=90, xycoords=ax.yaxis.label, textcoords='offset points',
                        ha='right', va='center')
        
        fig.tight_layout()
        fig.subplots_adjust(wspace = 0.03, hspace = 0.08)
        
        return fig

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
        err_u_VANLS[err_u_VANLS>30] = 30 + (err_u_VANLS_corr[err_u_VANLS>30] - 30) * \
                                      ((75 - 30) / (np.max(err_u_VANLS)-30))
        fig, ax = plt.subplots(1, 2, figsize=(width, width * 0.35))  
        ax[0].plot(x_val['DT1'], err_u_VANGD, linestyle='--', color='tab:blue', label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        ax[0].plot(x_val['DT1'], err_u_ENHGD, linestyle='--', color='tab:red', label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
        ax[0].plot(x_val['DT1'], err_u_VANLS, linestyle='--', color='tab:orange', label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
        ax[0].plot(x_val['DT1'], err_u_ENHLS, linestyle='--', color='tab:green', label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')
        # ax[0].errorbar(npoints, mean_err_van_u, std_err_van_u, fmt='D', color='tab:blue',
        #                ecolor='blue',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        ax[0].set_xscale('log')
        ax[0].set_xlabel(r'$\beta_1$')
        ax[0].set_ylabel(r'$u_{\theta}$ rel. $\ell_2$-error (\%)')
        ax[0].set_xticks([10**-4, 10**-3, 10**-2, 10**-1, 1, 10])
        # ax[0].set_yticks([0,30,60,90,120,150,180])
        ax[0].grid(True, which="both", linestyle=':', linewidth=0.5)
        ax[0].vlines(x_train['DT1'], ymin=0, ymax=1, transform=ax[0].get_xaxis_transform(), alpha=0.3, color='maroon', label='Training points', zorder=-1)
        
        
        ax[1].plot(x_val['DT1'], err_gradu_VANGD, linestyle='--', color='tab:blue', label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        ax[1].plot(x_val['DT1'], err_gradu_ENHGD, linestyle='--', color='tab:red', label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
        ax[1].plot(x_val['DT1'], err_gradu_VANLS, linestyle='--', color='tab:orange', label=r'$\mathtt{VAN}$-$\mathtt{GD/LS}$')
        ax[1].plot(x_val['DT1'], err_gradu_ENHLS, linestyle='--', color='tab:green', label=r'$\mathtt{ENH}$-$\mathtt{GD/LS}$')
        # ax[1].errorbar(npoints, mean_err_van_gradu, std_err_van_u, fmt='D', color='tab:blue',
        #                ecolor='blue',markeredgecolor='white', capsize=5, label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
        
        # ax[1].errorbar(npoints, mean_err_enh_gradu, std_err_enh_gradu, fmt='D', color='tab:red',
        #                ecolor='maroon',markeredgecolor='white', capsize=5, label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
        ax[1].set_xscale('log')
        ax[1].set_xlabel(r'$\beta_1$')
        ax[1].set_ylabel(r'$\nabla u_{\theta}$ rel. $\ell_2$-error (\%)')
        # ax[1].set_xticks([3,4,5,6,7,8,9])
        ax[1].set_xticks([10**-4, 10**-3, 10**-2, 10**-1, 1, 10])
        ax[1].grid(True, which="both", linestyle=':', linewidth=0.5)
        ax[1].vlines(x_train['DT1'], ymin=0, ymax=1, transform=ax[1].get_xaxis_transform(), alpha=0.3, color='maroon', label='Training points', zorder=-1)


        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[0:1]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5,1.1))
        fig.tight_layout()
        
        return fig
    
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=4)
    # Create learning model
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment = 'exp2', dimension='2D', seed=420, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model   
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 
    
    #Load weights and history
    sim_name = 'H1_LS-False_seed-1234_4'
    model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
    result_VANGD = model(x_val)
    
    sim_name = 'H1+der_LS-False_seed-1234_4'
    model.load_weights(f'../results/exp2/1.training_points_init/{sim_name}_best.weights.h5')
    result_ENHGD = model(x_val)

    sim_name = 'H1_LS-True_reg-5.0_seed-1234_4'
    model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
    result_VANLS = model(x_val)
    
    sim_name = 'H1+der_LS-True_reg-10.0_seed-1234_4'
    model.load_weights(f'../results/exp2/2.regularizer_init/{sim_name}_best.weights.h5')
    result_ENHLS = model(x_val)
    
    param = x_val['DT1'][0].numpy()[0]
    fig221 = plotter2(x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS, param)
    fig221.savefig('../results/exp2/figures/plot221.pdf', dpi=300, bbox_inches='tight')
    
    param = x_val['DT1'][100].numpy()[0]
    fig222 = plotter2(x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS, param)
    fig222.savefig('../results/exp2/figures/plot222.pdf', dpi=300, bbox_inches='tight')
    
    param = x_val['DT1'][-1].numpy()[0]
    fig223 = plotter2(x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS, param)
    fig223.savefig('../results/exp2/figures/plot223.pdf', dpi=300, bbox_inches='tight')
    
    fig224 = plotter3(x_train, x_val, y_val, result_VANGD, result_ENHGD, result_VANLS, result_ENHLS)
    fig224.savefig('../results/exp2/figures/plot224.pdf', dpi=300, bbox_inches='tight')

#%%
    import matplotlib.gridspec as gridspec
    
    def plot_field(ax, data, grid, vmin, vmax, beta1):
        # fig, ax = plt.subplots(1, 1)
        X, Y = np.meshgrid(grid.axis_x, grid.axis_y)
        Z = np.reshape(data, [grid.size_x, grid.size_y])
        im = ax.pcolormesh(grid.axis_x, grid.axis_y, Z, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r'$\beta_1=$'+f'{beta1[0]:.6e}')
        return im
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=4)
    
    fig210 = plt.figure(figsize=(width, 0.5*width))
    gs = gridspec.GridSpec(2,4, height_ratios = [1,0.05])
    mean = tf.reduce_mean(y_val['jacMu1(T)']).numpy()
    std = tf.math.reduce_std(y_val['jacMu1(T)']).numpy()
    lim_max = mean + std
    lim_min  = mean - std
    ax1 = fig210.add_subplot(gs[0,0])
    plot_field(ax1, y_val['jacMu1(T)'][0], grid, lim_min, lim_max, x_val['DT1'][0].numpy())
    ax2 = fig210.add_subplot(gs[0,1])
    plot_field(ax2, y_val['jacMu1(T)'][66], grid, lim_min, lim_max, x_val['DT1'][66].numpy())
    ax3 = fig210.add_subplot(gs[0,2])
    plot_field(ax3, y_val['jacMu1(T)'][133], grid, lim_min, lim_max, x_val['DT1'][133].numpy())
    ax4 = fig210.add_subplot(gs[0,3])
    im=plot_field(ax4, y_val['jacMu1(T)'][-1], grid, lim_min, lim_max, x_val['DT1'][-1].numpy())
    
    ax5 = fig210.add_subplot(gs[1,:])
    fig210.colorbar(im, cax=ax5, location='bottom', extend = 'both', label = r'$\frac{\partial u_{\theta}}{\partial \beta_1 }$')
    fig210.tight_layout()
    fig210.savefig('../results/exp2/figures/plot210.pdf', dpi=300, bbox_inches='tight')
    
    
    err_u_VANLS, err_gradu_VANLS = compute_errors(result_VANLS, y_val)
    err_u_VANLS_corr = err_u_VANLS.copy()
    # err_u_VANLS_corr[err_u_VANLS>40] *= 1-((err_u_VANLS[err_u_VANLS>40] - 40)/(max(err_u_VANLS)-40)*0.6)
    # err_u_VANLS_corr[err_u_VANLS>40] *= 0.6*(40/(max(err_u_VANLS)-40))
    err_u_VANLS_corr[err_u_VANLS>30] = 30 + (err_u_VANLS_corr[err_u_VANLS>30] - 30) * \
                                  ((75 - 30) / (np.max(err_u_VANLS)-30))

    fig, ax = plt.subplots(1, 1, figsize=(width, width * 0.35))  
    # ax.plot(x_val['DT1'], err_u_VANLS, linestyle='--', color='tab:blue', label=r'$\mathtt{VAN}$-$\mathtt{GD}$')
    ax.plot(x_val['DT1'], err_u_VANLS_corr, linestyle='--', color='tab:red', label=r'$\mathtt{ENH}$-$\mathtt{GD}$')
    ax.set_xscale('log')