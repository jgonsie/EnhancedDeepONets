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
    data_route = '../OpenFOAM/experiment1/training_data/'
    training_data = fRW.upload_training_data(data_route, experiment='exp1', dtype=dtype)
    
    #%% PLOTS 1.1 (dataset distribution)
    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=7)

    # Plot training distribution
    pure_conv_route = '../OpenFOAM/experiment1/pure_convection'
    pure_conv_case = fRW.upload_single_data(pure_conv_route, jacobian=False)
    pure_diff_route = '../OpenFOAM/experiment1/pure_diffusion'
    pure_diff_case = fRW.upload_single_data(pure_diff_route, jacobian=False)
    
    plot111 = pp.plot_log_data_distribution3(x_train, x_val, y_train, y_val, pure_conv_case,
                                    pure_diff_case, grid)
    
    plot111.savefig('../results/exp1/figures/plot111.pdf', dpi=300, bbox_inches='tight')

   
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
    
    
    # Create learning model
    net = DeepONet(layers_branch=[50,50,70], layers_trunk=[50,50,50,70], 
                   experiment = 'exp1', dimension='2D', seed=420, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model   
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 
    
    npoints = [3, 5, 7,9]
    seeds = [1234, 66, 353, 101, 42, 120, 4, 882] 
    
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
            model.load_weights(f'../results/exp1/1.training_points_init/{sim_name}_best.weights.h5')
            result = model(x_val)
            err_u, err_gradu = compute_errors_L2(result, y_val) 
            van_errs_u.append(tf.reduce_mean(err_u).numpy())
            van_errs_gradu.append(tf.reduce_mean(err_gradu).numpy())

            #Load weights and history
            sim_name = f'H1+der_LS-False_seed-{s}_{p}'
            model.load_weights(f'../results/exp1/1.training_points_init/{sim_name}_best.weights.h5')
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
    
    
    tr_points = 7
    van_GD_histories = []
    van_GDLS_histories = []
    enh_GD_histories = []
    enh_GDLS_histories = []
    
    for s in seeds:
        sim_name = f'H1_LS-False_seed-{s}_{tr_points}'
        van_GD_histories.append(pd.read_csv(f'../results/exp1/1.training_points_init/{sim_name}_history.csv'))
        sim_name = f'H1+der_LS-False_seed-{s}_{tr_points}'
        enh_GD_histories.append(pd.read_csv(f'../results/exp1/1.training_points_init/{sim_name}_history.csv'))
        sim_name = f'H1_LS-True_reg-5.0_seed-{s}_{tr_points}'
        van_GDLS_histories.append(pd.read_csv(f'../results/exp1/2.regularizer_init/{sim_name}_history.csv'))
        sim_name = f'H1+der_LS-True_reg-10.0_seed-{s}_{tr_points}'
        enh_GDLS_histories.append(pd.read_csv(f'../results/exp1/2.regularizer_init/{sim_name}_history.csv'))


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
    ax[0][0].set_ylim([10**-4,0.9])
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
    ax[0][1].set_ylim([10**-4,0.9])
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
    ax[1][0].set_ylim([10**-4,0.9])
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
    ax[1][1].set_ylim([10**-4,0.9])
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel(r'Loss')
    ax[1][1].legend()
    fig131.tight_layout()
    
    fig131.savefig('../results/exp1/figures/plot131.pdf', dpi=300, bbox_inches='tight')



    tr_points = 7
    regs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    seeds = [42, 120, 4, 882, 1234, 66, 353, 101]

    
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=tr_points)
    x_distrib = x_train # Change to x_val to evaluate validation distrib
    y_distrib = y_train # Change to y_val to evaluate validation distrib
    
    # model definition
    net = DeepONet(layers_branch=[50,50,70], layers_trunk=[50,50,50,70], 
                   experiment='exp1', dimension='2D', seed=420, dtypeid=dtype)  
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
        model.load_weights(f'../results/exp1/1.training_points_init/{sim_name}_best.weights.h5')
        
        result = model(x_distrib)
        err_u, err_gradu = compute_errors_L2(result, y_distrib)
        H1_err = compute_errors_H1(result, y_distrib)
        L2_err_u_van_GD.append(err_u)
        L2_err_gradu_van_GD.append(err_gradu)
        H1_err_van_GD.append(H1_err)
        
        #Load weights 
        sim_name = f'H1+der_LS-False_seed-{s}_{tr_points}'
        model.load_weights(f'../results/exp1/1.training_points_init/{sim_name}_best.weights.h5')
        
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
            sim_name = f'H1_LS-True_reg-{r}_seed-{s}_{tr_points}'
            model.load_weights(f'../results/exp1/2.regularizer_init/{sim_name}_best.weights.h5')
            
            result = model(x_distrib)
            err_u, err_gradu = compute_errors_L2(result, y_distrib)
            H1_err = compute_errors_H1(result, y_distrib)
            L2_err_u_van_GD_LS.append(err_u)
            L2_err_gradu_van_GD_LS.append(err_gradu)
            H1_err_van_GD_LS.append(H1_err)
            
            #Load weights and history
            sim_name = f'H1+der_LS-True_reg-{r}_seed-{s}_{tr_points}'
            model.load_weights(f'../results/exp1/2.regularizer_init/{sim_name}_best.weights.h5')
            
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
    
    npoints = 7
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=npoints)

    # Net definition
    net = DeepONet(layers_branch=[50,50,70], layers_trunk=[50,50,50,70], 
                    dimension='2D', seed=420, dtypeid=dtype)  
    loss_weights = generate_loss_weights(y_train)
    
    # Select model  
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)#, run_eagerly=False)
                  
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
        sim_name = f'H1_LS-False_{s}'
        model.load_weights(f'../results/exp1/3.initializer/{sim_name}_best.weights.h5')
        
        result = model(x_val)
        err_u, err_gradu = compute_errors(result, y_val)
        err_u_van_GD.append(err_u)
        err_gradu_van_GD.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1_LS-True_{s}'
        model.load_weights(f'../results/exp1/3.initializer/{sim_name}_best.weights.h5')
        
        result = model(x_val)
        err_u, err_gradu = compute_errors(result, y_val)
        err_u_van_GD_LS.append(err_u)
        err_gradu_van_GD_LS.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1+der_LS-False_{s}'
        model.load_weights(f'../results/exp1/3.initializer/{sim_name}_best.weights.h5')
        
        result = model(x_val)
        err_u, err_gradu = compute_errors(result, y_val)
        err_u_enh_GD.append(err_u)
        err_gradu_enh_GD.append(err_gradu)
        
        #Load weights and history
        sim_name = f'H1+der_LS-True_{s}'
        model.load_weights(f'../results/exp1/3.initializer/{sim_name}_best.weights.h5')
        
        result = model(x_val)
        err_u, err_gradu = compute_errors(result, y_val)
        err_u_enh_GD_LS.append(err_u)
        err_gradu_enh_GD_LS.append(err_gradu)

    mean_u_van_GD = np.mean(np.array(err_u_van_GD))
    mean_u_van_GD_LS = np.mean(np.array(err_u_van_GD_LS))
    mean_u_enh_GD = np.mean(np.array(err_u_enh_GD))
    mean_u_enh_GD_LS = np.mean(np.array(err_u_enh_GD_LS))
    
    std_u_van_GD = np.sqrt(np.mean(np.square(np.array(err_u_van_GD)-mean_u_van_GD)))
    std_u_van_GD_LS = np.sqrt(np.mean(np.square(np.array(err_u_van_GD_LS)-mean_u_van_GD_LS)))
    std_u_enh_GD = np.sqrt(np.mean(np.square(np.array(err_u_enh_GD)-mean_u_enh_GD)))
    std_u_enh_GD_LS = np.sqrt(np.mean(np.square(np.array(err_u_enh_GD_LS)-mean_u_enh_GD_LS)))
    
    mean_gradu_van_GD = np.mean(np.array(err_gradu_van_GD))
    mean_gradu_van_GD_LS = np.mean(np.array(err_gradu_van_GD_LS))
    mean_gradu_enh_GD = np.mean(np.array(err_gradu_enh_GD))
    mean_gradu_enh_GD_LS = np.mean(np.array(err_gradu_enh_GD_LS))
    
    std_gradu_van_GD = np.sqrt(np.mean(np.square(np.array(err_gradu_van_GD)-mean_gradu_van_GD)))
    std_gradu_van_GD_LS = np.sqrt(np.mean(np.square(np.array(err_gradu_van_GD_LS)-mean_gradu_van_GD_LS)))
    std_gradu_enh_GD = np.sqrt(np.mean(np.square(np.array(err_gradu_enh_GD)-mean_gradu_enh_GD)))
    std_gradu_enh_GD_LS = np.sqrt(np.mean(np.square(np.array(err_gradu_enh_GD_LS)-mean_gradu_enh_GD_LS)))
    
    print(f'u VAN-GD: {mean_u_van_GD:.2f} +- {std_u_van_GD:.2f}')
    print(f'gradu VAN-GD: {mean_gradu_van_GD:.2f} +- {std_gradu_van_GD:.2f}')
    
    print(f'u ENH-GD: {mean_u_enh_GD:.2f} +- {std_u_enh_GD:.2f}')
    print(f'gradu ENH-GD: {mean_gradu_enh_GD:.2f} +- {std_gradu_enh_GD:.2f}')
    
    print(f'u VAN-GD-LS: {mean_u_van_GD_LS:.2f} +- {std_u_van_GD_LS:.2f}')
    print(f'gradu VAN-GD_LS: {mean_gradu_van_GD_LS:.2f} +- {std_gradu_van_GD_LS:.2f}')
    
    print(f'u ENH-GD_LS: {mean_u_enh_GD_LS:.2f} +- {std_u_enh_GD_LS:.2f}')
    print(f'gradu ENH-GD_LS: {mean_gradu_enh_GD_LS:.2f} +- {std_gradu_enh_GD_LS:.2f}')
    
    
    #%%
    s = 353
    sim_name = f'H1_LS-True_{s}'
    df_history = pd.read_csv(f'../results/exp1/3.initializer/{sim_name}_history.csv')
    plot1 = pp.plot_loss(df_history)
    # plot2 = pp.plot_partialLosses_train(df_history)
    # plot3 = pp.plot_partialLosses_val(df_history)
    
    sim_name = f'H1+der_LS-True_{s}'
    df_history = pd.read_csv(f'../results/exp1/3.initializer/{sim_name}_history.csv')
    plot1 = pp.plot_loss(df_history)
    # plot2 = pp.plot_partialLosses_train(df_history)
    # plot3 = pp.plot_partialLosses_val(df_history)