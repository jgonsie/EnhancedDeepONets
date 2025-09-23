#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:29:38 2025

@author: jesusglezs97
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from SRC.utils import fix_random, fix_precission
from SRC.utils import prepare_raw_data_items_esp2, prepare_raw_data_items, generate_loss_weights
from SRC.models import DeepONet, NeuralOperatorModel
from SRC import foamRW as fRW
from SRC.grids2D import Grid
from SRC import postprocessing as pp

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['EAGER_CONSTANT_FOLDING'] = 'False'
# os.environ['XLA_FLAGS']= "--xla_gpu_cuda_data_dir=/home/jgonzalez/anaconda3/pkgs/cuda-nvvm-tools-12.6.77-he02047a_0 --xla_dump_to=/temp/xla_dump"
# os.environ['XLA_FLAGS']= '--xla_gpu_cuda_data_dir=/home/jgonzalez/anaconda3/pkgs/cuda-nvvm-tools-12.6.77-he02047a_0 --xla_disable_hlo_passes=constant_folding'
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# tf.config.experimental.get_memory_info("GPU:0")['current']

if __name__ == "__main__":
    
    fix_random(1234)
    dtype = fix_precission(double_precission = True)

    # Create grid
    grid = Grid(size_x = 50, size_y = 50, step_size = 1/50)

    # Data for training and validation
    data_route = '../OpenFOAM/experiment2/training_data/'
    training_data = fRW.upload_training_data(data_route, experiment='exp2', dtype=dtype)
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp2(training_data, train_split=5)
    # x_train, y_train, x_val, y_val = prepare_raw_data_items(training_data, train_split=8)

    # Create learning model
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment='exp2', dimension='2D', seed=420, dtypeid=dtype)  

    loss_weights = generate_loss_weights(y_train)
    
    # Select model
    loss = 'H1+der'
    model = NeuralOperatorModel(net, grid, loss, LS=False, kappas = loss_weights,
                                regularizer = 10**(-1))


    sim_name = f'{loss}_test_2'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)#, run_eagerly=False)
                  
    #Initialize model
    result = model(x_train) 

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'../results/exp2/{sim_name}_best.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,  # Save only the model weights
        verbose=0)
    nanchecker = keras.callbacks.TerminateOnNaN()
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=10000, batch_size=5, validation_batch_size=39,
                        callbacks=[nanchecker, checkpoint])
    
    model.save_weights(f'results/exp2/{sim_name}_final.weights.h5')
    
    df_history = pd.DataFrame(history.history)
    # plot1 = pp.plot_loss(df_history)
    # plot2 = pp.plot_partialLosses_train(df_history)
    # plot3 = pp.plot_partialLosses_val(df_history)
    # plt.savefig('f'results/exp1/{sim_name}.png', dpi=200, bbox_inches='tight')

    # df_history.to_csv(f'results/exp1/1.training_points/{sim_name}.csv', index=False)
    # df_history = pd.read_csv('f'experiments/L1+Lder_training.csv')
    
    def compute_errors_L2(result, y_val):
        err_u = tf.sqrt(tf.reduce_sum(tf.square(result['T']-y_val['T']),axis=1)/tf.reduce_sum(tf.square(y_val['T']),axis=1)) * 100
        grad_ut = tf.sqrt(tf.square(result['grad(T)_x'])+tf.square(result['grad(T)_y']))
        grad_uh = tf.sqrt(tf.square(y_val['grad(T)_x'])+tf.square(y_val['grad(T)_y']))
        err_gradu = tf.sqrt(tf.reduce_sum(tf.square(grad_ut-grad_uh),axis=1)/tf.reduce_sum(tf.square(grad_uh),axis=1)) * 100
        
        return tf.reduce_mean(err_u).numpy(), tf.reduce_mean(err_gradu).numpy()
    
    result = model(x_val)
    err_u, err_gradu = compute_errors_L2(result, y_val) 
    print(f'Error u: {err_u}\n Error gradu: {err_gradu}')
    
    model.load_weights(f'results/exp2/{sim_name}_best.weights.h5')
    result = model(x_val)
    err_u, err_gradu = compute_errors_L2(result, y_val) 
    print(f'Error u: {err_u}\n Error gradu: {err_gradu}')
    
    # Load weights with best performance
    # model.load_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')
    # model.save_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')