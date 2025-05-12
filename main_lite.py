#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:29:38 2025

@author: jesusglezs97
"""
#TODOS:
    # - Chequear que el LS empiece desde un valor de loss menor
    # - encontrar caso que vaya bien
  
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from SRC.utils import fix_random, fix_precission
from SRC.utils import prepare_raw_data_items_esp, prepare_raw_data_items, generate_loss_weights
from SRC.models import DeepONet, NeuralOperatorModel
from SRC import foamRW as fRW
from SRC.grids2D import Grid
from SRC import postprocessing as pp

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['XLA_FLAGS']= "--xla_gpu_cuda_data_dir=/home/jgonzalez/anaconda3/pkgs/cuda-nvvm-tools-12.6.77-he02047a_0 --xla_dump_to=/temp/xla_dump"

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
    data_route = './OpenFOAM/experiment1/training_data/'
    training_data = fRW.upload_training_data(data_route, jacobian=True, dtype=dtype)
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=3)
    x_train, y_train, x_val, y_val = prepare_raw_data_items(training_data, train_split=8)


    # Create learning model
    net = DeepONet(layers_branch=[50,50,70], layers_trunk=[50,50,50,70], 
                   dimension='2D', seed=420, dtypeid=dtype)
        
    loss_weights = generate_loss_weights(y_train)
    
    # Select model
    # model = NeuralOperatorModel(net, grid, 'vanilla', LS=True, regularizer = 10**(-2))
    # model = NeuralOperatorModel(net, grid, 'van+der',  LS=True, kappas=loss_weights,
    #                             regularizer = 10**(-1))
    model = NeuralOperatorModel(net, grid, 'H1', LS=False, kappas=loss_weights,
                                regularizer = 10**(-1))    
    # model = NeuralOperatorModel(net, grid, 'H1+der', LS=True, kappas = loss_weights,
    #                             regularizer = 10**(-1))


    sim_name = 'H1_20points'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)#, run_eagerly=True)
                  
    #Initialize model
    result = model(x_train) 

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'results/exp1/{sim_name}_weights_GD3.weights.h5',
        monitor='loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,  # Save only the model weights
        verbose=0)
    nanchecker = keras.callbacks.TerminateOnNaN()
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=10000, batch_size=20, validation_batch_size=36,
                        callbacks=[nanchecker])

    df_history = pd.DataFrame(history.history)
    # plot1 = pp.plot_loss(df_history)
    # plot2 = pp.plot_partialLosses_train(df_history)
    # plot3 = pp.plot_partialLosses_val(df_history)
    # plt.savefig('f'results/exp1/{sim_name}.png', dpi=200, bbox_inches='tight')

    df_history.to_csv(f'results/exp1/1.training_points/{sim_name}.csv', index=False)
    # df_history = pd.read_csv('f'experiments/L1+Lder_training.csv')

    # Load weights with best performance
    # model.load_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')
    # model.save_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')
    
    weights_original, weights_original2,  weights_normal_imp, weights_normal_exp, weights_david = model.check_LS(x_train,y_train)
