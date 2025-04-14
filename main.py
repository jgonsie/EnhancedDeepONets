#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:29:38 2025

@author: jesusglezs97
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from SRC.utils import DT_lognorm_dist, prepare_raw_data_custom, get_model_performance
from SRC.models import DeepONet, NeuralOperatorModel
from SRC.foamRW import foamRW as fRW
from SRC.grids2D import Grid
from SRC import postprocessing as pp

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras as keras
keras.backend.set_floatx('float64')

# Random seeds for being deterministic.
keras.utils.set_random_seed(1234)


if __name__ == "__main__":
    
    # Generate random samples following a log-normal distribution
    # samples = DT_lognorm_dist(1.0, -1.0, 200)
    # samples = np.sort(samples)
    # np.savetxt('exp1_randomDT.dat', samples, fmt='%.9f')
    
    # Create grid
    grid = Grid(size_x = 50, size_y = 50, step_size = 1/50)

    # Data for training and validation
    data_route = '../OpenFOAM/convectionDiffusion2D_10x10_mu_v2/training_data/'
    training_data = fRW.upload_training_data(data_route, jacobian=True)
    # x_train, y_train, x_val, y_val = prepare_raw_data(training_data, train_split=0.72)
    # x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[4,14,25])
    x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[1,9,25])
    plot0 = pp.plot_data_distribution(x_train, y_train, x_val, y_val)

    # Create learning model
    net = DeepONet(layers_branch_v=[50], layers_branch_mu=[50,50], 
                    layers_trunk=[50,50,50], num_rand_sampling = 20, dimension='2D')

    # Select model
    # model = NeuralOperatorModel(net, grid, 'vanilla', LS=True)
    # model = NeuralOperatorModel(net, grid, 'H1', LS=True)
    # model = NeuralOperatorModel(net, grid, 'phy', LS=True)
    model = NeuralOperatorModel(net, grid, 'van+der',  LS=True)
    # model = NeuralOperatorModel(net, grid, 'H1+der',  LS=True)
    # model = NeuralOperatorModel(net, grid, 'phy+der',  LS=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))#,
                  #jit_compile=True)#, run_eagerly=True)
                  
    #Initialize model
    result = model(x_train) 

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='experiments/L3+Lder_LS_weights_GD.ckpt',
        monitor='loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,  # Save only the model weights
        verbose=0)

    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), 
                        epochs=4000, batch_size=3, validation_batch_size=25)#,
                        # callbacks=[checkpoint])

    df_history = pd.DataFrame(history.history)
    plot1 = pp.plot_loss(df_history)
    plot2 = pp.plot_partialLosses_train(df_history)
    plot3 = pp.plot_partialLosses_val(df_history)
    plot4 = pp.plot_L2re_u(df_history, ymin = 0, ymax = 100)
    plot5 = pp.plot_L2re_gradu(df_history, ymin = 0, ymax = 100)
    # plt.savefig('L1+Lder/L1+Lder_loss_GD.png', dpi=200, bbox_inches='tight')

    df_history.to_csv(f'experiments/L3+Lder_LS.csv', index=False)
    # df_history = pd.read_csv('f'experiments/L1+Lder_training.csv')

    # Load weights with best performance
    model.load_weights('experiments/L3+Lder_LS_weights_GD.ckpt')

    epoch_min = np.argmin(history.history['loss'])
    print('#### TRAINING ####')
    print('MIN LOSS u: ', history.history['loss_u'][epoch_min])
    print('MIN ERR u:', history.history['L2re_u'][epoch_min])
    print('MIN ERR gradu:', history.history['L2re_grad_u'][epoch_min])
    print('#### VALIDATION ####')
    print('MIN LOSS u: ', history.history['val_loss_u'][epoch_min])
    print('MIN ERR u:', history.history['val_L2re_u'][epoch_min])
    print('MIN ERR gradu:', history.history['val_L2re_grad_u'][epoch_min])

    _ = get_model_performance(model.test_step, (x_train, y_train), (x_val, y_val), print_results=True)

    epochs_LBFGS = 150
    LBFGS_history = model.train_LBFGS(epochs_LBFGS) 
    LBFGS_history = model.train_LBFGS_LS(epochs_LBFGS)   
    
    df_history_LBFGS = pd.DataFrame(LBFGS_history, columns=list(LBFGS_history.keys()))
    df_history_LBFGS.to_csv(f'experiments/L3+Lder_LS_L-BFGS.csv', index=False)
    # df_history_LBFGS = pd.read_csv(f'experiments/L1+Lder_noLS_L-BFGS.csv')
    
    plot6 = pp.plot_loss(df_history_LBFGS)
    plot7 = pp.plot_partialLosses_train(df_history_LBFGS)
    plot8 = pp.plot_partialLosses_val(df_history_LBFGS)
    plot9 = pp.plot_L2re_u(df_history_LBFGS, ymin = 0, ymax = 50)
    plot10 = pp.plot_L2re_gradu(df_history_LBFGS, ymin = 0, ymax = 45)

    model.save_weights('experiments/L3+Lder_LS_weights_LBFGS.h5')
    