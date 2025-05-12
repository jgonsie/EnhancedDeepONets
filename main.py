#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:29:38 2025

@author: jesusglezs97
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from SRC.utils import DT_lognorm_dist, prepare_raw_data_percentage, prepare_raw_data_items, prepare_raw_data_custom, prepare_raw_data_items_esp, get_model_performance, extract_batch
from SRC.models import DeepONet, NeuralOperatorModel
from SRC import foamRW as fRW
from SRC.grids2D import Grid
from SRC import postprocessing as pp

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras as keras
dtype = 'float32'
keras.backend.set_floatx(dtype)

# Random seeds for being deterministic.
keras.utils.set_random_seed(1234)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# tf.config.experimental.get_memory_info("GPU:0")['current']

if __name__ == "__main__":
    
    # Generate random samples following a log-normal distribution
    # samples = DT_lognorm_dist(1.0, -1.0, 200)
    # np.savetxt('exp1_randomDT.dat', samples, fmt='%.9f')
    
    # Create grid
    grid = Grid(size_x = 50, size_y = 50, step_size = 1/50)

    # Data for training and validation
    data_route = './OpenFOAM/experiment1/training_data/'
    training_data = fRW.upload_training_data(data_route, jacobian=True,dtype='float32')
    # x_train, y_train, x_val, y_val = prepare_raw_data_percentage(training_data, train_split=0.1)
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=10)
    # x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[4,14,25])
    # x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[1,9,25])
    plot0 = pp.plot_data_distribution(x_train, y_train, x_val, y_val)
    
    # Plot training distribution 1
    pure_conv_route = './OpenFOAM/experiment1/pure_convection'
    pure_conv_case = fRW.upload_single_data(pure_conv_route, jacobian=False)
    pure_diff_route = './OpenFOAM/experiment1/pure_diffusion'
    pure_diff_case = fRW.upload_single_data(pure_diff_route, jacobian=False)
    E_conv = np.sum(pure_conv_case['T']**2) * (grid.step**2)
    E_diff = np.sum(pure_diff_case['T']**2) * (grid.step**2)
    E_training = np.sum(training_data['T']**2, axis=1) * (grid.step**2)
    plot00 = pp.plot_log_data_distribution(training_data['DT'], E_training, E_conv, E_diff)
 
    # Plot training distribution 2
    pure_conv_route = './OpenFOAM/experiment1/pure_convection'
    pure_conv_case = fRW.upload_single_data(pure_conv_route, jacobian=False)
    pure_diff_route = './OpenFOAM/experiment1/pure_diffusion'
    pure_diff_case = fRW.upload_single_data(pure_diff_route, jacobian=False)
    E_conv = np.sum(pure_conv_case['T']**2) * (grid.step**2)
    E_diff = np.sum(pure_diff_case['T']**2) * (grid.step**2)
    E_tr = np.sum(y_train['T']**2, axis=1) * (grid.step**2)
    E_val = np.sum(y_val['T']**2, axis=1) * (grid.step**2)
    plot00 = pp.plot_log_data_distribution2(x_train['DT'], E_tr, x_val['DT'], E_val,E_conv, E_diff, name='u')

    E_conv = np.sum(pure_conv_case['grad(T)_x']**2+ pure_conv_case['grad(T)_y']**2) * (grid.step**2)
    E_diff = np.sum(pure_diff_case['grad(T)_x']**2+pure_diff_case['grad(T)_y']**2) * (grid.step**2)
    E_tr = np.sum(y_train['grad(T)_x']**2+y_train['grad(T)_y']**2, axis=1) * (grid.step**2)
    E_val = np.sum(y_val['grad(T)_x']**2+y_val['grad(T)_y']**2, axis=1) * (grid.step**2)
    plot00 = pp.plot_log_data_distribution2(x_train['DT'], E_tr, x_val['DT'], E_val,E_conv, E_diff, name='gradu')


    # Create learning model
    net = DeepONet(layers_branch=[50,70], layers_trunk=[50,50,50,70], 
                   dimension='2D', seed=420, dtypeid=tf.float32)
                    

    # Select model
    # model = NeuralOperatorModel(net, grid, 'vanilla', LS=False, regularizer = 10**(-1))
    # model = NeuralOperatorModel(net, grid, 'H1', LS=False)
    # model = NeuralOperatorModel(net, grid, 'phy', LS=True)
    # model = NeuralOperatorModel(net, grid, 'van+der',  LS=False, regularizer = 10**(-2))
    model = NeuralOperatorModel(net, grid, 'H1+der',  LS=False, regularizer = 10**(-2))
    # model = NeuralOperatorModel(net, grid, 'phy+der',  LS=True)

    sim_name = 'H1+der_LS10-2'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=True)#, run_eagerly=True)
                  
    #Initialize model
    result = model(x_train) 

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'results/exp1/{sim_name}_weights_GD3.weights.h5',
        monitor='loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,  # Save only the model weights
        verbose=0)
    # nanchecker = keras.callbacks.TerminateOnNaN()
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=8000, batch_size=10, validation_batch_size=38,
                        callbacks=[])

    df_history = pd.DataFrame(history.history)
    plot1 = pp.plot_loss(df_history)
    plot2 = pp.plot_partialLosses_train(df_history)
    plot3 = pp.plot_partialLosses_val(df_history)
    plot4 = pp.plot_L2re_u(df_history, ymin = 0, ymax = 100)
    plot5 = pp.plot_L2re_gradu(df_history, ymin = 0, ymax = 100)
    # plt.savefig('f'results/exp1/{sim_name}.png', dpi=200, bbox_inches='tight')

    df_history.to_csv(f'results/exp1/{sim_name}GPU.csv', index=False)
    # df_history = pd.read_csv('f'experiments/L1+Lder_training.csv')

    # Load weights with best performance
    model.load_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')
    # model.save_weights(f'results/exp1/{sim_name}_weights_GD32.weights.h5')
    
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
    
    result = model(x_train) 
    item = 9
    f1 = pp.plot_field(result['T'][item,:,:], grid, 'DeepONet approx.', None, None)
    f2 = pp.plot_field(y_train['T'][item,:,:], grid, 'FVM solution', None, None)
    
    trainable_vars = [weight.value for weight in model.net.trainable_weights[:-1]]
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(trainable_vars)
        
        loss = model.evaluate_loss((x_train,y_train))
    
    grad = tape.gradient(loss, trainable_vars)
    
    # Generate spatial evaluation points
    batch_size = 10
    integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                    for k,v in model.integration_points.items()}
    
    # Prepare data for DeepONet
    x_curated = {**x_train, **model.integration_points}
    
    A, b = model.system(x_curated, y_train)
    A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
    b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
    alpha_new = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-4))

    x_batch, y_batch = extract_batch(x_val, y_val, initial=151, size=38)
    
    def L2_rel_err(y_true, y_pred):
        num = tf.sqrt(tf.reduce_sum(tf.square(y_true-y_pred), axis=[-1,-2]))
        den = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=[-1,-2]))
        L2RE = 100 * tf.reduce_mean(num/(den+1e-12))
        return L2RE
    
    tf.profiler.experimental.start('logs/')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=4, batch_size=10, validation_batch_size=38,
                        callbacks=[])
    tf.profiler.experimental.stop()
