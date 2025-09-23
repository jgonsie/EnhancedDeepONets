#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:05:15 2025

@author: jgonzalez
"""

import os
import keras
import tensorflow as tf
import pandas as pd
import argparse

from SRC.utils import fix_random, fix_precission
from SRC.utils import prepare_raw_data_items_esp, generate_loss_weights
from SRC.models import DeepONet, NeuralOperatorModel
from SRC import foamRW as fRW
from SRC.grids2D import Grid

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str)
parser.add_argument("--npoints", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


if __name__ == "__main__":
    
    loss_function = args.loss
    npoints = args.npoints
    seed = args.seed
    
    fix_random(1234)
    dtype = fix_precission(double_precission = True)

    # Create grid
    grid = Grid(size_x = 50, size_y = 50, step_size = 1/50)

    # Data for training and validation
    data_route = '../OpenFOAM/experiment2/training_data/'
    training_data = fRW.upload_training_data(data_route, experiment='exp2', dtype=dtype)
    x_train, y_train, x_val, y_val = prepare_raw_data_items_esp(training_data, train_split=npoints)

    # Create learning model
    net = DeepONet(layers_branch=[25,25,40], layers_trunk=[25,25,40], 
                   experiment='exp2', dimension='2D', seed=seed, dtypeid=dtype)
                    
    loss_weights = generate_loss_weights(y_train)
    
    # Select model
    model = NeuralOperatorModel(net, grid, loss_function, kappas = loss_weights,
                                LS=False)

    sim_name = f'{loss_function}_LS-False_seed-{seed}_{npoints}'
    sim_dir = '../results/exp2/1.training_points_init/'+sim_name
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  jit_compile=False)
                  
    #Initialize model
    result = model(x_train) 

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=sim_dir+'_best.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=True,  # Save only the model weights
            verbose=0)
    
    nanchecker = keras.callbacks.TerminateOnNaN()
    
    nbatchval = int((200-npoints) / 5)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=10000, batch_size=npoints, validation_batch_size=nbatchval,
                        callbacks=[checkpoint, nanchecker])

    df_history = pd.DataFrame(history.history)
    df_history.to_csv(sim_dir+'_history.csv', index=False)
    
    model.save_weights(sim_dir+'_final.weights.h5')