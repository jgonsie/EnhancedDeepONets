#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 20:32:26 2025

@author: jesusglezs97
"""
import tensorflow as tf
import keras as keras
import numpy as np
from scipy.stats import lognorm, loguniform

def extract_batch(x_train, y_train, initial, size):
    '''Extracts a batch of data from (x_train, y_train)'''
    
    keys = list(x_train.keys())
    samples = x_train[keys[0]].shape[0]
    indices = tf.range(start=0, limit=samples, dtype=tf.int32)

    indices_batch = indices[initial:initial+size]
        
    x_batch = {k:tf.gather(v, indices_batch, axis=0) for k,v in x_train.items()}
    y_batch = {k:tf.gather(v, indices_batch, axis=0) for k,v in y_train.items()}    
    
    return x_batch, y_batch

def prepare_raw_data_percentage(data, train_split = 0.8, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacUx(T)', 'jacUy(T)']):
    ''' Divides the raw data into training and validation sets'''
    
    samples, _ = data[outputs[0]].shape
    indices = tf.range(start=0, limit=samples, dtype=tf.int32)
    keras.utils.set_random_seed(1234)
    shuffled_indices = tf.random.shuffle(indices, seed=42)
    # shuffled_indices = indices
    
    training_ind = shuffled_indices[:int(samples*train_split)]
    validation_ind = shuffled_indices[int(samples*train_split):]
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val
    
def prepare_raw_data_items(data, train_split = 20, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacMu1(T)', 'jacMu2(T)', 'jacV(T)']):
    ''' Divides the raw data into training and validation sets'''
    
    samples, _ = data[outputs[0]].shape
    indices = tf.range(start=0, limit=samples, dtype=tf.int32)
    keras.utils.set_random_seed(1234)
    shuffled_indices = tf.random.shuffle(indices, seed=420)
    # shuffled_indices = indices
    
    training_ind = shuffled_indices[:int(train_split)]
    validation_ind = shuffled_indices[int(train_split):]
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val

def prepare_raw_data_items_esp(data, train_split = 20, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacMu1(T)', 'jacMu2(T)', 'jacV(T)']):
    ''' Divides the raw data into training (as spread as possible) and 
    validation sets '''
        
    if 'DT' not in data.keys():
        DTs = data['DT1']
    else:
        DTs = data['DT']
        
    # Take log10 of data for better spread across orders of magnitude
    log_data = np.log(DTs)
    
    # Create X bins over the log space
    bins = np.linspace(np.min(log_data), np.max(log_data), train_split + 1)
    
    # Compute bins centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    selected = []
    for center in bin_centers:
        # Convert centers back to linear space
        target = np.exp(center)
        # Find the closest point in the dataset
        closest = DTs[np.argmin(np.abs(DTs - target))]
        selected.append(closest)
    
    # Convert to numpy array
    selected = np.array(selected)
    
    # Get indices of selected training elements
    mask = np.isin(DTs, selected)
    training_ind = np.where(mask)[0]
    validation_ind = np.where(~mask)[0]
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val

def prepare_raw_data_items_esp2(data, train_split = 20, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacMu1(T)', 'jacMu2(T)', 'jacV(T)']):
    ''' Divides the raw data into training (as spread as possible) and 
    validation sets '''
        
    if 'DT' not in data.keys():
        DTs = data['DT1']
    else:
        DTs = data['DT']
        
    # Take log10 of data for better spread across orders of magnitude
    log_data = np.log(DTs)
    
    # Create X bins over the log space
    bins = np.linspace(np.min(log_data), np.max(log_data), train_split + 1)
    
    # Compute bins centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    selected = []
    for center in bin_centers:
        # Convert centers back to linear space
        target = np.exp(center)
        
        if center == bin_centers[0]:
            selected.append(DTs[0])
        elif center == bin_centers[-1]:
            selected.append(DTs[-1])
        else:
            # Find the closest point in the dataset
            closest = DTs[np.argmin(np.abs(DTs - target))]
            selected.append(closest)
    
    # Convert to numpy array
    selected = np.array(selected)
    
    # Get indices of selected training elements
    mask = np.isin(DTs, selected)
    training_ind = np.where(mask)[0]
    validation_ind = np.where(~mask)[0]
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val

def prepare_raw_data_custom(data, training_ind = [2,8,14], outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacUx(T)', 'jacUy(T)']):
    ''' Divides the raw data into training and validation sets'''
    
    samples, _ = data[outputs[0]].shape
    indices = tf.range(start=0, limit=samples, dtype=tf.int32)
    
    # Find the indices to keep (those NOT in train_points)
    mask = ~tf.reduce_any(tf.equal(indices[:, None], training_ind), axis=1)
    validation_ind = tf.boolean_mask(indices, mask)
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val

def DT_lognorm_dist(sigma, mu, samples, low_bound=0.01, up_bound=15):
    ''' Generates random samples from a log-norm distribution 
    sigma: spread
    mu: center in log space
    '''
    # Create the distribution object, truncated between [0.01, 15]
    # scale = np.exp(mu)
    scale=mu
    dist = lognorm(s=sigma, scale=scale)
    
    # Sample values within range
    realizations = []
    while len(realizations) < samples:
        x = dist.rvs()
        if low_bound <= x <= up_bound:
            realizations.append(x)
    realizations = np.array(realizations)
    
    return np.sort(realizations)

def DT_loguniform_dist(samples, low_bound=0.01, up_bound=15):
    ''' Generates random samples from a log-uniform distribution 
    '''
    # Create the distribution object
    dist = loguniform(low_bound, up_bound)
    
    # Sample realizations
    x = dist.rvs(samples)
    realizations = np.array(x)
    
    return np.sort(realizations)

def generate_loss_weights(y_train):
    '''Generates the loss weights based on the relation between the L2 norms of the data'''
    
    loss_weights = {}
    
    for k in y_train.keys():
        if k == 'T':
            loss_weights['loss_u'] = tf.constant((1.), dtype=y_train['T'].dtype)
        elif k == 'grad(T)_x':
            loss_weights['loss_gradu_x'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['grad(T)_x'])))     
        elif k == 'grad(T)_y': 
            loss_weights['loss_gradu_y'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['grad(T)_y']))) 
        elif k == 'jacMu(T)':
            loss_weights['loss_gradu_mu'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['jacMu(T)'])))     
        elif k == 'jacMu1(T)':
            loss_weights['loss_gradu_mu1'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['jacMu1(T)'])))     
        elif k == 'jacMu2(T)':
            loss_weights['loss_gradu_mu2'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['jacMu2(T)'])))     
        elif k == 'jacV(T)':
            loss_weights['loss_gradu_v'] = tf.sqrt(tf.reduce_sum(tf.square(y_train['T']))) / tf.sqrt(tf.reduce_sum(tf.square(y_train['jacV(T)'])))     

    return loss_weights


def get_model_performance(test_step, data_training, data_validation, print_results=False):
    '''Returns the losses and errors of the model inside the training and validation
    distribution for the new weights'''
    
    (x_train, y_train) = data_training
    (x_val, y_val) = data_validation
    
    metrics_tr = test_step((x_train, y_train))
    metrics_val = test_step((x_val, y_val))
    metrics_val = {'val_'+k:v for k,v in metrics_val.items()}
    
    if print_results == True:
        loss_tr = metrics_tr['loss_u']
        err_tr_u = metrics_tr['L2re_u']
        err_tr_grad_u = metrics_tr['L2re_grad_u']
        loss_val = metrics_val['val_loss_u']
        err_val_u = metrics_val['val_L2re_u']
        err_val_grad_u = metrics_val['val_L2re_grad_u']
        print('·····TRAINING DATA RESULTS·····')
        print(f'Loss u: {loss_tr:.3e}')
        print(f'Error u: {err_tr_u:.2f}')
        print(f'Error grad_u: {err_tr_grad_u:.2f}')
        print('·····VALIDATION DATA RESULTS·····')
        print(f'Loss u: {loss_val:.3e}')
        print(f'Error u: {err_val_u:.2f}')
        print(f'Error grad_u: {err_val_grad_u:.2f}')
    
    return {**metrics_tr, **metrics_val}

def fix_random(n):
    '''Fixes random seeds for reproducibility'''
    
    np.random.seed(n)
    keras.utils.set_random_seed(n)
    return

def fix_precission(double_precission = True):
    
    if double_precission:
        dtype = 'float64'
    else:
        dtype = 'float32'
        
    keras.backend.set_floatx(dtype)
    return dtype