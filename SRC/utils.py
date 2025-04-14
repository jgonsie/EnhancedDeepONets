#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 20:32:26 2025

@author: jesusglezs97
"""
import tensorflow as tf
import keras as keras
import numpy as np
from scipy.stats import lognorm

def prepare_raw_data(data, train_split = 0.8, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacUx(T)', 'jacUy(T)']):
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

def get_model_performance(test_step, x_train, y_train, x_val, y_val, print_results=False):
    '''Returns the losses and errors of the model inside the training and validation
    distribution for the new weights'''
    
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