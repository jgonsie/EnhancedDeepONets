#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 16 17:13:06 2024

@author: jesusglezs97
"""
import os
import numpy as np
import tensorflow as tf
import foamRW as fRW
import grids2D
import pandas as pd
import matplotlib.pyplot as plt
import postprocessing as pp
from tfp_optimizer import lbfgs_minimize

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras as keras
keras.backend.set_floatx('float64')
# Random seeds for being deterministic.
keras.utils.set_random_seed(1234)

import psutil
import gc 


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


class DeepONet(keras.Model):
    def __init__(self, layers_branch_v, layers_branch_mu, layers_trunk, 
                 num_rand_sampling = 100, dimension = '1D', **kwargs):
        
        assert layers_branch_v[-1]==layers_branch_mu[-1], f'The size of the last layer of the branches must be equal: {layers_branch_v[-1]} vs {layers_branch_mu[-1]}'
        assert layers_branch_v[-1]==layers_trunk[-1], f'The size of the last layer of the branch and trunk must be equal: {layers_branch_v[-1]} vs {layers_trunk[-1]}'
        
        super(DeepONet, self).__init__()
        
        self.num_basis_func = layers_trunk[-1]
        self.num_rand_sampling = num_rand_sampling
        self.fields_br_v, self.fields_br_mu, self.fields_tr = self.fields_by_dimension(dimension)
        self.dim = 1 if dimension == '1D' else 2
        
        # Random seeds for being deterministic.
        init = tf.keras.initializers.GlorotUniform(seed=42)
        
        # Create branch net for velocity
        # br_v_layers = [keras.layers.Dense(units=layer, activation="tanh", 
        #                                   use_bias=True) for layer in layers_branch_v[:-1]]
        # br_v_layers.append(keras.layers.Dense(units=layers_branch_v[-1], 
        #                                       activation="linear", 
        #                                       use_bias=True))
        # br_v_layers.insert(0, keras.Input(shape=(self.num_rand_sampling,2,)))
        # br_v_layers.insert(1, keras.layers.Flatten())
        # self.branch_v = keras.Sequential(br_v_layers, name='Branch_v')
        
        # Create branch net for mu
        br_mu_layers = [keras.layers.Dense(units=layer, activation="tanh", 
                                            use_bias=True, kernel_initializer=init) for layer in layers_branch_mu]
        self.branch_mu = keras.Sequential(br_mu_layers, name='Branch_mu')
        
        # Create trunk
        tru_layers = [keras.layers.Dense(units=layer, activation="tanh", 
                                         use_bias=True, kernel_initializer=init) for layer in layers_trunk]
        self.trunk = keras.Sequential(layers=tru_layers, name='Trunk')
        
        # Linear layer
        self.linear_layer = keras.layers.Dense(units=1, activation=None, 
                                               use_bias=False, name='linear')
        
    def fields_by_dimension(self, dimension):
        '''Selects the involved fields depending on the dimension'''
        
        fields_brv = ['v_x', 'v_y']
        fields_brmu = ['DT']
        fields_tr = ['coord_x', 'coord_y']
        
        if dimension == '1D':
            fields_brv = {k for k in fields_brv if '_y' not in k}
            fields_brmu = fields_brmu
            fields_tr = {k for k in fields_tr if '_y' not in k}
        elif dimension == '2D':
            fields_brv = fields_brv
            fields_brmu = fields_brmu
            fields_tr = fields_tr
        else:
            ValueError('Dimension not permitted')
        
        return fields_brv, fields_brmu, fields_tr
            
    def sort_and_reshape_state(self, state, input_keys, reshape=False):
        ''' Sorts the input data given the order in self.fields_watched and 
        reshapes the data'''
        
        if reshape == True:
            state_watched = tf.reshape(tf.concat([state[k] for k in input_keys], axis = -1), [-1, len(input_keys)])
        else:
            state_watched = tf.concat([state[k] for k in input_keys], axis=-1)
        
        return state_watched
    
    def construct_matrix(self, inputs):
        '''Construct all the matrix obtained previously to the linear layer 
        application'''
        
        dim_input_tr = inputs['coord_x'].shape.as_list()
        
        # x_brv = self.sort_and_reshape_state(inputs, self.fields_br_v) #(b,s,2)
        # # print('Input shape brv: ', x_brv.shape)
        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(x_brv)
        #     basis_brv = self.branch_v(x_brv)
        #     # basis = basis*(x_watched-0) #To impose boundary conditions
        #     # basis = basis*(x_watched-1)
        #     y_pred = self.linear_layer(basis_brv) #Necessary to initialize the layer
        # dbasis_brv = tf.squeeze(tape.batch_jacobian(basis_brv,x_brv))
        # # print('Output shape basis brv: ', basis_brv.shape)
        # # print('Output shape dbasis brv: ', dbasis_brv.shape)
        # del tape
        
        x_brmu = self.sort_and_reshape_state(inputs, self.fields_br_mu) #(b,1)
        # print('Input shape brmu: ', x_brmu.shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_brmu)
            basis_brmu = self.branch_mu(x_brmu)
            
        dbasis_brmu = tape.batch_jacobian(basis_brmu,x_brmu)
        dbasis_brmu = tf.squeeze(dbasis_brmu, axis=-1)
        # print('Output shape basis brmu: ', basis_brmu.shape)
        # print('Output shape dbasis brmu: ', dbasis_brmu.shape)
        del tape

        # y_pred = self.linear_layer(basis_brmu) #Necessary to initialize the layer
        
        x_tr = self.sort_and_reshape_state(inputs, self.fields_tr, reshape=True) #(b*e,2)
        # print('Input shape tr: ', x_tr.shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_tr)
            basis_tr = self.trunk(x_tr) #(b*e,j)

        dbasis_tr = tape.batch_jacobian(basis_tr,x_tr) #(b*e,j,2)
        basis_tr = tf.reshape(basis_tr, dim_input_tr[:-1]+[-1])
        dbasis_tr = tf.reshape(dbasis_tr, dim_input_tr[:-1]+dbasis_tr.shape.as_list()[1:])

        # print('Output shape basis tr: ', basis_tr.shape)
        # print('Output shape dbasis tr: ', dbasis_tr.shape)
        del tape

        # basis_brmu = tf.ones([dim_input_tr[0],self.num_basis_func], dtype='float64')
        # dbasis_brmu = tf.ones([dim_input_tr[0],self.num_basis_func,20], dtype='float64')
        basis_brv = tf.ones([dim_input_tr[0],self.num_basis_func], dtype='float64')
        dbasis_brv = tf.ones([dim_input_tr[0],self.num_basis_func,2], dtype='float64')


        return (basis_brv, basis_brmu, basis_tr, dbasis_brv, dbasis_brmu, dbasis_tr)
    
    
    # def construct_matrix_residual(self, inputs):
    #     dim_input = inputs['coord_x'].shape.as_list()
    #     x_watched, x_unwatched = self.classify_and_reshape_state(inputs)
        
    #     with tf.GradientTape(watch_accessed_variables=False) as tape_outer:
    #         tape_outer.watch(x_watched)
    #         with tf.GradientTape(watch_accessed_variables=False) as tape_inner:
    #             tape_inner.watch(x_watched)
    #             x = tf.concat([x_watched, x_unwatched], axis=1)
    #             basis = self.network(x)
    #             # basis = basis*(x_watched-0) #To impose boundary conditions
    #             # basis = basis*(x_watched-1)
    #             y_pred = self.linear_layer(basis) #Necessary to initialize the layer
    #         dbasis = tf.squeeze(tape_inner.batch_jacobian(basis,x_watched))
    #     ddbasis = tf.squeeze(tape_outer.batch_jacobian(dbasis,x_watched))
        
    #     basis = tf.reshape(basis, dim_input+[-1])
    #     dbasis = tf.reshape(dbasis, dim_input+dbasis.shape.as_list()[1:])
    #     ddbasis = tf.reshape(ddbasis, dim_input+ddbasis.shape.as_list()[1:])

    #     return (basis,dbasis,ddbasis)
    
    def call(self, inputs):
        '''Applies the linear layer to both the basis and the dbasis'''
        
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.construct_matrix(inputs)

        basis = tf.einsum('bj,bj,bej->bej', Umatrix_v, Umatrix_mu, Umatrix_x)
        db_vx = tf.einsum('bj,bj,bej->bej', dUmatrix_v[:,:,0], Umatrix_mu, Umatrix_x)
        db_vy = tf.einsum('bj,bj,bej->bej', dUmatrix_v[:,:,1], Umatrix_mu, Umatrix_x)
        db_mu = tf.einsum('bj,bj,bej->bej', Umatrix_v, dUmatrix_mu, Umatrix_x)
        db_x = tf.einsum('bj,bj,bej->bej', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,0])
        db_y = tf.einsum('bj,bj,bej->bej', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,1])

        U = self.linear_layer(basis)
        dU_vx = self.linear_layer(db_vx)
        dU_vy = self.linear_layer(db_vy)
        dU_mu = self.linear_layer(db_mu)
        dU_x = self.linear_layer(db_x)
        dU_y = self.linear_layer(db_y)
        
        result = {'T': U,
                  'grad(T)_x': dU_x,
                  'grad(T)_y': dU_y,
                  'jacUx(T)': dU_vx,
                  'jacUy(T)': dU_vy,
                  'jacMu(T)': dU_mu}
        
        return result
    
    
class my_model(keras.Model):
    
    def __init__(self, net, grid, system_constructor, quadrature = 'centroids',
                 LS = True, **kwargs):
        
        super(my_model, self).__init__()
        self.net = net
        self.grid = grid
        self.quadrature = tf.constant(quadrature, dtype=tf.string)
        self.n_points_quad = grid.n_points_by_method[quadrature]
        self.LS_activation = LS
                
        if system_constructor == 'vanilla':
            self.system = self.construct_LS_vanilla
            self.keys_phy = ['loss_u']
            self.keys_der = []
        elif system_constructor == 'physics_0':
            self.system = self.construct_LS_physics_0
            self.keys_phy = ['loss_u', 'loss_gradu_x', 'loss_gradu_y']
            self.keys_der = []
        # elif system_constructor == 'physics_1':
        #     self.system = self.construct_LS_physics_1
        #     self.keys_phy = ['loss_u', 'loss_vgradu_x', 'loss_vgradu_y',
        #                      'loss_mugradu_x', 'loss_mugradu_y']
        #     self.keys_der = []
        elif system_constructor == 'physics_2':
            self.system = self.construct_LS_physics_2
            self.keys_phy = ['loss_u', 'loss_vgradu', 'loss_mugradu']
            self.keys_der = []
        elif system_constructor == 'van+der':
            self.system = self.construct_LS_vander
            self.keys_phy = ['loss_u']
            self.keys_der = ['loss_gradu_mu']
        elif system_constructor == 'phy0+der':
            self.system = self.construct_LS_phy0der
            self.keys_phy = ['loss_u', 'loss_gradu_x', 'loss_gradu_y']
            self.keys_der = ['loss_gradu_mu']
        elif system_constructor == 'phy2+der':
            self.system = self.construct_LS_phy2der
            self.keys_phy = ['loss_u', 'loss_vgradu', 'loss_mugradu']
            self.keys_der = ['loss_gradu_mu']
        elif system_constructor == 'complete':
            self.system = self.construct_LS_complete
            self.keys_phy = ['loss_u', 'loss_vgradu', 'loss_mugradu']
            self.keys_der = ['loss_gradu_vx','loss_gradu_vy', 'loss_gradu_mu']
        elif system_constructor == 'test':
            self.system = self.construct_LS_test
            self.keys_phy = ['loss_u', 'loss_gradu_x', 'loss_gradu_y']
            self.keys_der = ['loss_gradu_mu']
        else:
            ValueError(f'Loss selected not found: {system_constructor}')
          
        # Initialize branch and trunk nets
        self.net.trunk.build(input_shape=(None, self.net.dim))
        self.net.branch_mu.build(input_shape=(None, 1))
        # self.net.branch_v.build(input_shape=(None, self.net.num_rand_sampling,self.net.dim))
        self.net.linear_layer.build(input_shape=(None, self.net.num_basis_func))
        
        # Generate random sampling of points
        self.random_sampling = self.grid.generate_random_sampling_of_points(self.net.num_rand_sampling)
      
        # Generate spatial evaluation points
        self.integration_points, _ = self.points_and_weights(1)
        
    def points_and_weights(self, batch_size):
        '''Generates new integration points and weights'''
        
        # Generate integration points and weights
        outputs = tf.numpy_function(
            func = grid.generate_quadrature_tf,
            inp  = [self.quadrature, tf.constant(batch_size), 
                    tf.constant(self.n_points_quad)], # Inputs as TensorFlow tensors
            Tout = [tf.float64, tf.float64, tf.float64]  # Data types of the outputs
            )
        
        points = {'coord_x': tf.ensure_shape(outputs[0], (batch_size, self.grid.ncells, self.n_points_quad)), 
                  'coord_y': tf.ensure_shape(outputs[1], (batch_size, self.grid.ncells, self.n_points_quad))}
        weights = tf.ensure_shape(outputs[2], (batch_size, self.grid.ncells, self.n_points_quad))
        
        return points, weights
    
    def construct_LS_vanilla(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the vanilla DeepONets'''

        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
               
        Amatrix = Umatrix #LHS of the LS system
        bvector = y_dict['T'] #RHS of the LS system
  
        return Amatrix, bvector
   
    def construct_LS_physics_0(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
       
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, dUmatrix_x, dUmatrix_y], axis=1)
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], y_dict['grad(T)_x'], y_dict['grad(T)_y']], axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_physics_1(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix_x, vdUmatrix_y, mudUmatrix_x, mudUmatrix_y], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        mudUmatrix_x_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_x_fv, vdUmatrix_y_fv, mudUmatrix_x_fv, mudUmatrix_y_fv], axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_physics_2(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bp,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bp,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y))
        #TODO: aqui podria utilizar tf.norm() para calcular la norma

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix, mudUmatrix], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bp,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bp,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv], axis=1)
        
        return Amatrix, Bvector

    def construct_LS_test(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        # dUmatrix_vx = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,0], coeffs_mu, basis_x)
        # dUmatrix_vy = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,1], coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bjs,bej->besj', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])

        
        # Build the terms for the loss
        # vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        # vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        # mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        # mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        dUmatrix_mu_unstacked = tf.unstack(dUmatrix_mu, axis=2) #[(b,e,j),...,(b,e,j)]
        
        # LHS of the LS system
        Alist = [Umatrix, dUmatrix_x, dUmatrix_y]+dUmatrix_mu_unstacked
        # Alist = [Umatrix]+dUmatrix_mu_unstacked
        Amatrix = tf.concat(Alist, axis=1)
        
        # Build data terms of the loss function
        # vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        # vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        # mudUmatrix_x_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        # mudUmatrix_y_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        dUmatrix_mu_unstacked_fv = tf.unstack(y_dict['jacMu(T)'], axis=2)
        
        # RHS of the LS system
        Blist = [y_dict['T'], y_dict['grad(T)_x'], y_dict['grad(T)_y']] + dUmatrix_mu_unstacked_fv
        # Blist = [y_dict['T']] + dUmatrix_mu_unstacked_fv
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_vander(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bj,bej->bej', coeffs_v, dcoeffs_mu, basis_x) 
                
        # LHS of the LS system
        Alist = [Umatrix, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # RHS of the LS system
        Blist = [y_dict['T'], y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_phy0der(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bj,bej->bej', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # LHS of the LS system
        Alist = [Umatrix, dUmatrix_x, dUmatrix_y, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # RHS of the LS system
        Blist = [y_dict['T'], y_dict['grad(T)_x'], y_dict['grad(T)_y'], y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_complete_mu_sum(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        # dUmatrix_vx = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,0], coeffs_mu, basis_x)
        # dUmatrix_vy = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,1], coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bjs,bej->besj', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y))
        # dUmatrix_vx_sum = tf.reduce_sum(dUmatrix_vx, axis=2)
        # dUmatrix_vy_sum = tf.reduce_sum(dUmatrix_vy, axis=2)
        dUmatrix_mu_sum = tf.reduce_sum(tf.abs(dUmatrix_mu), axis=2)
        
        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix, mudUmatrix, dUmatrix_mu_sum], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        # dUmatrix_x_fv = tf.gather(y_dict['jacUx(T)'], self.random_sampling, axis=2)
        # dUmatrix_y_fv = tf.gather(y_dict['jacUy(T)'], self.random_sampling, axis=2)
        # dUmatrix_mu_fv = tf.gather(y_dict['jacMu(T)'], self.random_sampling, axis=2)
        # dUmatrix_x_fv_sum = tf.reduce_sum(dUmatrix_x_fv, axis=2)
        # dUmatrix_y_fv_sum = tf.reduce_sum(dUmatrix_y_fv, axis=2)
        # dUmatrix_mu_fv_sum = tf.reduce_sum(dUmatrix_mu_fv, axis=2)
        
        # RHS of the LS system
        # Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, dUmatrix_x_fv_sum, dUmatrix_y_fv_sum, dUmatrix_mu_fv_sum], axis=1)
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, y_dict['jacMu(T)']], axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_complete_mu_matrix_direct(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_vx = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,0], coeffs_mu, basis_x)
        dUmatrix_vy = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,1], coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bjs,bej->besj', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Linear layer
        Umatrix = self.net.linear_layer(Umatrix)
        dUmatrix_vx = self.net.linear_layer(dUmatrix_vx)
        dUmatrix_vy = self.net.linear_layer(dUmatrix_vy)
        dUmatrix_mu = self.net.linear_layer(dUmatrix_mu)
        dUmatrix_x = self.net.linear_layer(dUmatrix_x)
        dUmatrix_y = self.net.linear_layer(dUmatrix_y)
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y))
        # dUmatrix_vx_sum = tf.reduce_sum(dUmatrix_vx, axis=2)
        # dUmatrix_vy_sum = tf.reduce_sum(dUmatrix_vy, axis=2)
        # dUmatrix_mu_sum = tf.reduce_sum(tf.abs(dUmatrix_mu), axis=2)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        # dUmatrix_x_fv = tf.gather(y_dict['jacUx(T)'], self.random_sampling, axis=2)
        # dUmatrix_y_fv = tf.gather(y_dict['jacUy(T)'], self.random_sampling, axis=2)
        # dUmatrix_mu_fv = tf.gather(y_dict['jacMu(T)'], self.random_sampling, axis=2)
        # dUmatrix_x_fv_sum = tf.reduce_sum(dUmatrix_x_fv, axis=2)
        # dUmatrix_y_fv_sum = tf.reduce_sum(dUmatrix_y_fv, axis=2)
        # dUmatrix_mu_fv_sum = tf.reduce_sum(dUmatrix_mu_fv, axis=2)
        
        lossU = tf.reduce_mean(tf.square(Umatrix - y_dict['T']))
        lossvdU = tf.reduce_mean(tf.square(vdUmatrix - vdUmatrix_fv))
        lossmudU = tf.reduce_mean(tf.square(mudUmatrix - mudUmatrix_fv))
        lossdUdmu = tf.reduce_mean(tf.reduce_sum(tf.square(dUmatrix_mu - y_dict['jacMu(T)']),axis=[-2]))
               
        loss = lossU+lossvdU+lossmudU+lossdUdmu
        partialLosses = {'loss_u':lossU, 
                         'loss_vgradu':lossvdU,
                         'loss_mugradu':lossmudU, 
                         'loss_gradu_mu':lossdUdmu}
        
        return loss, partialLosses
    
    def construct_LS_phy2der(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bj,bej->bej', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y #(b,e,j)
        mudUmatrix_x = tf.einsum('bp,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bp,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y)) #(b,e,j)
        
        # LHS of the LS system
        Alist = [Umatrix, vdUmatrix, mudUmatrix, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bp,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bp,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        
        # RHS of the LS system
        Blist = [y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    
    def construct_LS_complete_old(self, x_dict, y_dict, weights):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        # Contruct the matrix for LS system
        coeffs_v, coeffs_mu, basis_x, dcoeffs_v, dcoeffs_mu, dbasis_x = self.net.construct_matrix(x_dict) 
       
        Umatrix = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, basis_x)
        dUmatrix_vx = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,0], coeffs_mu, basis_x)
        dUmatrix_vy = tf.einsum('bjs,bj,bej->besj', dcoeffs_v[:,:,1], coeffs_mu, basis_x)
        dUmatrix_mu = tf.einsum('bj,bjs,bej->besj', coeffs_v, dcoeffs_mu, basis_x)
        dUmatrix_x = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,0])
        dUmatrix_y = tf.einsum('bj,bj,bej->bej', coeffs_v, coeffs_mu, dbasis_x[:,:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y))
        dUmatrix_vx_sum = tf.reduce_sum(dUmatrix_vx, axis=2)
        dUmatrix_vy_sum = tf.reduce_sum(dUmatrix_vy, axis=2)
        dUmatrix_mu_sum = tf.reduce_sum(dUmatrix_mu, axis=2)
        
        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix, mudUmatrix, dUmatrix_vx_sum, dUmatrix_vy_sum, dUmatrix_mu_sum], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bep,bep->bep', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        # dUmatrix_x_fv = tf.gather(y_dict['jacUx(T)'], self.random_sampling, axis=2)
        # dUmatrix_y_fv = tf.gather(y_dict['jacUy(T)'], self.random_sampling, axis=2)
        # dUmatrix_mu_fv = tf.gather(y_dict['jacMu(T)'], self.random_sampling, axis=2)
        # dUmatrix_x_fv_sum = tf.reduce_sum(dUmatrix_x_fv, axis=2)
        # dUmatrix_y_fv_sum = tf.reduce_sum(dUmatrix_y_fv, axis=2)
        # dUmatrix_mu_fv_sum = tf.reduce_sum(dUmatrix_mu_fv, axis=2)
        
        # RHS of the LS system
        # Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, dUmatrix_x_fv_sum, dUmatrix_y_fv_sum, dUmatrix_mu_fv_sum], axis=1)
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, y_dict['jacUx(T)'], y_dict['jacUy(T)'], y_dict['jacMu(T)']], axis=1)
        
        return Amatrix, Bvector
        
    def resolve_LS(self, x_dict, y_dict):
        '''Resolves the Least-Squares system and updates the weights of the 
        linear layer'''

        # Construct LS system
        A, b = self.system(x_dict, y_dict) #(b,c*e,j) (b,c*e,1)

        # Solve LS system (op1)
        A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
        b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
        alpha_new = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-2))#10**(-3))
        # Update linear layer weights
        computable_vars = self.net.linear_layer.weights[0]
        computable_vars.assign(alpha_new)
        
        #--------------------------------------------------------------------------
        # # Solve LS system (op2)
        # # Construct LS system
        # A, b = self.system(x_dict, y_dict) #(b,c*e,j) (b,c*e,1)
        
        # A_flat = tf.reshape(A,[-1,A.shape[-1]])
        # b_flat = tf.reshape(b,[-1,b.shape[-1]])
        # LHS_original = tf.matmul(A_flat,A_flat, transpose_a=True)
        # RHS_original = tf.matmul(A_flat,b_flat, transpose_a=True)
        # alpha_new = tf.linalg.solve(LHS_original,RHS_original)
        # # Update linear layer weights
        # computable_vars = self.net.linear_layer.weights[0]
        # computable_vars.assign(alpha_new)
        
        #--------------------------------------------------------------------------
        
        # # Solve LS system (op3)
        # # Construct first and second derivative of the loss wrt alpha when alpha=0
        # self.net.linear_layer.weights[0].assign(tf.zeros_like(self.net.linear_layer.weights[0]))
        
        # with tf.GradientTape() as t1:
        #     with tf.GradientTape() as t2:
        #         A, b = self.system(x_dict, y_dict)
        #         unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        #         loss, partialLosses = self.weighted_loss(unweighted_loss)
                
        #     J = t2.gradient(loss,self.net.linear_layer.weights[0])# the RHS of the system: -2*B^t*l  
        # H = tf.squeeze(t1.jacobian(J,self.net.linear_layer.weights[0])) # the LHS of the system: 2*B^t*B
        
        # LHS_normal = 1/2*H
        # RHS_normal = -1/2*J
        
        # alpha_normal = tf.linalg.solve(LHS_normal,RHS_normal)
        # self.net.linear_layer.weights[0].assign(alpha_normal)
  
        return 
    
    def check_LS(self, x_dict, y_dict):
        
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points, integ_weights = self.points_and_weights(batch_size)
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        # Construct first and second derivative of the loss wrt alpha when alpha=0
        computable_vars = self.net.linear_layer.weights[0]
        computable_vars.assign(tf.zeros_like(computable_vars))
        
        with tf.GradientTape() as t1:
            with tf.GradientTape() as t2:
                A, b = self.system(x_curated, y_dict)
                unweighted_loss = tf.square(self.net.linear_layer(A)-b)
                loss, partialLosses = self.weighted_loss(unweighted_loss)
                
            J = t2.gradient(loss,computable_vars)# the RHS of the system: -2*B^t*l  
        H = tf.squeeze(t1.jacobian(J,computable_vars)) # the LHS of the system: 2*B^t*B
                
        A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
        b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
        
        tf.print('A shape:', A.shape)
        tf.print('A_flat shape:', A_flat.shape)
        tf.print('A_mean shape:', A_flat.shape)
        tf.print('b shape:', b.shape)
        tf.print('b_flat shape:', b_flat.shape)
        tf.print('b_mean shape:', b_flat.shape)
        tf.print('J shape:', J.shape)
        tf.print('H shape:', H.shape)
        
        LHS_normal = 1/2*H
        RHS_normal = -1/2*J
        
        LHS_original = tf.matmul(A_flat,A_flat, transpose_a=True)
        RHS_original = tf.matmul(A_flat,b_flat, transpose_a=True)
        
        weights_normal = tf.linalg.solve(LHS_normal,RHS_normal)
        # weights_original = tf.linalg.solve(LHS_original,RHS_original)
        weights_original = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-13))
        weights_david = -tf.matmul(tf.linalg.inv(H),J)
        tf.print('weights: ', weights_david)
        
        return LHS_original, RHS_original, weights_original, LHS_normal, RHS_normal, weights_normal
    
    
    def compute_kappas(self, x_dict, y_dict):
        ''' Computes the kappas for the weighting loss based on the ratios of the
        first epoch'''
        
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points, integ_weights = self.points_and_weights(batch_size)
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        # Run the model
        A, b = self.system(x_curated, y_dict, integ_weights)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        
        keys = ['loss_u', 'loss_vgradu', 'loss_mugradu', 'loss_gradu_vx', 
                'loss_gradu_vy', 'loss_gradu_mu']
        
        # Computation of the number of parcial loss terms involver
        items = unweighted_loss.shape[1]
        nL = int(items/self.grid.ncells)
        
        # Extract the partial losses from the general loss
        Lpartial = tf.split(unweighted_loss, num_or_size_splits=nL, axis=1)
        meanLpartial = {keys[n]:tf.reduce_mean(Lpartial[n]) for n in range(nL)}
        # meanL = tf.reduce_mean([v for k,v in meanLpartial.items()])
        
        # Computation of the weights or pondering factors
        # self.kappas = {k:tf.stop_gradient(meanL/(v + 1e-12)) for k,v in meanLpartial.items()}
        # self.kappas = {k:tf.stop_gradient(tf.reduce_max([v for v in meanLpartial.values()])/v) for k,v in meanLpartial.items()}
        # self.kappas = {'loss_u': tf.convert_to_tensor(([1.]), dtype=tf.float64), 
        #                'loss_vgradu': tf.convert_to_tensor(([1.]), dtype=tf.float64),
        #                'loss_mugradu': tf.convert_to_tensor(([1.]), dtype=tf.float64),
        #                'loss_gradu_vx': tf.convert_to_tensor(([1.]), dtype=tf.float64),
        #                'loss_gradu_vy': tf.convert_to_tensor(([1.]), dtype=tf.float64), 
        #                'loss_gradu_mu': tf.convert_to_tensor(([1.]), dtype=tf.float64),
        #                }
        self.kappas = {k:tf.convert_to_tensor(([1.]), dtype=tf.float64) for k in meanLpartial.keys()}
        
        return 

    def weighted_loss(self, unweighted_loss):
        ''' Computes the final weighted loss'''
        
        # Computation of the number of parcial loss terms involved
        nItems_phy = len(self.keys_phy)
        nItems_der = len(self.keys_der)
        
        # Divide array of loss into phy and der
        Lphy = unweighted_loss[:,:nItems_phy*self.grid.ncells,:]
        Lder = unweighted_loss[:,nItems_phy*self.grid.ncells:,:]
        
        # Extract the partial losses from the general loss
        Lpartial_phy = tf.split(Lphy, num_or_size_splits=nItems_phy, axis=1) #[(b,e,1),...,(b,e,1)]
        Lpartial_der = tf.split(Lder, num_or_size_splits=max(nItems_der,1), axis=1) #[(b,e,1),...,(b,e,1)]
        meanLpartial_phy = {self.keys_phy[n]:tf.reduce_mean(Lpartial_phy[n]) for n in range(nItems_phy)}
        meanLpartial_der = {self.keys_der[n]:tf.reduce_mean(Lpartial_der[n]) for n in range(nItems_der)}
        meanLpartial = {**meanLpartial_phy, **meanLpartial_der}
        # meanLpartial = {self.keys[n]:tf.reduce_mean(tf.reduce_sum(Lpartial[n], axis=[-1,-2])) for n in range(nL)}
        
        # loss = tf.reduce_sum([self.kappas[k]*meanLpartial[k] for k in meanLpartial.keys()])        
        loss = tf.reduce_sum([1. * meanLpartial[k] for k in meanLpartial.keys()])
        
        return loss, meanLpartial
    
    def L1_relative_error(self, y_true, y_pred):
        ''' Computes de L1 relative error'''
        
        L1RE = 100 * tf.norm(y_true-y_pred, ord=1) / tf.norm(y_true, ord=1)
        
        return L1RE
    
    def L2_relative_error(self, y_true, y_pred):
        ''' Computes de L2 relative error'''
        
        num = tf.sqrt(tf.reduce_sum(tf.square(y_true-y_pred), axis=[-1,-2]))
        den = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=[-1,-2]))
        L2RE = 100 * tf.reduce_mean(num / (den + 1e-12))
        
        return L2RE
    
#   @tf.function(jit_compile=True)
    def train_step(self, data):
        ''' Training loop'''
        
        x_dict, y_dict = data 
        
        # Generate spatial evaluation points
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                        for k,v in self.integration_points.items()}
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        if self.LS_activation == True:
            # Construct and resolve LS and update weights of linear layer
            self.resolve_LS(x_curated, y_dict)
            # Assign training values
            # trainable_vars = [weight.value for weight in self.net.trainable_weights[:-1]] #GPU
            trainable_vars = self.net.trainable_weights[:-1] #CPU  
        else:
            # Assign training values
            # trainable_vars = [weight.value for weight in self.net.trainable_weights] #GPU
            trainable_vars = self.net.trainable_weights #CPU
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_vars)
            
            A, b = self.system(x_curated, y_dict)
            unweighted_loss = tf.square(self.net.linear_layer(A)-b)
            loss, partialLosses = self.weighted_loss(unweighted_loss)
            # loss = tf.reduce_mean(unweighted_loss)
            
            # loss, partialLosses = self.construct_LS_complete_mu_matrix_direct(x_curated, y_dict, integ_weights)
            
            # tf.print('Mean x coord: ', tf.reduce_mean(x_curated['coord_x']))
            # tf.print('Mean branch weights: ', tf.reduce_mean(self.net.branch_mu.trainable_weights[0]))
            # tf.print('Mean trunk weights: ', tf.reduce_mean(self.net.trunk.trainable_weights[0]))
                    
        result = self.call(x_dict)    
        
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))

        metrics = {'loss': loss,
                  'L2re_u': self.L2_relative_error(y_dict['T'], result['T']),
                  'L2re_grad_u': self.L2_relative_error(
                      tf.sqrt(tf.square(y_dict['grad(T)_x']) + tf.square(y_dict['grad(T)_y'])),
                      tf.sqrt(tf.square(result['grad(T)_x']) + tf.square(result['grad(T)_y'])))}
        
        result = {**metrics, **partialLosses}
        
        return result
    
    
    def test_step(self, data):
        ''' Validation loop'''
        
        x_dict, y_dict = data 
        
        # Generate spatial evaluation points
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                        for k,v in self.integration_points.items()}
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        result = self.call(x_dict)    

        metrics = {'loss': loss,
                  'L2re_u': self.L2_relative_error(y_dict['T'], result['T']),
                  'L2re_grad_u': self.L2_relative_error(
                      tf.sqrt(tf.square(y_dict['grad(T)_x']) + tf.square(y_dict['grad(T)_y'])),
                      tf.sqrt(tf.square(result['grad(T)_x']) + tf.square(result['grad(T)_y'])))}
        
        result = {**metrics, **partialLosses}
        
        return result
        
    def train_step_BFGS(self, data):
        
        x_dict, y_dict = data 
        
        # Generate spatial evaluation points
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                        for k,v in self.integration_points.items()}
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        return loss
    
    def train_step_LSBFGS(self, data):
        
        x_dict, y_dict = data 
        
        # Generate spatial evaluation points
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                        for k,v in self.integration_points.items()}
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **integ_points}
        
        # Solve LS and update lineal layer
        self.resolve_LS(x_curated, y_dict)
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        return loss
    
    def call(self, data):
        '''Prepares the data, calls the DeepONet and integrates de results'''
        
        # Generate spatial evaluation points 
        batch_size = data[list(data.keys())[0]].shape[0]
        
        integ_points = {k:tf.tile(v,[batch_size,1,1]) 
                        for k,v in self.integration_points.items()}
        
        # Prepare data for DeepONet
        data_curated = {**data, **integ_points}
        
        # Call DeepONet and linear layer
        net_output = self.net(data_curated) 
        
        result = {'T': net_output['T'],
                   'grad(T)_x': net_output['grad(T)_x'],
                   'grad(T)_y': net_output['grad(T)_y'],
                   'jacUx(T)': net_output['jacUx(T)'],
                   'jacUy(T)': net_output['jacUy(T)'],
                   'jacMu(T)': net_output['jacMu(T)']}
        
        return result
    
    def get_coefss_and_basis(self, data):
        
        #Generate integration points and weights (if random)
        batch_size = data[list(data.keys())[0]].shape[0]
        # weights = tf.tile(self.integ_weights,[batch_size,1,1])
        integ_points, integ_weights = self.points_and_weights(batch_size)
        
        # Prepare data for DeepONet
        data_curated = {**data, **integ_points}
        
        # Call DeepONet and linear layer
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.net.construct_matrix(data_curated)
        
        results = {'br_v': Umatrix_v,
                   'br_mu':Umatrix_mu,
                   'tr_int': self.integrate(integ_weights, Umatrix_x),
                   'tr': Umatrix_x,
                   'dbr_vx': dUmatrix_v[:,:,:,0],
                   'dbr_vy': dUmatrix_v[:,:,:,1],
                   'dbr_mu':dUmatrix_mu,
                   'dtr_x_int': self.integrate(integ_weights, dUmatrix_x[:,:,:,:,0]),
                   'dtr_y_int': self.integrate(integ_weights, dUmatrix_x[:,:,:,:,1]),
                   'dtr_x': dUmatrix_x[:,:,:,:,0],
                   'dtr_y': dUmatrix_x[:,:,:,:,1],
                   'coord_x' : data_curated['coord_x'],
                   'coord_y': data_curated['coord_y'],
                   'weights': integ_weights
                   }
        
        return results
    
class initialize_and_compute_kappas(keras.callbacks.Callback):
   ''' Monitors memory usage on epoch begin and end, collect garbage'''
   def __init__(self, x_train, y_train):
       super().__init__()
       self.x_train = x_train
       self.y_train = y_train
       
   def on_train_begin(self, logs=None):
        self.model.compute_kappas(self.x_train, self.y_train)
        
        
class LSonEpoch(keras.callbacks.Callback):
   def __init__(self, x_train, y_train):
       super().__init__()
       self.x_train = x_train
       self.y_train = y_train

   def on_epoch_begin(self, epoch, logs=None):
       self.model.resolve_LS((self.x_train,self.y_train))
       
       
class MemoryUsageCallbackExtended(keras.callbacks.Callback):
   ''' Monitors memory usage on epoch begin and end, collect garbage'''
   def on_epoch_begin(self, epoch, logs=None):
       print('**Epoch {}**'.format(epoch))
       print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

   def on_epoch_end(self, epoch, logs=None):
       print('Memory usage on epoch end: {}'.format(psutil.Process(os.getpid()).memory_info().rss))
       gc.collect()
       
# you can then create the callback by passing the correct attributes
# my_callback = CustomCallback(testRatings, testNegatives, topK, evaluation_threads)

def get_model_performance(model, x_train, y_train, x_val, y_val, print_results=False):
    '''Returns the losses and errors of the model inside the training and validation
    distribution for the new weights'''
    
    metrics_tr = model.test_step((x_train, y_train))
    metrics_val = model.test_step((x_val, y_val))
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
    
#%% INITIAL TESTS

grid = grids2D.Grid(size_x = 10, size_y = 10, step_size = 0.1)

# Data for training and validation
data_route = '../OpenFOAM/convectionDiffusion2D_10x10_mu_v2/training_data/'
training_data = fRW.upload_training_data(data_route, jacobian=True)
# x_train, y_train, x_val, y_val = prepare_raw_data(training_data, train_split=0.72)
# x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[4,14,25])
x_train, y_train, x_val, y_val = prepare_raw_data_custom(training_data, training_ind=[1,9,25])
plot0 = pp.plot_data_distribution(x_train, y_train, x_val, y_val)

# Select just one sample
# sample = 1
# x_train = {k:v[sample:sample+1] for k,v in x_train.items()}
# y_train = {k:v[sample:sample+1] for k,v in y_train.items()}
# x_val = {k:v[sample:sample+1] for k,v in x_val.items()}
# y_val = {k:v[sample:sample+1] for k,v in y_val.items()}

# Create learning model
net = DeepONet(layers_branch_v=[50], layers_branch_mu=[50,50], 
                layers_trunk=[50,50,50], num_rand_sampling = 20, dimension='2D')

# Select model
# model = my_model(net, grid, 'vanilla',   LS=True)
# model = my_model(net, grid, 'physics_0', LS=True)
# model = my_model(net, grid, 'physics_2', LS=True)
# model = my_model(net, grid, 'van+der',  LS=True)
# model = my_model(net, grid, 'phy0+der',  LS=True)
model = my_model(net, grid, 'phy2+der',  LS=True)

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
                    epochs=4000, batch_size=3, validation_batch_size=25,
                    callbacks=[checkpoint])

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

_ = get_model_performance(model, x_train, y_train, x_val, y_val, print_results=True)

metrics_LBFGS = []
epochs_LBFGS = 150
print('~~~~~BEGINNING OF LBFGS OPTIMIZATION~~~~~')
for i in range(epochs_LBFGS):
    # Traininig all the weights with LBFGS
    # results_LBFGS = lbfgs_minimize(model.trainable_weights, model.train_step_BFGS, (x_train, y_train))
    # Traininig lineal layer with LS and the rest of the weights with LBFGS
    results_LBFGS = lbfgs_minimize(model.trainable_weights[:-1], model.train_step_LSBFGS, (x_train, y_train))
    
    metrics_LBFGS.append(get_model_performance(model, x_train, y_train, x_val, y_val, print_results=True))

LBFGS_history = {k: [d[k].numpy() for d in metrics_LBFGS] for k in metrics_LBFGS[0].keys()}
df_history_LBFGS = pd.DataFrame(LBFGS_history, columns=list(LBFGS_history.keys()))
df_history_LBFGS.to_csv(f'experiments/L3+Lder_LS_L-BFGS.csv', index=False)
# df_history_LBFGS = pd.read_csv(f'experiments/L1+Lder_noLS_L-BFGS.csv')

plot6 = pp.plot_loss(df_history_LBFGS)
plot7 = pp.plot_partialLosses_train(df_history_LBFGS)
plot8 = pp.plot_partialLosses_val(df_history_LBFGS)
plot9 = pp.plot_L2re_u(df_history_LBFGS, ymin = 0, ymax = 50)
plot10 = pp.plot_L2re_gradu(df_history_LBFGS, ymin = 0, ymax = 45)

model.save_weights('experiments/L3+Lder_LS_weights_LBFGS.h5')

# plt.plot(df_LBFGS_history['loss'], label='Data')
# plt.yscale("log")
# plt.title("L-BFGS Loss")
# # plt.savefig('C3_5.BFGS_200elems.png', dpi=200, bbox_inches='tight')
# plt.show()


# model.load_weights('L1+Lder/weights_LBFGS.h5')



# # Generate spatial evaluation points
# batch_size = x_train[list(x_train.keys())[0]].shape[0]
# integ_points = {k:tf.tile(v,[batch_size,1,1]) 
#                 for k,v in model.integration_points.items()}

# # Prepare data for DeepONet
# x_curated = {**x_train, **integ_points}

# A, b = model.system(x_curated, y_train)

# A1=A[0,:100,:].numpy()
# A2=A[0,100:,:].numpy()

# keys = ['loss']
# fig, ax = plt.subplots(1, 1)
# for k in keys:
#     ax.plot(np.array(df_history[k]), label=k)
# ax.set_yscale("log")
# ax.set_title('Loss evolution')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()

# plot1.savefig(f'figures/loss_van_noLS_one.png', dpi=200, bbox_inches='tight')


# pp.plot_field(result['T'][batch,:,0], grid, 'u pred', None, None)
# pp.plot_field(result['grad(T)_x'][batch,:,0], grid, 'grad(u)_x pred', None, None)
# pp.plot_field(result['grad(T)_y'][batch,:,0], grid, 'grad(u)_y pred', None, None)
# pp.plot_field(result['jacUx(T)'][batch,:,0], grid, 'partial(U/vx) pred', None, None)
# pp.plot_field(result['jacUy(T)'][batch,:,0], grid, 'partial(U/vy) pred', None, None)
# pp.plot_field(result['jacMu(T)'][batch,:,0], grid, 'partial(U/mu) pred', None, None)

# pp.plot_field(y_train['T'][batch,:,0], grid, 'u true', None, None)
# pp.plot_field(y_train['grad(T)_x'][batch,:,0], grid, 'grad(u)_x true', None, None)
# pp.plot_field(y_train['grad(T)_y'][batch,:,0], grid, 'grad(u)_y true', None, None)
# pp.plot_field(y_train['jacUx(T)'][batch,:,0], grid, 'partial(U/vx) true', None, None)
# pp.plot_field(y_train['jacUy(T)'][batch,:,0], grid, 'partial(U/vy) true', None, None)
# pp.plot_field(y_train['jacMu(T)'][batch,:,0], grid, 'partial(U/mu) true', None, None)

# c_and_b = model.get_coefss_and_basis(x_train)
# for j in range(model.net.num_basis_func):
#     pp.plot_field(c_and_b['tr_int'][0,:,j], grid, f'Base function {j}', None, None)
    
# plt.scatter(np.array([0,1,2]), c_and_b['br_mu'][0,:],label='coeffs mu')
# plt.scatter(np.array([0,1,2]), c_and_b['br_v'][0,:],label='coeffs v')
# plt.xticks([0,1,2,3,4])
# plt.grid()
# plt.legend()
# plt.show()
# # Construct the grid
# # grid = grids2D.Grid(size_x = 50, size_y = 50, step_size = 0.02)
# grid = grids2D.Grid(size_x = 10, size_y = 10, step_size = 0.1)

# # Data for training
# data_route = '../OpenFOAM/convectionDiffusion2D_10x10/training_data/'
# training_data = fRW.upload_training_data(data_route, jacobian=True)
# x_train, y_train_full, _, _ = prepare_raw_data(training_data, train_split=1.)

# # Data for validation
# data_route = '../OpenFOAM/convectionDiffusion2D_10x10/validation_data/'
# validation_data = fRW.upload_training_data(data_route, jacobian=True)
# x_val, y_val_full, _, _ = prepare_raw_data(validation_data, train_split=1.)

# # Create learning model
# net = DeepONet(layers_branch_v=[10,10,5], layers_branch_mu=[10,10,5], 
#                layers_trunk=[10,10,5], num_rand_sampling = 20, dimension='2D')

# # model = my_model(net, grid, 'complete', 'GaussP1', LS=True)
# model = my_model(net, grid, 'vanilla', 'GaussP2', LS=True)
# # model = my_model(net, grid, 'vanilla', 'MC')
# # model = my_model(net, grid, 'complete', 'QMC', LS=True)

# # net = DeepONetv2(layers_branch=[10,10,5], 
# #                layers_trunk=[10,10,5], num_rand_sampling = 20, dimension='2D')
# # model = my_model(net, grid, 'complete', LS=False)

# y_train = process_jacobians(y_train_full, model.random_sampling)
# y_val = process_jacobians(y_val_full, model.random_sampling)

# #Save and load weights
# # model.save_weights('weights_op1.h5')
# # U, dU_vx, dU_vy, dU_mu, dU_x, dU_y = model(x_train)
# # model.load_weights('weights_op1.h5')

# #Compilation of the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=10**(-1)))#, run_eagerly=True)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', 
#                                                  factor=0.5, patience=15, 
#                                                  min_lr=10e-7, min_delta=0.001, 
#                                                  verbose=1)
# earlystopping = keras.callbacks.EarlyStopping(monitor='loss',
#                                                   patience=100,
#                                                   restore_best_weights=False,
#                                                   min_delta=0.0001)
    
# init_and_kappas = initialize_and_compute_kappas(x_train, y_train)
# memoryusage = MemoryUsageCallbackExtended()
    
# history = model.fit(x_train, y_train, validation_data = (x_val, y_val), 
#                     epochs=1000, batch_size=8, callbacks=[reduce_lr, 
#                                                           earlystopping,
#                                                           init_and_kappas])#,
#                                                           #memoryusage])

# df_history = pd.DataFrame(history.history)

# df_history.to_csv(f'model1_training.csv', index=False)
# df_history_1 = reloaded_df_history = pd.read_csv('0.training_full_code_400it.csv')


#%% ONE SAMPLE; ONLY TRUNK; INTEGRATION POINTS

def L2_relative_error(y_true, y_pred):
    ''' Computes de L2 relative error'''
    
    num = tf.sqrt(tf.reduce_sum(tf.square(y_true-y_pred), axis=[-1,-2]))
    den = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=[-1,-2]))
    L2RE = 100 * tf.reduce_mean(num / (den + 1e-12))
    
    return L2RE

grid = grids2D.Grid(size_x = 10, size_y = 10, step_size = 0.1)

# Data for training and validation
data_route = '../OpenFOAM/convectionDiffusion2D_10x10_mu_v2/training_data/'
training_data = fRW.upload_training_data(data_route, jacobian=True)
x_train, y_train, x_val, y_val = prepare_raw_data(training_data, train_split=0.72)

sample = 1
x_train = {k:v[sample:sample+1] for k,v in x_train.items()}
y_train = {k:v[sample:sample+1] for k,v in y_train.items()}
x_val = {k:v[sample:sample+1] for k,v in x_val.items()}
y_val = {k:v[sample:sample+1] for k,v in y_val.items()}

# Create trunk
layers_trunk = [50,50,50]
tru_layers = [keras.layers.Dense(units=layer, activation='tanh', 
                                          use_bias=True) for layer in layers_trunk]
net_tr = keras.Sequential(layers=tru_layers, name='Trunk')

# Create branch
layers_branch = [50,50]
br_layers = [keras.layers.Dense(units=layer, activation='tanh', 
                                          use_bias=True) for layer in layers_branch]
net_br = keras.Sequential(layers=br_layers, name='Branch')


linear_layer = keras.layers.Dense(units=1, activation=None, use_bias=False,
                                  name='linear')#, kernel_initializer=tf.keras.initializers.Ones())

optimizer=keras.optimizers.Adam(learning_rate=0.01)#10**(-3))
epochs = 4000
# model_type = 'simple'
model_type = 'complete'
LS = 'NO'

loss_eval = []
err_eval = []

if LS == 'NO' and model_type == 'simple':
    points, weights = grid.generate_quadrature_centroids(batch=1)
    input_keys = ['coord_x', 'coord_y']
    coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
    # tf.print('Coords shape: ', coords_res.shape)
    for epoch in tf.range(epochs):
        with tf.GradientTape() as tape:
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            y_pred = linear_layer(basis_fun)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights+linear_layer.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights+linear_layer.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
    
    
elif LS== 'EXP' and model_type == 'simple':
    points, weights = grid.generate_quadrature_centroids(batch=1)
    input_keys = ['coord_x', 'coord_y']
    coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
    # tf.print('Coords shape: ', coords_res.shape) 
        
    for epoch in tf.range(epochs): 
        A = net_tr(coords_res)
        y_pred = linear_layer(A) #Initialization
        alpha = tf.linalg.lstsq(A, y_train['T'], l2_regularizer=10**(-13))
        # tf.print(alpha.shape)
        linear_layer.weights[0].assign(tf.reduce_mean(alpha,axis=0))
        with tf.GradientTape() as tape:
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            y_pred = linear_layer(basis_fun)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
        
elif LS== 'EXP2' and model_type == 'simple':
    for epoch in tf.range(epochs):
        points, weights = grid.generate_quadrature_centroids(batch=1)
        input_keys = ['coord_x', 'coord_y']
        coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
        # tf.print('Coords shape: ', coords_res.shape)
        with tf.GradientTape() as tape:
            A = net_tr(coords_res)
            y_pred = linear_layer(A) #Initialization
            A_flat = tf.squeeze(A, axis=0)
            b_flat = tf.squeeze(y_train['T'], axis=0)
            LHS_original = tf.matmul(A_flat,A_flat, transpose_a=True)
            RHS_original = tf.matmul(A_flat,b_flat, transpose_a=True)
            alpha = tf.linalg.solve(LHS_original,RHS_original)
            # tf.print(alpha.shape)
            linear_layer.weights[0].assign(alpha)
            
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            y_pred = linear_layer(basis_fun)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights)#+linear_layer.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights))#+linear_layer.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
        
elif LS== 'IMP' and model_type == 'simple':
    # Initialization
    basis_fun = net_tr(coords_res)
    y_pred = linear_layer(basis_fun)
    
    for epoch in tf.range(epochs):
        points, weights = grid.generate_quadrature_centroids(batch=1)
        input_keys = ['coord_x', 'coord_y']
        coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
        # tf.print('Coords shape: ', coords_res.shape)
        
        # Construct first and second derivative of the loss wrt alpha when alpha=0
        linear_layer.weights[0].assign(tf.zeros_like(linear_layer.weights[0]))
        
        with tf.GradientTape() as t1:
            with tf.GradientTape() as t2:
                basis_fun = net_tr(coords_res)
                y_pred = linear_layer(basis_fun)
                loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
                
            J = t2.gradient(loss,linear_layer.weights[0])# the RHS of the system: -2*B^t*l  
        H = tf.squeeze(t1.jacobian(J,linear_layer.weights[0])) # the LHS of the system: 2*B^t*B
        
        LHS_normal = 1/2*H
        RHS_normal = -1/2*J
        
        weights_normal = tf.linalg.solve(LHS_normal,RHS_normal)
        linear_layer.weights[0].assign(weights_normal)
        
        with tf.GradientTape() as tape:
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            y_pred = linear_layer(basis_fun)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
    
elif LS == 'NO' and model_type == 'complete':
    batch = x_train['DT'].shape[0]
    points, weights = grid.generate_quadrature_centroids(batch=batch)
    input_keys = ['coord_x', 'coord_y']
    # coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
    coords_res = tf.concat([points[k] for k in input_keys], axis = -1)
    # tf.print('Coords shape: ', coords_res.shape)
    for epoch in tf.range(epochs):
        with tf.GradientTape() as tape:
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            basis_coef = net_br(x_train['DT'])
            # tf.print('Basis coeff shape: ', basis_coef.shape)
            y_pred = tf.einsum('bj,bej->bej', basis_coef, basis_fun)
            y_pred = linear_layer(y_pred)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights+net_br.trainable_weights+linear_layer.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights+net_br.trainable_weights+linear_layer.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
            
elif LS== 'EXP' and model_type == 'complete':
    batch = x_train['DT'].shape[0]
    points, weights = grid.generate_quadrature_centroids(batch=batch)
    input_keys = ['coord_x', 'coord_y']
    # coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
    coords_res = tf.concat([points[k] for k in input_keys], axis = -1)
    # tf.print('Coords shape: ', coords_res.shape) 
        
    for epoch in tf.range(epochs): 
        basis_fun = net_tr(coords_res)
        basis_coef = net_br(x_train['DT'])
        A = tf.einsum('bj,bej->bej', basis_coef, basis_fun)
        y_pred = linear_layer(A) #Initialization
        # alpha = tf.linalg.lstsq(A, y_train['T'], l2_regularizer=10**(-3))
        # linear_layer.weights[0].assign(tf.reduce_mean(alpha,axis=0))
        
        A_flat = tf.reshape(A,[-1,A.shape[-1]])
        b_flat = tf.reshape(y_train['T'],[-1,y_train['T'].shape[-1]])
        alpha = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-8))
        linear_layer.weights[0].assign(alpha)
        # tf.print(alpha.shape)
        
        with tf.GradientTape() as tape:
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            basis_coef = net_br(x_train['DT'])
            # tf.print('Basis coeff shape: ', basis_coef.shape)
            A = tf.einsum('bj,bej->bej', basis_coef, basis_fun)
            y_pred = linear_layer(A)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights+net_br.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights+net_br.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')
        
elif LS== 'EXP2' and model_type == 'complete':
    batch = x_train['DT'].shape[0]
    points, weights = grid.generate_quadrature_centroids(batch=batch)
    input_keys = ['coord_x', 'coord_y']
    # coords_res = tf.reshape(tf.concat([tf.expand_dims(points[k], axis=-1) for k in input_keys], axis = -1), [1,-1, len(input_keys)])
    coords_res = tf.concat([points[k] for k in input_keys], axis = -1)
    # tf.print('Coords shape: ', coords_res.shape)
    for epoch in tf.range(epochs):   
        basis_fun = net_tr(coords_res)
        basis_coef = net_br(x_train['DT'])
        A = tf.einsum('bj,bej->bej', basis_coef, basis_fun)
        y_pred = linear_layer(A) #Initialization
        A_flat = tf.reshape(A,[-1,A.shape[-1]])
        b_flat = tf.reshape(y_train['T'],[-1,y_train['T'].shape[-1]])
        LHS_original = tf.matmul(A_flat,A_flat, transpose_a=True)
        RHS_original = tf.matmul(A_flat,b_flat, transpose_a=True)
        alpha = tf.linalg.solve(LHS_original,RHS_original)
        # tf.print(alpha.shape)
        linear_layer.weights[0].assign(alpha)
        with tf.GradientTape() as tape:    
            basis_fun = net_tr(coords_res)
            # tf.print('Basis funct shape: ', basis_fun.shape)
            basis_coef = net_br(x_train['DT'])
            # tf.print('Basis coeff shape: ', basis_coef.shape)
            A = tf.einsum('bj,bej->bej', basis_coef, basis_fun)
            y_pred = linear_layer(A)
            # tf.print('u shape: ', y_pred.shape)
            loss = tf.reduce_mean(tf.square(y_pred-y_train['T']))
        L2error = L2_relative_error(y_train['T'], y_pred)    
        grad = tape.gradient(loss, net_tr.trainable_weights+net_br.trainable_weights)#+linear_layer.trainable_weights)
        optimizer.apply_gradients(zip(grad, net_tr.trainable_weights+net_br.trainable_weights))#+linear_layer.trainable_weights))
        loss_eval.append(loss)
        err_eval.append(L2error)
        tf.print(f'Epoch {epoch}/{epochs}: {loss:.3e}, {L2error:.4f}')        
plot1 = pp.plot_field(y_pred[0,:,0], grid, 'u pred', None, None)
plot2 = pp.plot_field(y_train['T'][0,:,0], grid, 'u true', None, None)
fig1, ax = plt.subplots(1, 1)
ax.plot(loss_eval, label='loss')
ax.set_yscale("log")
ax.set_title('Loss evolution')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

# plot1.savefig(f'lite/pred_LSexp2.png', dpi=200, bbox_inches='tight')
# plot2.savefig(f'lite/true_LSimp.png', dpi=200, bbox_inches='tight')
# fig1.savefig(f'lite/loss_LSimp.png', dpi=200, bbox_inches='tight')

fig2, ax2 = plt.subplots(1, 1)
ax2.plot(err_eval, label='l2-error u')
ax2.set_title('Error u evolution')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Error (%)')
ax2.set_ylim(0, 100)

epoch_min = np.argmin(loss_eval)
print('MIN LOSS: ', loss_eval[epoch_min])
print('MIN ERR u:', err_eval[epoch_min])

#%%
integ_points = {k:tf.tile(v,[1,1,1]) for k,v in model.integration_points.items()}
x_curated = {**x_train, **integ_points}
A, b = model.system(x_curated, y_train)
A1 = A[:,100:,:]
A2 = A[:,:100,:]
b1 = b[:,100:,:]
b2 = b[:,:100,:]

A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
alphaEXP =  tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-12))

A_flat = tf.reshape(A,[-1,A.shape[-1]])
b_flat = tf.reshape(b,[-1,b.shape[-1]])
LHS_original = tf.matmul(A_flat,A_flat, transpose_a=True)
RHS_original = tf.matmul(A_flat,b_flat, transpose_a=True)
alphaEXP2 = tf.linalg.solve(LHS_original,RHS_original)

A1_flat = tf.reshape(A1,[-1,A1.shape[-1]])
b1_flat = tf.reshape(b1,[-1,b1.shape[-1]])
A2_flat = tf.reshape(A2,[-1,A2.shape[-1]])
b2_flat = tf.reshape(b2,[-1,b2.shape[-1]])
LHS1 = tf.matmul(A1_flat,A1_flat, transpose_a=True)
LHS2 = tf.matmul(A2_flat,A2_flat, transpose_a=True)
RHS1 = tf.matmul(A1_flat,b1_flat, transpose_a=True)
RHS2 = tf.matmul(A2_flat,b2_flat, transpose_a=True)
LHS = LHS1+LHS2
RHS = RHS1+RHS2
alphaEXP3 = tf.linalg.solve(LHS,RHS)