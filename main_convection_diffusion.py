#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr 16 17:13:06 2024

@author: jesusglezs97
"""
import numpy as np
import tensorflow as tf
import foamRW as fRW
import grids2D
import pandas as pd
import matplotlib.pyplot as plt
from tfp_optimizer import lbfgs_minimize

from time import time
tf.keras.backend.set_floatx('float64')
# Random seeds ror being deterministic.
tf.keras.utils.set_random_seed(1234)

def prepare_raw_data(data, train_split = 0.8, outputs=['T', 'grad(T)_x', 'grad(T)_y', 'jacMu(T)', 'jacUx(T)', 'jacUy(T)']):
    ''' Divides the raw data into training and validation sets'''
    
    samples, _ = data[outputs[0]].shape
    indices = tf.range(start=0, limit=samples, dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices, seed=42)
    shuffled_indices = indices
    
    training_ind = shuffled_indices[:int(samples*train_split)]
    validation_ind = shuffled_indices[int(samples*train_split):]
    
    x_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_train = {k:tf.expand_dims(tf.gather(v, training_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    x_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k not in outputs}
    y_val = {k:tf.expand_dims(tf.gather(v, validation_ind, axis=0), axis=-1) for k,v in data.items() if k in outputs}
    
    return x_train, y_train, x_val, y_val

class DeepONet(tf.keras.Model):
    def __init__(self, layers_branch_v, layers_branch_mu, layers_trunk, 
                 num_rand_sampling = 100, dimension = '1D', **kwargs):
        super(DeepONet, self).__init__()
        
        assert layers_branch_v[-1]!=layers_branch_mu, 'The size of the last layer of the branches must be equal'
        assert layers_branch_v[-1]!=layers_trunk, 'The size of the last layer of the branch and trunk must be equal'
        
        self.num_rand_sampling = num_rand_sampling
        self.fields_brv, self.fields_brmu, self.fields_tr = self.fields_by_dimension(dimension)

        # Create branch net for velocity
        br_v_layers = [tf.keras.layers.Dense(units=layer, activation="tanh", 
                                                 use_bias=True)
                           for layer in layers_branch_v]
        br_v_layers.insert(0, tf.keras.Input(shape=(self.num_rand_sampling,2,)))
        br_v_layers.insert(1, tf.keras.layers.Flatten())
        self.branch_v = tf.keras.Sequential(br_v_layers)
        
        # Create branch net for mu
        br_mu_layers = [tf.keras.layers.Dense(units=layer, activation="tanh", 
                                                 use_bias=True)
                           for layer in layers_branch_mu]
        br_mu_layers.insert(0, tf.keras.Input(shape=(self.num_rand_sampling,)))
        self.branch_mu = tf.keras.Sequential(br_mu_layers)
        
        # Create trunk
        tru_layers = [tf.keras.layers.Dense(units=layer, activation="tanh", 
                                                 use_bias=True)
                           for layer in layers_trunk]
        self.trunk = tf.keras.Sequential(tru_layers)
        
        # Linear layer
        self.lineal_layer = tf.keras.layers.Dense(units=1, 
                                                  activation=None, use_bias=False, 
                                                  name='Lineal')
        
    def fields_by_dimension(self, dimension):
        '''Selects the involved fields depending on the dimension'''
        
        fields_brv = ['v_x_sampled', 'v_y_sampled']
        fields_brmu = ['DT_sampled']
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
            state_watched = tf.concat([tf.reshape(state[k], [-1,1]) for k in input_keys], axis=-1)
        else:
            state_watched = tf.concat([state[k] for k in input_keys], axis=-1)

        return state_watched
    
    def construct_matrix(self, inputs):
        '''Construct all the matrix obtained previously to the linear layer 
        application'''
        
        dim_input_tr = inputs['coord_x'].shape.as_list()
        
        x_brv = self.sort_and_reshape_state(inputs, self.fields_brv)
        # print('Input shape brv: ', x_brv.shape)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_brv)
            basis_brv = self.branch_v(x_brv)
            # basis = basis*(x_watched-0) #To impose boundary conditions
            # basis = basis*(x_watched-1)
            y_pred = self.lineal_layer(basis_brv) #Necessary to initialize the layer
        dbasis_brv = tf.squeeze(tape.batch_jacobian(basis_brv,x_brv))
        # print('Output shape basis brv: ', basis_brv.shape)
        # print('Output shape dbasis brv: ', dbasis_brv.shape)

        x_brmu = self.sort_and_reshape_state(inputs, self.fields_brmu)
        # print('Input shape brmu: ', x_brmu.shape)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_brmu)
            basis_brmu = self.branch_mu(x_brmu)

        dbasis_brmu = tf.squeeze(tape.batch_jacobian(basis_brmu,x_brmu))
        # print('Output shape basis brmu: ', basis_brmu.shape)
        # print('Output shape dbasis brmu: ', dbasis_brmu.shape)
        
        x_tr = self.sort_and_reshape_state(inputs, self.fields_tr, reshape=True)
        # print('Input shape tr: ', x_tr.shape)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tr)
            basis_tr = self.trunk(x_tr)

        dbasis_tr = tf.squeeze(tape.batch_jacobian(basis_tr,x_tr))
        dbasis_tr = tf.reshape(dbasis_tr, dim_input_tr+dbasis_tr.shape.as_list()[1:])
        # print('Output shape basis tr: ', basis_tr.shape)
        # print('Output shape dbasis tr: ', dbasis_tr.shape)
        
        basis_tr = tf.reshape(basis_tr, dim_input_tr+[-1])

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
    #             y_pred = self.lineal_layer(basis) #Necessary to initialize the layer
    #         dbasis = tf.squeeze(tape_inner.batch_jacobian(basis,x_watched))
    #     ddbasis = tf.squeeze(tape_outer.batch_jacobian(dbasis,x_watched))
        
    #     basis = tf.reshape(basis, dim_input+[-1])
    #     dbasis = tf.reshape(dbasis, dim_input+dbasis.shape.as_list()[1:])
    #     ddbasis = tf.reshape(ddbasis, dim_input+ddbasis.shape.as_list()[1:])

    #     return (basis,dbasis,ddbasis)
    
 
    def call(self, inputs):
        '''Applies the linear layer to both the basis and the dbasis'''
        
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.construct_matrix(inputs)
        
        basis = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, Umatrix_x)
        db_vx = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,0], Umatrix_mu, Umatrix_x)
        db_vy = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,1], Umatrix_mu, Umatrix_x)
        db_mu = tf.einsum('bj,bjs,beij->beisj', Umatrix_v, dUmatrix_mu, Umatrix_x)
        db_x = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,0])
        db_y = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,1])
        
        U = self.lineal_layer(basis)
        dU_vx = self.lineal_layer(db_vx)
        dU_vy = self.lineal_layer(db_vy)
        dU_mu = self.lineal_layer(db_mu)
        dU_x = self.lineal_layer(db_x)
        dU_y = self.lineal_layer(db_y)
        
        return U, dU_vx, dU_vy, dU_mu, dU_x, dU_y
    
class my_model(tf.keras.Model):
    
    def __init__(self, net, grid, LS_constructor, **kwargs):
        super(my_model, self).__init__()
        self.net = net
        self.grid = grid
        if LS_constructor == 'vanilla':
            self.LS_constructor = self.construct_LS_vanilla
        elif LS_constructor == 'physics_1':
            self.LS_constructor = self.construct_LS_physics_1
        elif LS_constructor == 'physics_2':
            self.LS_constructor = self.construct_LS_physics_2
        elif LS_constructor == 'complete':
            self.LS_constructor = self.construct_LS_complete
            
        # Generate integration points and weights (if not random)
        self.integ_points, self.integ_weights = self.grid.generate_integration_points_and_weights_nonrandom()
        
        # Generate random sampling of points
        self.random_sampling = self.grid.generate_random_sampling_of_points(self.net.num_rand_sampling)
        
    def integrate(self, weights, fun_values):
        '''Integrates a function over the finite volume by giving the weights and
        the value of the function at the integrating points'''
        
        return tf.einsum('bei,beij->bej', weights, fun_values)
    
    def integrate_derivative(self, weights, fun_values):
        '''Integrates a derivative function over the finite volume by giving the weights and
        the value of the function at the integrating points'''
        
        return tf.einsum('bei,beisj->besj', weights, fun_values)
    
    def construct_LS_vanilla(self, x_dict, y_dict, weights):
        '''Constructs the Least-Squares system A x = b for the vanilla DeepONets'''

        #Contruct the matrix for LS system
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.net.construct_matrix(x_dict) 
        
        basis = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, Umatrix_x)
        Umatrix = self.integrate(weights, basis) #Integration
        
        Amatrix = Umatrix #LHS of the LS system
        bvector = y_dict['T'] #RHS of the LS system
  
        return Amatrix, bvector
    
    def construct_LS_physics_1(self, x_dict, y_dict, weights):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.net.construct_matrix(x_dict) 
        
        basis = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, Umatrix_x)
        db_vx = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,0], Umatrix_mu, Umatrix_x)
        db_vy = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,1], Umatrix_mu, Umatrix_x)
        db_mu = tf.einsum('bj,bjs,beij->beisj', Umatrix_v, dUmatrix_mu, Umatrix_x)
        db_x = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,0])
        db_y = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,1])

        # Integrate the functions
        Umatrix = self.integrate(weights, basis)
        dUmatrix_vx = self.integrate_derivative(weights, db_vx) 
        dUmatrix_vy = self.integrate_derivative(weights, db_vy) 
        dUmatrix_mu = self.integrate_derivative(weights, db_mu) 
        dUmatrix_x = self.integrate(weights, db_x) 
        dUmatrix_y = self.integrate(weights, db_y) 
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix_x, vdUmatrix_y, mudUmatrix_x, mudUmatrix_y], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bej->bej', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bej->bej', x_dict['v_y'], y_dict['grad(T)_y'])
        mudUmatrix_x_fv = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_x_fv, vdUmatrix_y_fv, mudUmatrix_x_fv, mudUmatrix_y_fv], axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_physics_2(self, x_dict, y_dict, weights):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.net.construct_matrix(x_dict) 
        
        basis = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, Umatrix_x)
        db_vx = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,0], Umatrix_mu, Umatrix_x)
        db_vy = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,1], Umatrix_mu, Umatrix_x)
        db_mu = tf.einsum('bj,bjs,beij->beisj', Umatrix_v, dUmatrix_mu, Umatrix_x)
        db_x = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,0])
        db_y = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,1])

        # Integrate the functions
        Umatrix = self.integrate(weights, basis)
        dUmatrix_vx = self.integrate_derivative(weights, db_vx) 
        dUmatrix_vy = self.integrate_derivative(weights, db_vy) 
        dUmatrix_mu = self.integrate_derivative(weights, db_mu) 
        dUmatrix_x = self.integrate(weights, db_x) 
        dUmatrix_y = self.integrate(weights, db_y) 
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_x)
        mudUmatrix_y = tf.einsum('bep,bej->bej', tf.sqrt(x_dict['DT']), dUmatrix_y)
        mudUmatrix = tf.sqrt(tf.square(mudUmatrix_x) + tf.square(mudUmatrix_y))

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix, mudUmatrix], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bei,bei->bei', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bei,bei->bei', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bei,bei->bei', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bei,bei->bei', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv], axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_complete(self, x_dict, y_dict, weights):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        Umatrix_v, Umatrix_mu, Umatrix_x, dUmatrix_v, dUmatrix_mu, dUmatrix_x = self.net.construct_matrix(x_dict) 
        
        basis = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, Umatrix_x)
        db_vx = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,0], Umatrix_mu, Umatrix_x)
        db_vy = tf.einsum('bjs,bj,beij->beisj', dUmatrix_v[:,:,:,1], Umatrix_mu, Umatrix_x)
        db_mu = tf.einsum('bj,bjs,beij->beisj', Umatrix_v, dUmatrix_mu, Umatrix_x)
        db_x = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,0])
        db_y = tf.einsum('bj,bj,beij->beij', Umatrix_v, Umatrix_mu, dUmatrix_x[:,:,:,:,1])

        # Integrate the functions
        Umatrix = self.integrate(weights, basis)
        dUmatrix_vx = self.integrate_derivative(weights, db_vx) 
        dUmatrix_vy = self.integrate_derivative(weights, db_vy) 
        dUmatrix_mu = self.integrate_derivative(weights, db_mu) 
        dUmatrix_x = self.integrate(weights, db_x) 
        dUmatrix_y = self.integrate(weights, db_y) 
        
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
        vdUmatrix_x_fv = tf.einsum('bei,bei->bei', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bei,bei->bei', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bei,bei->bei', tf.sqrt(x_dict['DT']), y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bei,bei->bei', tf.sqrt(x_dict['DT']), y_dict['grad(T)_y'])
        mudUmatrix_fv = tf.sqrt(tf.square(mudUmatrix_x_fv) + tf.square(mudUmatrix_y_fv))
        dUmatrix_x_fv = tf.gather(y_dict['jacUx(T)'], self.random_sampling, axis=2)
        dUmatrix_y_fv = tf.gather(y_dict['jacUy(T)'], self.random_sampling, axis=2)
        dUmatrix_mu_fv = tf.gather(y_dict['jacMu(T)'], self.random_sampling, axis=2)
        dUmatrix_x_fv_sum = tf.reduce_sum(dUmatrix_x_fv, axis=2)
        dUmatrix_y_fv_sum = tf.reduce_sum(dUmatrix_y_fv, axis=2)
        dUmatrix_mu_fv_sum = tf.reduce_sum(dUmatrix_mu_fv, axis=2)
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, dUmatrix_x_fv_sum, dUmatrix_y_fv_sum, dUmatrix_mu_fv_sum], axis=1)
        
        return Amatrix, Bvector
        
    def resolve_LS(self, x_dict, y_dict, weights):
        '''Resolves the Least-Squares system and updates the weights of the 
        linear layer'''

        # Construct LS system
        A, b = self.LS_constructor(x_dict, y_dict, weights)

        # Resolver LS system
        A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
        b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
        alpha_new = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-10))
        
        # Update linear layer weights
        computable_vars = self.net.lineal_layer.weights[0]
        computable_vars.assign(alpha_new)
  
        return 
    
    # def construct_residual(self, inputs, points, weights):
        
    #     x_dict, y_dict = inputs   
    #     x_dict_mod = x_dict.copy()
    #     x_dict_mod.update(points)
    #     x_dict_mod['v_x'] = tf.tile(x_dict['v_x'], [1,1,4])
    #     x_dict_mod['v_y'] = tf.tile(x_dict['v_y'], [1,1,4])
    #     x_dict_mod['DT'] = tf.tile(x_dict['DT'], [1,1,4])

    #     Umatrix, dUmatrix, ddUmatrix = self.net.construct_matrix_residual(x_dict_mod) 
        
    #     U = self.integrate(weights, Umatrix) #Integration
    #     vdU_x = tf.einsum('bei,beij->beij', x_dict_mod['v_x'], dUmatrix_x)
    #     vdU_y = tf.einsum('bei,beij->beij', x_dict_mod['v_y'], dUmatrix_y)
    #     vdU_x = self.integrate(weights, vdUmatrix_x) #Integration
    #     vdU_y = self.integrate(weights, vdUmatrix_y) #Integration
    #     muddU_x = tf.einsum('bei,beij->beij', tf.sqrt(x_dict_mod['DT']), dUmatrix_x)
    #     muddU_y = tf.einsum('bei,beij->beij', tf.sqrt(x_dict_mod['DT']), dUmatrix_y)
    #     muddU_x = self.integrate(weights, mudUmatrix_x) #Integration
    #     muddU_y = self.integrate(weights, mudUmatrix_y) #Integration
    #     self.net.lineal_layer()
    #     return
    
    # def weighted_loss(self, unweighted_loss):
        
    #     items = unweighted_loss.shape[1]
    #     nL = int(items/self.grid.ncells)
    #     mean_loss = tf.reduce_sum(unweighted_loss)/nL
        

    #     Ls = tf.split(unweighted_loss, num_or_size_splits=nL, axis=1)
    #     w = [mean_loss/(tf.reduce_sum(Li) + 1e-10) for Li in Ls]
    #     contributions = [wi*tf.reduce_sum(Li) for wi,Li in zip(w,Ls)]
    #     weighted_loss = tf.reduce_sum(contributions)/self.grid.ncells
        
    #     parcialLosses = [Li/self.grid.ncells for Li in contributions]
    #     # L1, L2, L3, L4, L5, L6 = tf.split(unweighted_loss, num_or_size_splits=nL, axis=1)
        
    #     # w1 = mean_loss/(tf.reduce_sum(L1) + 1e-10)
    #     # w2 = mean_loss/(tf.reduce_sum(L2) + 1e-10)
    #     # w3 = mean_loss/(tf.reduce_sum(L3) + 1e-10)
    #     # w4 = mean_loss/(tf.reduce_sum(L4) + 1e-10)
    #     # w5 = mean_loss/(tf.reduce_sum(L5) + 1e-10)
    #     # w6 = mean_loss/(tf.reduce_sum(L6) + 1e-10)
        
    #     # weighted_loss = (w1*tf.reduce_sum(L1) + w2*tf.reduce_sum(L2) 
    #     # + w3*tf.reduce_sum(L3) + w4*tf.reduce_sum(L4) + w5*tf.reduce_sum(L5)
    #     # + w6*tf.reduce_sum(L6))/self.grid.ncells
        
    #     return weighted_loss, parcialLosses
    
    def train_step(self, data):
        ''' Training loop'''
        
        x_dict, y_dict = data 
        
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        weights = tf.tile(self.integ_weights,[batch_size,1,1])
        # Prepare data for DeepONet
        x_curated = self.prepare_data(x_dict)

        # Construct and resolve LS and update weights of linear layer
        self.resolve_LS(x_curated, y_dict, weights)
        
        trainable_vars = self.net.trainable_variables[:-1]
        # Ls = self.giveme_loss(data)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(trainable_vars)
            A, b = self.LS_constructor(x_curated, y_dict, weights)
            loss = tf.reduce_sum(tf.square(self.net.lineal_layer(A)-b))/(self.grid.ncells)
            # unweighted_loss = tf.square(self.net.lineal_layer(A)-b)
            # loss, Ls = self.weighted_loss(unweighted_loss)
            
        (U, dU_vx, dU_vy, dU_mu, dU_x, dU_y) = self.call(x_dict)    
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))

        result = {'loss': loss,
                  'mse_u': tf.reduce_mean(tf.keras.metrics.mse(y_dict['T'], U)),
                  'mse_grad_u': tf.reduce_mean(tf.keras.metrics.mse([y_dict['grad(T)_x'], y_dict['grad(T)_y']] , [dU_x, dU_y])),
                  'mse_grad_u_x': tf.reduce_mean(tf.keras.metrics.mse(y_dict['grad(T)_x'], dU_x)),
                  'mse_grad_u_y': tf.reduce_mean(tf.keras.metrics.mse(y_dict['grad(T)_y'], dU_y))}
        
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y_train, y_pred)
        # print([metric for metric in self.metrics])     			
        # Return a dict mapping metric names to current value
        # metrics = {m.name: m.result() for m in self.metrics}
        return result
    
    def test_step(self, data):
        ''' Validation loop'''
        
        x_dict, y_dict = data 
        
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
        weights = tf.tile(self.integ_weights,[batch_size,1,1])
        # Prepare data for DeepONet
        x_curated = self.prepare_data(x_dict)
        
        A, b = self.LS_constructor(x_curated, y_dict, weights)
        loss = tf.reduce_sum(tf.square(self.net.lineal_layer(A)-b))/(self.grid.ncells)      
        (U, dU_vx, dU_vy, dU_mu, dU_x, dU_y) = self.call(x_dict)    

        result = {'loss': loss,
                  'mse_u': tf.reduce_mean(tf.keras.metrics.mse(y_dict['T'], U)),
                  'mse_grad_u': tf.reduce_mean(tf.keras.metrics.mse([y_dict['grad(T)_x'], y_dict['grad(T)_y']] , [dU_x, dU_y])),
                  'mse_grad_u_x': tf.reduce_mean(tf.keras.metrics.mse(y_dict['grad(T)_x'], dU_x)),
                  'mse_grad_u_y': tf.reduce_mean(tf.keras.metrics.mse(y_dict['grad(T)_y'], dU_y))}
        
        return result
       
    # def giveme_loss(self, data):
    #     batch_size = data[0][list(data[0].keys())[0]].shape[0]
    #     points, weights = self.grid.generate_integration_points_and_weights(batch_size)
    #     A, b = self.construct_LS(data, points, weights)
    #     loss = tf.square(self.net.lineal_layer(A)-b)
        
    #     items = loss.shape[1]
    #     nL = int(items/self.grid.ncells)
    #     Ls = tf.split(loss, num_or_size_splits=nL, axis=1)
    #     Ls = [tf.reduce_sum(i)/self.grid.ncells for i in Ls]
        
    #     return Ls
        
    # def train_step_BFGS(self, data):
        
    #     #Generate integration points and weights
    #     batch_size = data[0][list(data[0].keys())[0]].shape[0]
    #     points, weights = self.grid.generate_integration_points_and_weights(batch_size)
        
    #     # A, b = self.construct_LS(data, points, weights)
        
    #     # computable_variables = self.net.lineal_layer.weights[0]
    #     # A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
    #     # b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
        
    #     # alpha_new = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=10**(-10))
    #     # computable_variables.assign(alpha_new)
        
    #     self.resolve_LS(data)
        
    #     A, b = self.construct_LS(data, points, weights)
    #     loss = tf.reduce_sum(tf.square(self.net.lineal_layer(A)-b))/(self.grid.ncells)
    #     # loss = tf.reduce_sum(tf.abs(self.net.lineal_layer(A)-b))/self.grid.size_x
    #     return loss

    def prepare_data(self, data):
        '''Prepares the data for the DeepONet: random sampling and extracting 
        values of the integrating points'''
        
        resulting_data = {}
        batch_size = data[list(data.keys())[0]].shape[0]
        
        # Random sampling velocity and difussivity
        v_x_sampled = tf.gather(data['v_x'], self.random_sampling, axis=1)
        v_y_sampled = tf.gather(data['v_y'], self.random_sampling, axis=1)
        DT_sampled = tf.gather(data['DT'], self.random_sampling, axis=1)
        
        # Generate integration points
        coord_x = tf.tile(self.integ_points['coord_x'],[batch_size,1,1])
        coord_y = tf.tile(self.integ_points['coord_y'],[batch_size,1,1])
        
        resulting_data['v_x'] = data['v_x']
        resulting_data['v_y'] = data['v_y']
        resulting_data['DT'] = data['DT']
        resulting_data['v_x_sampled'] = v_x_sampled
        resulting_data['v_y_sampled'] = v_y_sampled
        resulting_data['DT_sampled'] = DT_sampled
        resulting_data['coord_x'] = coord_x
        resulting_data['coord_y'] = coord_y
        
        return resulting_data
    
    def call(self, data):
        '''Prepares the data, calls the DeepONet and integrates de results'''
        
        #Generate integration points and weights (if random)
        batch_size = data[list(data.keys())[0]].shape[0]
        # points, weights = self.grid.generate_integration_points_and_weights(batch_size)
        weights = tf.tile(self.integ_weights,[batch_size,1,1])
        
        # Prepare data for DeepONet
        data_curated = self.prepare_data(data)
        
        # Call DeepONet and lineal layer
        U, dU_vx, dU_vy, dU_mu, dU_x, dU_y = self.net(data_curated) 
        
        # Integrate results 
        U = self.integrate(weights, U) 
        dU_vx = self.integrate_derivative(weights, dU_vx) 
        dU_vy = self.integrate_derivative(weights, dU_vy) 
        dU_mu = self.integrate_derivative(weights, dU_mu) 
        dU_x = self.integrate(weights, dU_x) 
        dU_y = self.integrate(weights, dU_y) 
        
        return (U, dU_vx, dU_vy, dU_mu, dU_x, dU_y)


class LSonEpoch(tf.keras.callbacks.Callback):
   def __init__(self, x_train, y_train):
       super().__init__()
       self.x_train = x_train
       self.y_train = y_train

   def on_epoch_begin(self, epoch, logs=None):
       self.model.resolve_LS((self.x_train,self.y_train))

# you can then create the callback by passing the correct attributes
# my_callback = CustomCallback(testRatings, testNegatives, topK, evaluation_threads)

#%% 
def plot_field(data, grid, field, vmin, vmax):
    fig, ax = plt.subplots(1, 1)
    X, Y = np.meshgrid(grid.axis_x, grid.axis_y)
    Z = np.reshape(data, [grid.size_x, grid.size_y])
    im = ax.pcolormesh(grid.axis_x, grid.axis_y, Z, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(field)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    return fig

#Plot some data
# plot_field(y_train['T'][10,:], grid, 'u')
# plot_field(y_train['vgrad(T)_x'][10,:], grid, 'vgrad(u)_x')

#%%
# Construct the grid
grid = grids2D.Grid(size_x = 50, size_y = 50, step_size = 0.02)

# Data for training
data_route = '../../OpenFOAM/convectionDiffusion2D/training_dataSS/'
training_data = fRW.upload_training_data(data_route)
x_train, y_train, _, _ = prepare_raw_data(training_data, train_split=1.)

# Data for validation
data_route = '../../OpenFOAM/convectionDiffusion2D/validation_dataSS/'
validation_data = fRW.upload_training_data(data_route)
x_val, y_val, _, _ = prepare_raw_data(validation_data, train_split=1.)

# Create learning model
net = DeepONet(layers_branch_v=[20,20,10], layers_branch_mu=[20,20,10], 
               layers_trunk=[10,10,10], num_rand_sampling = 100, dimension='2D')
model = my_model(net, grid, 'complete')

#Save and load weights
# model.save_weights('weights_op1.h5')
# U, dU_vx, dU_vy, dU_mu, dU_x, dU_y = model(x_train)
# model.load_weights('weights_op1.h5')

#Compilation of the model
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=10**(-7)))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10**(-2)))
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5, patience=15, 
                                                 min_lr=10e-7, min_delta=0.001, 
                                                 verbose=1)
history = model.fit(x_train, y_train, 
                    epochs=20, batch_size=4)#, callbacks=[reduce_lr])

df_history = pd.DataFrame(history.history)

df_history.to_csv(f'model1_training.csv', index=False)
# df_history_1 = reloaded_df_history = pd.read_csv('0.training_full_code_400it.csv')

#Plots of the loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.yscale("log")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.savefig(f'figures/C1.loss_log_50x50elems_full.png', dpi=200, bbox_inches='tight')
plt.show()

#Plot the MSE of u
plt.plot(history.history['mse_u'], label='u error training')
plt.plot(history.history['val_mse_u'], label='u error validation')
plt.yscale("log")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')
# plt.savefig(f'figures/C1.loss_log_50x50elems_full.png', dpi=200, bbox_inches='tight')
plt.show()

#Plot the MSE of grad(u)
plt.plot(history.history['mse_grad_u'], label='grad(u) error training')
plt.plot(history.history['val_mse_grad_u'], label='grad(u) error validation')
plt.yscale("log")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MSE')
# plt.savefig(f'figures/C1.loss_log_50x50elems_full.png', dpi=200, bbox_inches='tight')
plt.show()

# Bar plot comparing models
models = ("Model 1", "Model 2", "Model 3", "Model 4")
variables = {
    'Loss': (18.35, 18.43, 14.98, 19.54),
    'MSE u': (38.79, 48.83, 47.50, 24.56),
    'MSE grad(u)': (189.95, 195.82, 217.19, 200.34),
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in variables.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss/MSE')
ax.set_title('Comparative of models performance')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)
# plt.savefig(f'figures/C1.loss_log_50x50elems_full.png', dpi=200, bbox_inches='tight')
plt.show()

#%% COMPARISON OF MODELS

# Construct the grid
grid = grids2D.Grid(size_x = 50, size_y = 50, step_size = 0.02)

# Data for training
data_route = '../../OpenFOAM/convectionDiffusion2D/training_data/'
training_data = fRW.upload_training_data(data_route, jacobian=True)
x_train, y_train, _, _ = prepare_raw_data(training_data, train_split=1.)

# Data for validation
data_route = '../../OpenFOAM/convectionDiffusion2D/validation_data/'
validation_data = fRW.upload_training_data(data_route, jacobian=True)
x_val, y_val, _, _ = prepare_raw_data(validation_data, train_split=1.)

# Create learning model
net = DeepONet(layers_branch_v=[20,20,10], layers_branch_mu=[20,20,10], 
               layers_trunk=[10,10,10], num_rand_sampling = 100, dimension='2D')
model1 = my_model(net, grid, 'vanilla')
model2 = my_model(net, grid, 'physics_1')
model3 = my_model(net, grid, 'physics_2')
model4 = my_model(net, grid, 'complete')

model = [model1, model2, model3, model4]
result = []
for i in range(len(model)):
    model = model[i]
    
    #Compilation of the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10**(-2)))
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5, patience=15, 
                                                 min_lr=10e-7, min_delta=0.001, 
                                                 verbose=1)
    history = model.fit(x_train, y_train, validation_data = (x_val, y_val), 
                    epochs=2000, batch_size=8, callbacks=[reduce_lr])
    
    history = model.fit(x_train, y_train, 
                    epochs=2000, batch_size=8, callbacks=[reduce_lr])
    
    model.save_weights(f'model{i+1}_weights.h5')
    result.append(history)
    df_history = pd.DataFrame(history.history)
    df_history.to_csv(f'model{i+1}_training.csv', index=False)

    #Plots of the loss
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'figures/model{i+1}_loss.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    #Plot the MSE of u
    plt.plot(history.history['mse_u'], label='u error training')
    plt.plot(history.history['val_mse_u'], label='u error validation')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig(f'figures/C1.model{i+1}_u.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    #Plot the MSE of grad(u)
    plt.plot(history.history['mse_grad_u'], label='grad(u) error training')
    plt.plot(history.history['val_mse_grad_u'], label='grad(u) error validation')
    plt.yscale("log")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig(f'figures/model{i+1}_gradu.png', dpi=200, bbox_inches='tight')
    plt.show()

# Bar plot comparing models on training
models = ("Model 1", "Model 2", "Model 3", "Model 4")
variables = {
    'Loss': (result[0].history['loss'][-1],
             result[1].history['loss'][-1],
             result[2].history['loss'][-1],
             result[3].history['loss'][-1]),
    'MSE u': (result[0].history['mse_u'][-1],
              result[1].history['mse_u'][-1],
              result[2].history['mse_u'][-1],
              result[3].history['mse_u'][-1]),
    'MSE grad(u)': (result[0].history['mse_grad_u'][-1],
                    result[1].history['mse_grad_u'][-1],
                    result[2].history['mse_grad_u'][-1],
                    result[3].history['mse_grad_u'][-1]),
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in variables.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss or MSE')
ax.set_title('Comparative of models performance on training')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)
plt.savefig('figures/models_training_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

# Bar plot comparing models on validation
models = ("Model 1", "Model 2", "Model 3", "Model 4")
variables = {
    'Loss': (result[0].history['val_loss'][-1],
             result[1].history['val_loss'][-1],
             result[2].history['val_loss'][-1],
             result[3].history['val_loss'][-1]),
    'MSE u': (result[0].history['val_mse_u'][-1],
              result[1].history['val_mse_u'][-1],
              result[2].history['val_mse_u'][-1],
              result[3].history['val_mse_u'][-1]),
    'MSE grad(u)': (result[0].history['val_mse_grad_u'][-1],
                    result[1].history['val_mse_grad_u'][-1],
                    result[2].history['val_mse_grad_u'][-1],
                    result[3].history['val_mse_grad_u'][-1]),
}

x = np.arange(len(models))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in variables.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Loss or MSE')
ax.set_title('Comparative of models performance on training')
ax.set_xticks(x + width, models)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)
plt.savefig('figures/models_training_comparison.png', dpi=200, bbox_inches='tight')
plt.show()

#%% PLOTING OF PREDICTION
U, dU_vx, dU_vy, dU_mu, dU_x, dU_y = model(x_train)

sample = 0
vx = x_train['v_x'][sample,0,0]
vy = x_train['v_y'][sample,0,0]
mu = x_train['DT'][sample,0,0]
print(f'vx = {vx}\nvy= {vy}\nmu= {mu}\n')
u_max = tf.reduce_max(y_train['T'][sample,:])
u_min = tf.reduce_min(y_train['T'][sample,:])
dux_max = tf.reduce_max(y_train['grad(T)_x'][sample,:])
dux_min = tf.reduce_min(y_train['grad(T)_x'][sample,:])
duy_max = tf.reduce_max(y_train['grad(T)_y'][sample,:])
duy_min = tf.reduce_min(y_train['grad(T)_y'][sample,:])
dumu_max = tf.reduce_max(y_train['gradMu(T)'][sample,:])
dumu_min = tf.reduce_min(y_train['gradMu(T)'][sample,:])
p1 = plot_field(U[sample,:], grid, 'u_NN', u_min, u_max)
p2 = plot_field(y_train['T'][sample,:], grid, 'u_true', u_min, u_max)
p3 = plot_field(dux[sample,:], grid, 'grad(u)_x_NN', dux_min, dux_max)
p4 = plot_field(y_train['grad(T)_x'][sample,:], grid, 'grad(u)_x_true', dux_min, dux_max)
p5 = plot_field(duy[sample,:], grid, 'grad(u)_y_NN', duy_min, duy_max)
p6 = plot_field(y_train['grad(T)_y'][sample,:], grid, 'grad(u)_y_true', duy_min, duy_max)
p7 = plot_field(dumu[sample,:], grid, 'grad(u)_mu_NN', dumu_min, dumu_max)
p8 = plot_field(y_train['gradMu(T)'][sample,:], grid, 'grad(u)_mu_true', dumu_min, dumu_max)

p1.savefig(f'figures/C1.u_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p2.savefig(f'figures/C1.u_true_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p3.savefig(f'figures/C1.dux_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p4.savefig(f'figures/C1.dux_true_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p5.savefig(f'figures/C1.duy_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p6.savefig(f'figures/C1.duy_true_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p7.savefig(f'figures/C1.dumu_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p8.savefig(f'figures/C1.dumu_true_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')

#%% PLOTTING PREDICTION DIFFERENT (NO INTEGRATION) POINTS
num = 100
coord_x_test = np.linspace(0,1,num)
coord_y_test = np.linspace(0,1,num)
nx,ny = np.meshgrid(coord_x_test, coord_y_test)
nx = np.reshape(nx, -1)
ny = np.reshape(ny, -1)
coords_x = [np.expand_dims(nx, axis=1) for i in range(8)]
coords_y = [np.expand_dims(ny, axis=1) for i in range(8)]
ones = tf.ones((nx.shape[0],1), dtype=tf.float64)
v_x = [ones*x_train['v_x'][0,0,0],
       ones*x_train['v_x'][1,0,0],
       ones*x_train['v_x'][2,0,0],
       ones*x_train['v_x'][3,0,0],
       ones*x_train['v_x'][4,0,0],
       ones*x_train['v_x'][5,0,0],
       ones*x_train['v_x'][6,0,0],
       ones*x_train['v_x'][7,0,0]]
v_y = [ones*x_train['v_y'][0,0,0],
       ones*x_train['v_y'][1,0,0],
       ones*x_train['v_y'][2,0,0],
       ones*x_train['v_y'][3,0,0],
       ones*x_train['v_y'][4,0,0],
       ones*x_train['v_y'][5,0,0],
       ones*x_train['v_y'][6,0,0],
       ones*x_train['v_y'][7,0,0]]
DT  = [ones*x_train['DT'][0,0,0],
       ones*x_train['DT'][1,0,0],
       ones*x_train['DT'][2,0,0],
       ones*x_train['DT'][3,0,0],
       ones*x_train['DT'][4,0,0],
       ones*x_train['DT'][5,0,0],
       ones*x_train['DT'][6,0,0],
       ones*x_train['DT'][7,0,0]]
x_val={}
x_val['v_x'] = tf.stack(v_x)
x_val['v_y'] = tf.stack(v_y)
x_val['coord_x'] = tf.stack(coords_x)
x_val['coord_y'] = tf.stack(coords_y)
x_val['DT'] = tf.stack(DT)

u_val, gradux_val, graduy_val, gradumu_val = net(x_val)
grid_val = grids2D.Grid(size_x = num, size_y = num, step_size = 1/num)
sample = 0
u_max = tf.reduce_max(y_train['T'][sample,:])
u_min = tf.reduce_min(y_train['T'][sample,:])
dux_max = tf.reduce_max(y_train['grad(T)_x'][sample,:])
dux_min = tf.reduce_min(y_train['grad(T)_x'][sample,:])
duy_max = tf.reduce_max(y_train['grad(T)_y'][sample,:])
duy_min = tf.reduce_min(y_train['grad(T)_y'][sample,:])
dumu_max = tf.reduce_max(y_train['gradMu(T)'][sample,:])
dumu_min = tf.reduce_min(y_train['gradMu(T)'][sample,:])

vol = 0.0004
p9 = plot_field(u_val[sample,:]*vol,grid_val, 'u_NN', u_min, u_max)
plot_field(y_train['T'][sample,:], grid, 'u_true', u_min, u_max)
p10 = plot_field(gradux_val[sample,:]*vol, grid_val, 'grad(u)_x_NN', dux_min, dux_max)
plot_field(y_train['grad(T)_x'][sample,:], grid, 'grad(u)_x_true', dux_min, dux_max)
p11 = plot_field(graduy_val[sample,:]*vol, grid_val, 'grad(u)_y_NN', duy_min, duy_max)
plot_field(y_train['grad(T)_y'][sample,:], grid, 'grad(u)_y_true', duy_min, duy_max)
p12 = plot_field(gradumu_val[sample,:]*vol, grid_val, 'grad(u)_mu_NN', dumu_min, dumu_max)
plot_field(y_train['gradMu(T)'][sample,:], grid, 'grad(u)_mu_true', dumu_min, dumu_max)

p9.savefig(f'figures/C2.u_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p10.savefig(f'figures/C2.dux_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p11.savefig(f'figures/C2.duy_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p12.savefig(f'figures/C2.dumu_pred_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')

#Plotting of the quadrature points
x_val={}
points, weights = grid.generate_integration_points_and_weights(4)
x_val['coord_x'] = tf.reshape(points['coord_x'], [4,-1,1])
x_val['coord_y'] = tf.reshape(points['coord_y'], [4,-1,1])
x_val['coord_x'] = points['coord_x']
x_val['coord_y'] = points['coord_y']
ones = tf.ones((x_val['coord_x'].shape[1:]), dtype=tf.float64)
v_x = [ones*x_train['v_x'][0,0,0],
       ones*x_train['v_x'][1,0,0],
       ones*x_train['v_x'][2,0,0],
       ones*x_train['v_x'][3,0,0]]
v_y = [ones*x_train['v_y'][0,0,0],
       ones*x_train['v_y'][1,0,0],
       ones*x_train['v_y'][2,0,0],
       ones*x_train['v_y'][3,0,0]]
x_val['v_x'] = tf.stack(v_x)
x_val['v_y'] = tf.stack(v_x)

u_val_2, dux_val_2, duy_val_2 = net(x_val)

u_val_res = tf.einsum('bei,beij->bej', weights, tf.squeeze(u_val_2))
vgradux_val_res = tf.einsum('bei,beij->bej', weights, tf.squeeze(dux_val_2))
vgraduy_val_res = tf.einsum('bei,beij->bej', weights, tf.squeeze(duy_val_2))

sample = 3
vol = 0.0004
plot_field(u_val_res[sample,:]*vol,grid, 'u_NN')
plot_field(y_train['T'][sample,:], grid, 'u_true')
plot_field(vgradux_val_res[sample,:]*vol, grid, 'vgrad(u)_x_NN')
plot_field(y_train['vgrad(T)_x'][sample,:], grid, 'vgrad(u)_x_true')
plot_field(vgraduy_val_res[sample,:]*vol, grid, 'vgrad(u)_y_NN')
plot_field(y_train['vgrad(T)_y'][sample,:], grid, 'vgrad(u)_y_true')

#%% PLOTTING PREDICTION OUT OF TRAINING DATA

# Data for validation
data_route = '../OpenFOAM/convectionDiffusion2D/validation_dataSS/'
validation_data = fRW.upload_training_data(data_route)
x_val, y_val, _, _ = prepare_raw_data(validation_data, train_split=1.)

u_val, dux_val, duy_val, dumu_val = model(x_val)
sample = 4
vol = 0.0004
u_max = tf.reduce_max(y_val['T'][sample,:])
u_min = tf.reduce_min(y_val['T'][sample,:])
dux_max = tf.reduce_max(y_val['grad(T)_x'][sample,:])
dux_min = tf.reduce_min(y_val['grad(T)_x'][sample,:])
duy_max = tf.reduce_max(y_val['grad(T)_y'][sample,:])
duy_min = tf.reduce_min(y_val['grad(T)_y'][sample,:])
dumu_max = tf.reduce_max(y_val['gradMu(T)'][sample,:])
dumu_min = tf.reduce_min(y_val['gradMu(T)'][sample,:])
p13 = plot_field(u_val[sample,:],grid, 'u_NN', u_min, u_max)
p14 = plot_field(y_val['T'][sample,:], grid, 'u_true', u_min, u_max)
p15 = plot_field(dux_val[sample,:], grid, 'grad(u)_x_NN', dux_min, dux_max)
p16 = plot_field(y_val['grad(T)_x'][sample,:], grid, 'grad(u)_x_true', dux_min, dux_max)
p17 = plot_field(duy_val[sample,:], grid, 'grad(u)_y_NN', duy_min, duy_max)
p18 = plot_field(y_val['grad(T)_y'][sample,:], grid, 'grad(u)_y_true', duy_min, duy_max)
p19 = plot_field(dumu_val[sample,:], grid, 'grad(u)_mu_NN', dumu_min, dumu_max)
p20 = plot_field(y_val['gradMu(T)'][sample,:], grid, 'grad(u)_mu_true', dumu_min, dumu_max)

vx = x_val['v_x'][sample,0,0]
vy = x_val['v_y'][sample,0,0]
mu = x_val['DT'][sample,0,0]
print(f'vx = {vx}\nvy= {vy}\nmu= {mu}\n')
p13.savefig(f'figures/C3.u_pred_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p14.savefig(f'figures/C3.u_true_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p15.savefig(f'figures/C3.dux_pred_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p16.savefig(f'figures/C3.dux_true_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p17.savefig(f'figures/C3.duy_pred_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p18.savefig(f'figures/C3.duy_true_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p19.savefig(f'figures/C3.dumu_pred_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')
p20.savefig(f'figures/C3.dumu_true_val_50x50elems_{sample}_{vx}-{vy}-{mu}_full.png', dpi=200, bbox_inches='tight')