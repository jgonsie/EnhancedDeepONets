#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:12:34 2024

@author: jesusglezs97
"""
import numpy as np
import tensorflow as tf
import random
from scipy.stats import qmc

class Grid():
    def __init__(self, size_x, size_y, step_size, seed=42):
        
        self.dim = 2
        self.size_x = size_x
        self.size_y = size_y
        self.step = step_size
        #TODO: Take as input step_size_x and step_size_y
        self.area = step_size*step_size
        self.ncells = size_x * size_y
        self.grid = self.get_mesh()
        self.n_points_by_method = {'GaussP1': 2*self.dim,
                                   'GaussP2': 2*self.dim,
                                   'MC':      4,
                                   'QMC':     16,
                                   'centroids': 1}
        
        self.RNG = np.random.default_rng(seed)
        self.qmc_sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.RNG)
        
    def get_mesh(self, shift = (0, 0)):
        """Generates the grid mesh"""
        
        if len(shift) != 2:
          raise ValueError('shift length must be equal to two')
        half_step = self.step / 2.
        shift_x = (1 + shift[0]) * half_step
        shift_y = (1 + shift[1]) * half_step
        self.axis_x = shift_x + self.step * np.arange(self.size_x)
        self.axis_y = shift_y + self.step * np.arange(self.size_y)
        return np.meshgrid(
            self.axis_x,
            self.axis_y,
            indexing='ij')
    
    def generate_quadrature_gauss_p1_random(self, batch=1):
        '''Generate random integration points and weights using Gauss for p1'''
        
        def generate_weights(points_x, points_y):
            '''Generates integration weights for exact integration in p=1'''
            
            def weights_x(coords):
                w11 = (coords[:,:,1:2] - 1/2) / (coords[:,:,1:2] - coords[:,:,0:1])
                w21 = 1 - w11
                w12 = (coords[:,:,3:4] - 1/2) / (coords[:,:,3:4] - coords[:,:,2:3])
                w22 = 1 - w12
            
                return np.concatenate((w11, w21, w12, w22), axis=-1)
            
            def weights_y(coords):
                w11 = (coords[:,:,2:3] - 1/2) / (coords[:,:,2:3] - coords[:,:,0:1])
                w12 = 1 - w11
                w21 = (coords[:,:,3:4] - 1/2) / (coords[:,:,3:4] - coords[:,:,1:2])
                w22 = 1 - w21
            
                return np.concatenate((w11, w21, w12, w22), axis=-1)
            
            wx = weights_x(points_x)
            wy = weights_y(points_y)
            
            return np.einsum('bei,bei->bei', wx, wy)
        
        points = {}
        weights = {}
        
        # Generate random points on unity square
        x11 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,self.ncells)), axis=2)
        x21 = np.expand_dims(np.random.uniform(low=0.5, high=1.0,
                                              size=(batch,self.ncells)), axis=2)
        x12 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,self.ncells)), axis=2)
        x22 = np.expand_dims(np.random.uniform(low=0.5, high=1.0,
                                              size=(batch,self.ncells)), axis=2)
        y11 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,self.ncells)), axis=2)
        y21 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,self.ncells)), axis=2)
        y12 = np.expand_dims(np.random.uniform(low=0.5, high=1.0,
                                              size=(batch,self.ncells)), axis=2)
        y22 = np.expand_dims(np.random.uniform(low=0.5, high=1.0,
                                              size=(batch,self.ncells)), axis=2)

        # Gather points and generate weights on unity square
        points_x_unity = np.concatenate((x11, x21, x12, x22), axis=2) 
        points_y_unity = np.concatenate((y11, y21, y12, y22), axis=2) 
        weights_unity = generate_weights(points_x_unity, points_y_unity)
        
        # Transform points and weights of unity square to real mesh
        points_x = self.step * points_x_unity + np.tile(np.expand_dims(self.axis_x - (self.step/2),axis=1),[batch,self.size_x,4])
        points_y = self.step * points_y_unity + np.tile(np.expand_dims(np.repeat(self.axis_y-(self.step/2), repeats=self.size_y),axis=1),[batch,1,4])
        weights = self.area * weights_unity
        
        # Convert values to tf.Tensors
        points['coord_x'] = tf.convert_to_tensor(points_x)
        points['coord_y'] = tf.convert_to_tensor(points_y)
        weights = tf.convert_to_tensor(weights)
        
        return (points, weights)
    
    def generate_quadrature_gauss_p2(self, batch=1):
        '''Generate integration points and weights using Guass for p2'''
        
        points = {}
        weights = {}
        
        x1 = (0.5-np.sqrt(3)/6) * np.ones((1, len(self.axis_x),1))
        x2 = (0.5+np.sqrt(3)/6) * np.ones((1, len(self.axis_x),1))
        y1 = (0.5-np.sqrt(3)/6) * np.ones((1, len(self.axis_y),1))
        y2 = (0.5+np.sqrt(3)/6) * np.ones((1, len(self.axis_y),1))
        points_x_1D = self.step * np.concatenate((x1, x2), axis=2) + np.tile(np.expand_dims(self.axis_x - (self.step/2),axis=1),[batch,1,2])
        points_y_1D = self.step * np.concatenate((y1, y2), axis=2) + np.tile(np.expand_dims(self.axis_y - (self.step/2),axis=1),[batch,1,2])        
        
        mesh_x, mesh_y = np.meshgrid(points_x_1D, points_y_1D)
        points_x = np.zeros((self.size_x*self.size_y,4))
        points_y = np.zeros((self.size_x*self.size_y,4))
        elem = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                points_x[elem,:] = np.reshape(mesh_x[2*y:2*y+2,2*x:2*x+2], [-1])
                points_y[elem,:] = np.reshape(mesh_y[2*y:2*y+2,2*x:2*x+2], [-1])
                elem += 1
        
        points['coord_x'] = np.tile(tf.convert_to_tensor(np.expand_dims(points_x, axis=0)), [batch,1,1])
        points['coord_y'] = np.tile(tf.convert_to_tensor(np.expand_dims(points_y, axis=0)), [batch,1,1])
        
        weights = self.area * np.ones(points['coord_x'].shape) * 0.25
        
        return (points, weights)
    
    def generate_quadrature_mc(self, batch=1, npoints=10):
        '''Generate integration points and weights using Montecarlo quadrature'''
        
        points = {}
        weights = {}
        
        # Generate points and weights on unity square
        random_points = self.RNG.uniform(size=(batch, self.ncells, npoints, self.dim))
        points_x_unity = random_points[:,:,:,0]
        points_y_unity = random_points[:,:,:,1]
        weights_unity = np.ones(random_points.shape[:-1])/npoints
        
        # Transform points and weights of unity square to real mesh
        points_x = self.step * points_x_unity + np.tile(np.expand_dims(self.axis_x - (self.step/2),axis=1),[batch,self.size_x,npoints])
        points_y = self.step * points_y_unity + np.tile(np.expand_dims(np.repeat(self.axis_y-(self.step/2), repeats=self.size_y),axis=1),[batch,1,npoints])
        weights = self.area * weights_unity
        
        # Convert values to tf.Tensors
        points['coord_x'] = tf.convert_to_tensor(points_x)
        points['coord_y'] = tf.convert_to_tensor(points_y)
        weights = tf.convert_to_tensor(weights)
        
        return (points, weights)
    
    def generate_quadrature_qmc(self, batch=1, npoints=10):
        '''Generate integration points and weights using Quasi-Montecarlo quadrature'''
        
        points = {}
        weights = {}
        
        # Generate points and weights on unity square
        power = int(np.ceil(np.log2(npoints)))
        random_points = self.qmc_sampler.random_base2(m=power).reshape((-1,2))
        self.qmc_sampler.reset()
        self.qmc_sampler._scramble()
        points_x_unity = np.tile(np.expand_dims(random_points[:,0],axis=(0,1)),[batch,self.ncells,1])
        points_y_unity = np.tile(np.expand_dims(random_points[:,1],axis=(0,1)),[batch,self.ncells,1])
        weights_unity = np.ones((batch,self.ncells,2**power))*(2.**-power)

        # Transform points and weights of unity square to real mesh
        points_x = self.step * points_x_unity + np.tile(np.expand_dims(self.axis_x-(self.step/2),axis=1),[batch,self.size_x,2**power])
        points_y = self.step * points_y_unity + np.tile(np.expand_dims(np.repeat(self.axis_y-(self.step/2), repeats=self.size_y),axis=1),[batch,1,2**power])
        weights = self.area * weights_unity
        
        # Convert values to tf.Tensors
        # points['coord_x'] = tf.convert_to_tensor(points_x)
        # points['coord_y'] = tf.convert_to_tensor(points_y)
        # weights = tf.convert_to_tensor(weights)
        points['coord_x'] = points_x
        points['coord_y'] = points_y
        weights = weights
        
        return (points, weights)
    
    def generate_quadrature_centroids(self, batch=1):
        '''Generate integration points and weights using Quasi-Montecarlo quadrature'''
        
        points = {}
        weights = {}
        
        x = np.asarray(self.axis_x, dtype='float32')
        y = tf.repeat(x,repeats=self.size_y)
        x_exp = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
        y_exp = tf.expand_dims(tf.expand_dims(y, axis=-1), axis=0)
        points['coord_x'] = tf.tile(x_exp,[batch,self.size_y,1])
        points['coord_y'] = tf.tile(y_exp,[batch,1,1])
        
        # weights = tf.ones([batch,self.ncells,1], dtype='float64') * self.area
        weights = tf.ones([batch,self.ncells,1], dtype='float32')
        
        return (points, weights)
    
    def generate_quadrature_tf(self, method, batch=1, n_points=4):
        
        if type(method) == np.ndarray: method = np.array2string(method)[2:-1]
        else: method = method.decode("utf-8")
       
        batch = int(batch)
        n_points = int(n_points)

        if method == 'GaussP1':
            (points, weights) = self.generate_quadrature_gauss_p1_random(batch)
        elif method == 'GaussP2':
            (points, weights) = self.generate_quadrature_gauss_p2(batch)
        elif method == 'MC':
            (points, weights) = self.generate_quadrature_mc(batch, n_points)
        elif method == 'QMC':
            (points, weights) = self.generate_quadrature_qmc(batch, n_points)
        elif method == 'centroids':
            (points, weights) = self.generate_quadrature_centroids (batch)
        else:
            ValueError(f'Quadrature method selected not found: {method}')
        
        return [points['coord_x'], points['coord_y'], weights]
    
    def generate_random_sampling_of_points(self, num_samples):
        
        random.seed(1234)
        random_sampling = random.sample(range(self.ncells), num_samples)
        
        return random_sampling
    
#############################################################################
####TRANSFORMATION OF POINTS AND WEIGHTS FROM UNITY SQUARE TO REAL SQUARE####
#############################################################################

#  1 - - - - 1               c - - - - d
#  |         |               |         |
#  |         |     --TO-->   |         |
#  |         |               |         |
#  0 - - - - 1               a - - - - b
#
#  x_new = a + (b-a) x_unity
#  y_new = c + (c-a) y_unity
#  w_new = (b-a) * (c-a) * w_unity
#############################################################################
#############################################################################


# cells_x = 10
# cells_y = 10
# step_size = 1/cells_x
# grid = Grid(size_x = cells_x, size_y = cells_y, step_size = step_size)
# pointsP1, weightsP1 = grid.generate_quadrature_gauss_p1_random(batch=7)
# pointsP2, weightsP2 = grid.generate_quadrature_gauss_p2(batch=7)
# pointsMC, weightsMC = grid.generate_quadrature_mc(batch=7,npoints=10)
# pointsQMC, weightsQMC = grid.generate_quadrature_qmc(batch=7, npoints=16)
# pointsCENT, weightsCENT = grid.generate_quadrature_centroids(batch=7)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 1)
# ax.scatter(pointsP1['coord_x'][0,0,:], pointsP1['coord_y'][0,0,:], 
#             s=weightsP1[0,0,:]*100000, alpha=0.5)
# ax.set_xlim(0, 1/cells_x)
# ax.set_ylim(0, 1/cells_y)
# ax.set_title('Gauss P1 quadrature')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.scatter(pointsP2['coord_x'][0,0,:], pointsP2['coord_y'][0,0,:], 
#             s=weightsP2[0,0,:]*100000, alpha=0.5)
# ax.set_xlim(0, 1/cells_x)
# ax.set_ylim(0, 1/cells_y)
# ax.set_title('Gauss P2 quadrature')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.scatter(pointsMC['coord_x'][0,0,:], pointsMC['coord_y'][0,0,:], 
#             s=weightsMC[0,0,:]*100000, alpha=0.5)
# ax.set_xlim(0, 1/cells_x)
# ax.set_ylim(0, 1/cells_y)
# ax.set_title('MC quadrature')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.scatter(pointsQMC['coord_x'][0,0,:], pointsQMC['coord_y'][0,0,:], 
#             s=weightsQMC[0,0,:]*100000, alpha=0.5)
# ax.set_xlim(0, 1/cells_x)
# ax.set_ylim(0, 1/cells_y)
# ax.set_title('QMC quadrature')
# plt.tight_layout()
# plt.show()