#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:12:34 2024

@author: jesusglezs97
"""
import numpy as np
import tensorflow as tf
import random


class Grid():
    def __init__(self, size_x, size_y, step_size):
        
        self.size_x = size_x
        self.size_y = size_y
        self.step = step_size
        #TODO: Take as input step_size_x and step_size_y
        self.volume = step_size*step_size
        self.ncells = size_x * size_y
        self.grid = self.get_mesh()
        
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
    
    def generate_weights(self, coord_1, coord_2):
        '''Generates integration weights for exact integration in p=1'''
        
        w1 = (coord_2-1/2)/(coord_2-coord_1)
        w2 = 1 - w1
        
        # w1 = 2*coord_2/(coord_1+coord_2)
        # w2 = 2*coord_1/(coord_1+coord_2)
        return np.concatenate((w1, w2), axis=2)
    
    def generate_integration_points_and_weights(self, batch = 1):
        '''Generate new random integration points and weights in every call'''
        
        points = {}
        weights = {}
        
        x1 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,len(self.axis_x))), axis=2)
        x2 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,len(self.axis_x))), axis=2)
        y1 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,len(self.axis_y))), axis=2)
        y2 = np.expand_dims(np.random.uniform(low=0.0, high=0.5,
                                              size=(batch,len(self.axis_y))), axis=2)

        # points_x = self.step * np.concatenate((-x1, x1), axis=2) + np.tile(np.expand_dims(self.axis_x,axis=1),[1,1,2])
        # points_y = self.step * np.concatenate((-y1, y1), axis=2) + np.tile(np.expand_dims(self.axis_y,axis=1),[1,1,2])
        # weights_x = self.step * self.generate_weights(x1, x2)
        # weights_y = self.step * self.generate_weights(y1, y2)
        
        # points_x = self.step *  np.sqrt(3)/3 * np.ones((batch, len(self.axis_x),2)) + np.tile(np.expand_dims(self.axis_x,axis=1),[1,1,2])
        # points_y = self.step *  np.sqrt(3)/3 * np.ones((batch, len(self.axis_y),2)) + np.tile(np.expand_dims(self.axis_y,axis=1),[1,1,2])
        # weights_x = self.step * np.ones(points_x.shape)
        # weights_y = self.step * np.ones(points_y.shape)
        
        x = (0.5-((3-np.sqrt(3))/6)) * np.ones((batch, len(self.axis_x),1))
        y = (0.5-((3-np.sqrt(3))/6)) * np.ones((batch, len(self.axis_y),1))
        points_x_1D = self.step *  np.concatenate((-x, x), axis=2) + np.tile(np.expand_dims(self.axis_x,axis=1),[1,1,2])
        points_y_1D = self.step *  np.concatenate((-y, y), axis=2) + np.tile(np.expand_dims(self.axis_y,axis=1),[1,1,2])        
        
        mesh_x, mesh_y = np.meshgrid(points_x_1D, points_y_1D)
        self.aa=(mesh_x,mesh_y)
        points_x = np.zeros((self.size_x*self.size_y,4))
        points_y = np.zeros((self.size_x*self.size_y,4))
        elem = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                points_x[elem,:] = np.reshape(mesh_x[2*y:2*y+2,2*x:2*x+2], [-1])
                points_y[elem,:] = np.reshape(mesh_y[2*y:2*y+2,2*x:2*x+2], [-1])
                elem += 1
        
        points['coord_x'] = tf.convert_to_tensor(np.tile(np.expand_dims(points_x,axis=0),[batch,1,1]))
        points['coord_y'] = tf.convert_to_tensor(np.tile(np.expand_dims(points_y,axis=0),[batch,1,1]))
        
        weights = self.volume * np.ones(points['coord_x'].shape) * 0.25
        
        return (points, weights)
    
    def generate_integration_points_and_weights_nonrandom(self):
        '''Generate new integration points and weights in every call'''
        
        points = {}
        weights = {}
        
        x = (0.5-((3-np.sqrt(3))/6)) * np.ones((1, len(self.axis_x),1))
        y = (0.5-((3-np.sqrt(3))/6)) * np.ones((1, len(self.axis_y),1))
        points_x_1D = self.step *  np.concatenate((-x, x), axis=2) + np.tile(np.expand_dims(self.axis_x,axis=1),[1,1,2])
        points_y_1D = self.step *  np.concatenate((-y, y), axis=2) + np.tile(np.expand_dims(self.axis_y,axis=1),[1,1,2])        
        
        mesh_x, mesh_y = np.meshgrid(points_x_1D, points_y_1D)
        self.aa=(mesh_x,mesh_y)
        points_x = np.zeros((self.size_x*self.size_y,4))
        points_y = np.zeros((self.size_x*self.size_y,4))
        elem = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                points_x[elem,:] = np.reshape(mesh_x[2*y:2*y+2,2*x:2*x+2], [-1])
                points_y[elem,:] = np.reshape(mesh_y[2*y:2*y+2,2*x:2*x+2], [-1])
                elem += 1
        
        points['coord_x'] = tf.convert_to_tensor(np.expand_dims(points_x, axis=0))
        points['coord_y'] = tf.convert_to_tensor(np.expand_dims(points_y, axis=0))
        
        weights = self.volume * np.ones(points['coord_x'].shape) * 0.25
        
        return (points, weights)
    
    def generate_random_sampling_of_points(self, num_samples):
        
        random.seed(1234)
        random_sampling = random.sample(range(self.ncells), num_samples)
        
        return random_sampling
    
    
# cells_4 = 50
# grid_4 = Grid(size_x = cells_4, size_y = 50, step_size = 1/cells_4)
# points, weights = grid_4.generate_integration_points_and_weights(10)