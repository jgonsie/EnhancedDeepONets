#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:38:47 2025

@author: jgonzalez
"""

import tensorflow as tf
import keras

class relativeL2error(keras.metrics.Metric):
    def __init__(self, name="relative_L2_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num = self.add_weight(name='num', initializer='zeros')
        self.count_total = self.add_weight(name='counter', initializer='zeros')
        
    def update_state(self, y_true, y_pred):
        num = tf.reduce_sum(tf.square(y_true-y_pred), axis=[-1,-2])
        den = tf.reduce_sum(tf.square(y_true), axis=[-1,-2])
        self.num.assign_add(tf.reduce_mean(tf.sqrt(num / (den + 1e-12))))
        
        self.count_total.assign_add(1)
        
    def result(self):
        return self.num / self.count_total * 100
    
    def reset_states(self):
        self.num.assign(0.0)
        self.count_total.assign(0.0)

class relativeL1error(keras.metrics.Metric):
    def __init__(self, name="relative_L1_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num = self.add_weight(name='num', initializer='zeros')
        self.den = self.add_weight(name='den', initializer='zeros')
        self.count_total = self.add_weight(name='counter', initializer='zeros')
    
    def update_state(self, y_true, y_pred):
        num = tf.reduce_sum(y_true-y_pred)
        den = tf.reduce_sum(y_true)
        self.num.assign_add(num)
        self.den.assign_add(den)
        
        count = tf.size(y_true)
        self.count_total.assign_add(count)

    def result(self):
        return (self.num / (self.den + 1e-12)) / self.count_total * 100
    
    def reset_states(self):
        self.num.assign(0.0)
        self.den.assign(0.0)
        self.count_total.assign(0.0)

        
class myCustomLoss(keras.metrics.Metric):
    def __init__(self, name="customLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.error_tracker = self.add_weight(name='customLoss', initializer='zeros')
        self.count_total = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, loss):
        self.error_tracker.assign_add(loss)
        self.count_total.assign_add(1.0)
        
    def result(self):
        return self.error_tracker / self.count_total
    
    def reset_states(self):
        self.error_tracker.assign(0.0)
        self.count_total.assign(0.0)
        

class myCustomLoss2(tf.keras.metrics.Metric):
    def __init__(self, name="myCustomLoss2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_weighted_mse = self.add_weight(name="total_weighted_mse", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros")

    def update_state(self, loss, y_true):
        # Assume y_true and y_pred are scalar (mean MSE for the batch)
        batch_size = tf.cast(tf.size(y_true), y_true.dtype)

        # Accumulate weighted MSE
        self.total_weighted_mse.assign_add(loss * batch_size)
        self.total_samples.assign_add(batch_size)

    def result(self):
        return self.total_weighted_mse / self.total_samples

    def reset_states(self):
        self.total_weighted_mse.assign(0.0)
        self.total_samples.assign(0.0)