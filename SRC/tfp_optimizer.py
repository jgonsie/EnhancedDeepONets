#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:25:55 2024

@author: jesusglezs97
"""

"""TensorFlow interface for TensorFlow Probability optimizers."""

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp

LBFGS_options = {}

def set_LBFGS_options(
    maxcor=100,
    ftol=0,
    gtol=1e-8,
    maxiter=15000,
    maxfun=None,
    maxls=50,
):
    """Sets the hyperparameters of L-BFGS.

    The L-BFGS optimizer used in each backend:

    - TensorFlow 1.x: `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_
    - TensorFlow 2.x: `tfp.optimizer.lbfgs_minimize <https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize>`_
    - PyTorch: `torch.optim.LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_
    - Paddle: `paddle.incubate.optimizers.LBFGS <https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/incubate/optimizer/LBFGS_en.html>`_

    I find empirically that torch.optim.LBFGS and scipy.optimize.minimize are better than
    tfp.optimizer.lbfgs_minimize in terms of the final loss value.

    Args:
        maxcor (int): `maxcor` (scipy), `num_correction_pairs` (tfp), `history_size` (torch), `history_size` (paddle).
            The maximum number of variable metric corrections used to define the limited
            memory matrix. (The limited memory BFGS method does not store the full
            hessian but uses this many terms in an approximation to it.)
        ftol (float): `ftol` (scipy), `f_relative_tolerance` (tfp), `tolerance_change` (torch), `tolerance_change` (paddle).
            The iteration stops when `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`.
        gtol (float): `gtol` (scipy), `tolerance` (tfp), `tolerance_grad` (torch), `tolerance_grad` (paddle).
            The iteration will stop when `max{|proj g_i | i = 1, ..., n} <= gtol` where
            `pg_i` is the i-th component of the projected gradient.
        maxiter (int): `maxiter` (scipy), `max_iterations` (tfp), `max_iter` (torch), `max_iter` (paddle).
            Maximum number of iterations.
        maxfun (int): `maxfun` (scipy), `max_eval` (torch), `max_eval` (paddle).
            Maximum number of function evaluations. If ``None``, `maxiter` * 1.25.
        maxls (int): `maxls` (scipy), `max_line_search_iterations` (tfp), `maxls=0` disables line search and otherwise defaults to 25 (torch).
            Maximum number of line search steps (per iteration).

    Warning:
        If L-BFGS stops earlier than expected, set the default float type to 'float64':

        .. code-block:: python

            dde.config.set_default_float("float64")
    """
    LBFGS_options["maxcor"] = maxcor
    LBFGS_options["ftol"] = ftol
    LBFGS_options["gtol"] = gtol
    LBFGS_options["maxiter"] = maxiter
    LBFGS_options["maxfun"] = maxfun if maxfun is not None else int(maxiter * 1.25)
    LBFGS_options["maxls"] = maxls

set_LBFGS_options(maxcor=100,
                  ftol=0,
                  gtol=1e-8,
                  maxiter=50,
                  maxfun=None,
                  maxls=50)

class LossAndFlatGradient:
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, trainable_variables, build_loss, train_data):
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss
        self.train_data = train_data
        
        # Shapes of all trainable parameters
        self.shapes = tf.shape_n(trainable_variables)
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    # @tf.function(jit_compile=True) has an error.
    @tf.function
    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.

        Args:
           weights_1d: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.build_loss(self.train_data)
        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.

        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.

        Args:
            weights: A list of tf.Tensor representing the weights.

        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)


def lbfgs_minimize(trainable_variables, build_loss, train_data, previous_optimizer_results=None):
    """TensorFlow interface for tfp.optimizer.lbfgs_minimize.

    Args:
        trainable_variables: Trainable variables, also used as the initial position.
        build_loss: A function to build the loss function expression.
        previous_optimizer_results
    """

    func = LossAndFlatGradient(trainable_variables, build_loss, train_data)
    initial_position = None
    if previous_optimizer_results is None:
        initial_position = func.to_flat_weights(trainable_variables)

    results = tfp.optimizer.lbfgs_minimize(
        func,
        initial_position=initial_position,
        previous_optimizer_results=previous_optimizer_results,
        num_correction_pairs=LBFGS_options["maxcor"],
        tolerance=LBFGS_options["gtol"],
        x_tolerance=0,
        f_relative_tolerance=LBFGS_options["ftol"],
        max_iterations=LBFGS_options["maxiter"],
        parallel_iterations=1,
        max_line_search_iterations=LBFGS_options["maxls"],
    )
    # The final optimized parameters are in results.position.
    # Set them back to the variables.
    func.set_flat_weights(results.position)
    return results