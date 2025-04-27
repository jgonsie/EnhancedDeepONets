#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:44:29 2024

@author: jesusglezs97
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_field(data, grid, label, vmin, vmax):
    fig, ax = plt.subplots(1, 1)
    X, Y = np.meshgrid(grid.axis_x, grid.axis_y)
    Z = np.reshape(data, [grid.size_x, grid.size_y])
    im = ax.pcolormesh(grid.axis_x, grid.axis_y, Z, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(label)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
    return fig

def plot_loss(df_history, ymin = None, ymax = None):
    keys = ['loss', 'val_loss']
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=k)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_title('Loss evolution')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_partialLosses_train(df_history):
    keys = list(df_history.keys())
    keys = [i for i in keys if 'loss_' in i and 'val' not in i]
    labels = {'loss_u': 'u', 'loss_gradu_x': 'grad(u)_x', 'loss_gradu_y': 'grad(u)_y',
              'loss_vgradu': 'vgrad(u)', 'loss_mugradu': 'mugrad(u)',
              'loss_gradu_vx': 'partial(u/vx)', 'loss_gradu_vy': 'partial(u/vy)',
              'loss_gradu_mu': 'partial(u/mu)', 'loss_vgradu_x': 'vgrad(u)_x', 
              'loss_vgradu_y': 'vgrad(u)_y', 'loss_mugradu_x': 'mugrad(u)_x',
              'loss_mugradu_y': 'mugrad(u)_y'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_yscale("log")
    ax.set_title('Training partial losses')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_partialLosses_val(df_history):
    keys = list(df_history.keys())
    keys = [i for i in keys if 'loss_' in i and 'val' in i]
    labels = {'val_loss_u': 'u', 'val_loss_gradu_x': 'grad(u)_x', 'val_loss_gradu_y': 'grad(u)_y',
              'val_loss_vgradu': 'vgrad(u)', 'val_loss_mugradu': 'mugrad(u)',
              'val_loss_gradu_vx': 'partial(u/vx)', 'val_loss_gradu_vy': 'partial(u/vy)',
              'val_loss_gradu_mu': 'partial(u/mu)', 'val_loss_vgradu_x': 'vgrad(u)_x',
              'val_loss_vgradu_y': 'vgrad(u)_y', 'val_loss_mugradu_x': 'mugrad(u)_x',
              'val_loss_mugradu_y': 'mugrad(u)_y'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_yscale("log")
    ax.set_title('Validation partial losses')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_L2re_train(df_history, ymin = None, ymax = None):
    keys = ['L2re_u', 'L2re_grad_u']
    labels = {'L2re_u': 'u', 'L2re_grad_u': 'grad(u)'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_ylim(ymin, ymax)
    ax.set_title('L2 rel. error (%) training set')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_L2re_val(df_history, ymin = None, ymax = None):
    keys = ['val_L2re_u', 'val_L2re_grad_u']
    labels = {'L2re_u': 'u', 'L2re_grad_u': 'grad(u)'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_ylim(ymin, ymax)
    ax.set_title('L2 rel. error (%) validation set')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_L2re_u(df_history, ymin = None, ymax = None):
    keys = ['L2re_u', 'val_L2re_u']
    labels = {'L2re_u': 'u training', 'val_L2re_u': 'u validation'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_ylim(ymin, ymax)
    ax.set_title('u L2 rel. error (%) ')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_L2re_gradu(df_history, ymin = None, ymax = None):
    keys = ['L2re_grad_u', 'val_L2re_grad_u']
    labels = {'L2re_grad_u': 'grad(u) training', 'val_L2re_grad_u': 'grad(u) validation'}
    fig, ax = plt.subplots(1, 1)
    for k in keys:
        ax.plot(np.array(df_history[k]), label=labels[k])
    ax.set_ylim(ymin, ymax)
    ax.set_title('grad(u) L2 rel. error (%)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def plot_data_distribution(x_train, y_train, x_val, y_val):
    fig, ax = plt.subplots(1,1)
    # ax.scatter(x_train['DT'], np.ones(x_train['DT'].shape), label='Training points')
    # ax.scatter(x_val['DT'], np.ones(x_val['DT'].shape), label='Validation points')
    ax.scatter(x_train['DT'], np.mean(y_train['T'], axis=(-1,-2)), zorder=2, label='Training points')
    ax.scatter(x_val['DT'], np.mean(y_val['T'], axis=(-1,-2)), zorder=1, label='Validation points')
    # ax.set_yticks([])
    ax.set_xlabel(r"$\hat{\mu}$")
    ax.set_ylabel(r"mean $(u_h)$")
    plt.legend()
    return fig

def plot_log_data_distribution(DTs, E_training, E_conv, E_diff):
    plot = plt.figure(figsize=(8, 5))
    plt.scatter(DTs, E_training, color='blue', s=10, label='Simulations')
    # plt.plot(D_values, metrics, color='blue', linestyle='--', alpha=0.5)
    plt.xscale('log')  # Log scale for D
    plt.xlabel('Diffusion Coefficient $(\mu)$ [$m^2/s$]')
    plt.ylabel('$||u_h||_2$')
    ax = plt.gca()
    xlim0, xlim1 = ax.get_xlim()
    plt.axvspan(xlim0, 0.14, alpha=0.1, color='red', label='Convection-Dominated')
    plt.axvspan(1.41, xlim1*10, alpha=0.1, color='green', label='Diffusion-Dominated')
    secax = ax.secondary_xaxis('top', functions=(lambda x: np.sqrt(2) / (x+1e-8), lambda x: np.sqrt(2) / (x+1e-8)))
    secax.set_xlabel('Pe number [-]')
    plt.axhline(E_conv, color='red', linestyle='--', label='Pure-convection Energy')
    plt.axhline(E_diff, color='green', linestyle='--', label='Pure-diffusion Energy')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(left=xlim0, right=xlim1)
    plt.show()
    return plot

def plot_log_data_distribution2(DTs_tr, E_tr, DTs_val, E_val, E_conv, E_diff):
    plot = plt.figure(figsize=(8, 5))
    plt.scatter(DTs_tr, E_tr, color='red', s=30, marker='D',zorder=2, edgecolors='white', label='Training data')
    plt.scatter(DTs_val, E_val, color='blue', s=10 ,zorder=1, label='Validation data')
    # plt.plot(D_values, metrics, color='blue', linestyle='--', alpha=0.5)
    plt.xscale('log')  # Log scale for D
    plt.xlabel('Diffusion Coefficient $(\mu)$ [$m^2/s$]')
    plt.ylabel('$||u_h||_2^2$')
    ax = plt.gca()
    xlim0, xlim1 = ax.get_xlim()
    plt.axvspan(xlim0, 0.14, alpha=0.1, color='red', label='Convection-Dominated')
    plt.axvspan(1.41, xlim1*10, alpha=0.1, color='green', label='Diffusion-Dominated')
    secax = ax.secondary_xaxis('top', functions=(lambda x: np.sqrt(2) / (x+1e-8), lambda x: np.sqrt(2) / (x+1e-8)))
    secax.set_xlabel('Pe number [-]')
    plt.axhline(E_conv, color='orange', linestyle='--', label='Pure-convection Energy')
    plt.axhline(E_diff, color='green', linestyle='--', label='Pure-diffusion Energy')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.xlim(left=xlim0, right=xlim1)
    plt.show()
    return plot