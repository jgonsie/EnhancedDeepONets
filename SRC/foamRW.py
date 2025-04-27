#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:43:55 2024

@author: jesusglezs97
"""

import numpy as np
import openfoamparser as Ofpp
import os
from joblib import Parallel, delayed

def upload_case(case_directory, time, fieldNames, case):
    '''Reads the data from OpenFOAM and loads it in numpy in vector form'''

    result = {}
    ncells = case.num_cell+1
    # ncells_b = {}
    # for k in case.boundary.keys():
    #     ncells_b[k] = len(list(case.boundary_cells(k)))

    def upload_time(time, field):
        def prepare_data(reading, ncells):
            if type(reading).__module__ != np.__name__: # reading is not a np.array (velocity) of ncells
                result_t = (reading * np.ones((ncells)))
            elif reading.shape[0] != ncells: # reading is a np.array of float/int
                result_t = (reading * np.ones((ncells, len(reading))))
            else: # reading is a np.array of ncells
                result_t = (reading)
            return result_t
        
        route = os.path.join(case_directory, str(time), field)
        if field == 'grad(T)':
            upload = Ofpp.parse_internal_field(route)
            result_centers = prepare_data(upload, ncells)
        else:
            upload = Ofpp.parse_field_all(route)
            result_centers = prepare_data(upload[0], ncells)

        result_boundaries = {}
        # for k,v in upload[1].items():
        #     if len(list(v.keys())) >= 1 and b'value' in list(v.keys()):
        #         result_boundaries[k] = prepare_data(v[b'value'], ncells_b[k])

        return  (result_centers, result_boundaries)
   
    for field in fieldNames:
        # print(case_directory, time, field)
        temp = upload_time(time, field)    
        center_values = temp[0]
        # if sorted(temp[0][1].keys()) != sorted(temp[-1][1].keys()):
        #     raise ValueError (f'foamRW: Not same boundaries for all timesteps in field: {field}.\nRemember to substitute $internalField by the corresponding value in the OpenFOAM field files!')
        # boundary_values = {k:[dic[k] for _, dic in temp] for k in temp[0][1].keys()}

        if field == 'U':
            result['v_x'] = np.array([i[0] for i in center_values])
            result['v_y'] = np.array([i[1] for i in center_values])
            # result['x_velocity_b']  = {}
            # result['y_velocity_b']  = {}
            # for boundary in ncells_b.keys():
            #     if boundary in boundary_values.keys():
            #         result['x_velocity_b'][boundary.decode('utf-8')] = tf.convert_to_tensor([i[:,0] for i in boundary_values[boundary]], dtype=dtype)
            #         result['y_velocity_b'][boundary.decode('utf-8')] = tf.convert_to_tensor([i[:,1] for i in boundary_values[boundary]], dtype=dtype)
        elif 'grad' in field:
            result[field+'_x'] = np.array([i[0] for i in center_values])
            result[field+'_y'] = np.array([i[1] for i in center_values])
        else:
            result[field] = np.array(center_values)
            # result[field+'_b']  = {}
            # for boundary in ncells_b.keys():
            #     if boundary in boundary_values.keys():
            #         result[field+'_b'][boundary.decode('utf-8')] = tf.convert_to_tensor(boundary_values[boundary], dtype=dtype)

    return result

def upload_jacobian(root_directory):
    f = open(root_directory, "r")
    columns = []  
    for x in f:
      columns.append(np.array(x.split(' ')[:-1], dtype=float))
    jac = np.stack(columns)
    f.close()
    
    return jac

def upload_diagonal(root_directory):
    f = open(root_directory+'/gradMu(T)', "r")
    columns = []  
    for x in f:
      columns.append(float(x))
    gradMu = np.stack(columns)
    f.close()
    
    return gradMu

def upload_vector(root_directory):
    f = open(root_directory, "r")
    columns = []  
    for x in f:
      columns.append(float(x))
    gradMu = np.stack(columns)
    f.close()
    
    return gradMu

def upload_training_data(root_directory, time=0.1, diagonal=True, jacobian=False, dtype='float64'):
    
    case_directories = np.sort(os.listdir(root_directory))
    field_names = os.listdir(root_directory+case_directories[0]+'/0')
    foam_case = Ofpp.FoamMesh(root_directory+case_directories[0])

    data = Parallel(n_jobs=-1)(delayed(upload_case)(root_directory+case_dir, time, field_names, foam_case) for case_dir in case_directories) 
    # data = [upload_case(root_directory+case_dir, time, field_names, foam_case) for case_dir in case_directories]
    
    result = {}
    
    for field in data[0].keys():
        result[field] = np.stack([i[field] for i in data], dtype=dtype)
        
    if jacobian == True:
        data_jac_mu = Parallel(n_jobs=-1)(delayed(upload_vector)(root_directory+case_dir+'/jacMu(T)') for case_dir in case_directories) 
        result['jacMu(T)'] = np.stack(data_jac_mu, dtype=dtype)
        # data_jac_mu = Parallel(n_jobs=-1)(delayed(upload_jacobian)(root_directory+case_dir+'/jacUx(T)') for case_dir in case_directories) 
        # result['jacUx(T)'] = np.stack(data_jac_mu)
        # data_jac_mu = Parallel(n_jobs=-1)(delayed(upload_jacobian)(root_directory+case_dir+'/jacUy(T)') for case_dir in case_directories) 
        # result['jacUy(T)'] = np.stack(data_jac_mu)
        
    # if diagonal == True:
    #     data_diag = Parallel(n_jobs=-1)(delayed(upload_diagonal)(root_directory+case_dir) for case_dir in case_directories) 
    #     result['gradMu(T)'] = np.stack(data_diag)
    
    if 'DT' not in field_names:
        dts = [float(dirs.split('-')[0].split('_')[1]) for dirs in case_directories]
        result['DT'] = np.stack(dts, axis=0, dtype=dtype)
        
    return result

def upload_single_data(root_directory, time=0.1, diagonal=True, jacobian=False):
    
    field_names = os.listdir(root_directory+'/0')
    foam_case = Ofpp.FoamMesh(root_directory)

    data = upload_case(root_directory, time, field_names, foam_case)
    
    result = {}
    for k,v in data.items():
        result[k] = np.expand_dims(v, axis=0)
    
    if jacobian == True:
        data_jac_mu = upload_vector(root_directory+'/jacMu(T)') 
        result['jacMu(T)'] = np.expand_dims(data_jac_mu, axis=0)
    
    # if 'DT' not in field_names:
    #     dts = [float(dirs.split('-')[0].split('_')[1]) for dirs in case_directories]
    #     result['DT'] = np.stack(dts, axis=0)
        
    return result

# data_route = '../OpenFOAM/tests/pure_convection'
# training_data = upload_single_data(data_route, jacobian=True)

# data = upload_training_data('../OpenFOAM/pureConvection1D/training_dataSS/')
# data = upload_training_data('../OpenFOAM/convectionDiffusion2D/training_dataSS/')
# data = upload_training_data('../OpenFOAM/convectionDiffusion2D_10x10_mu_v2/training_data/', jacobian=True)