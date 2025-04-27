# -*- coding: utf-8 -*-

"""
Created on Tue Apr 16 17:13:06 2024

@author: jesusglezs97
"""
import tensorflow as tf
import keras as keras
from SRC.tfp_optimizer import lbfgs_minimize
from SRC.utils import get_model_performance
from SRC.metrics import relativeL2error, myCustomLoss

class DeepONet(keras.Model):
    def __init__(self, layers_branch_v, layers_branch_mu, layers_trunk, 
                 num_rand_sampling = 100, dimension = '1D', seed = 42,
                 dtypeid = tf.float64, **kwargs):
        
        # assert layers_branch_v[-1]==layers_branch_mu[-1], f'The size of the last layer of the branches must be equal: {layers_branch_v[-1]} vs {layers_branch_mu[-1]}'
        # assert layers_branch_v[-1]==layers_trunk[-1], f'The size of the last layer of the branch and trunk must be equal: {layers_branch_v[-1]} vs {layers_trunk[-1]}'
        
        super().__init__()
        
        # Random seeds for being deterministic.
        keras.utils.set_random_seed(1234)
        
        self.num_basis_func = layers_trunk[-1]
        self.fields_br, self.fields_tr = self.fields_by_dimension(dimension)
        self.dim = 1 if dimension == '1D' else 2
        self.dtypeid = dtypeid
        
        # Random seeds for being deterministic.
        init = tf.keras.initializers.GlorotUniform(seed=seed)
        
        # Create branch net for velocity
        # br_v_layers = [keras.layers.Dense(units=layer, activation="tanh", 
        #                                   use_bias=True) for layer in layers_branch_v[:-1]]
        # br_v_layers.append(keras.layers.Dense(units=layers_branch_v[-1], 
        #                                       activation="linear", 
        #                                       use_bias=True))
        # br_v_layers.insert(0, keras.Input(shape=(self.num_rand_sampling,2,)))
        # br_v_layers.insert(1, keras.layers.Flatten())
        # self.branch_v = keras.Sequential(br_v_layers, name='Branch_v')
        
        # Create branch net
        br_layers = [keras.layers.Dense(units=layer, activation="tanh", 
                                            use_bias=True, kernel_initializer=init) for layer in layers_branch_mu]
        self.branch = keras.Sequential(br_layers, name='Branch')
        
        # Create trunk
        tru_layers = [keras.layers.Dense(units=layer, activation="tanh", 
                                         use_bias=True, kernel_initializer=init) for layer in layers_trunk]
        self.trunk = keras.Sequential(layers=tru_layers, name='Trunk')
        
        # Linear layer
        self.linear_layer = keras.layers.Dense(units=1, activation=None, 
                                               use_bias=False, kernel_initializer=init,
                                               name='linear')
        
    def fields_by_dimension(self, dimension):
        '''Selects the involved fields depending on the dimension'''
        
        # fields_br = ['DT', 'v_x', 'v_y']
        fields_br = ['DT']
        fields_tr = ['coord_x', 'coord_y']
        
        if dimension == '1D':
            fields_br = {k for k in fields_br if '_y' not in k}
            fields_tr = {k for k in fields_tr if '_y' not in k}
        elif dimension == '2D':
            fields_br = fields_br
            fields_tr = fields_tr
        else:
            ValueError('Dimension not permitted')
        
        return fields_br, fields_tr
            
    def sort_state(self, state, input_keys):
        ''' Sorts the input data given the order in self.fields_watched and 
        reshapes the data'''
        
        state_watched = tf.concat([state[k] for k in input_keys], axis=-1)
        
        return state_watched
    
    def construct_matrix(self, inputs):
        '''Construct all the matrix obtained previously to the linear layer 
        application'''
        
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
        
        x_br = self.sort_state(inputs, self.fields_br) #(b,1)
        # print('Input shape brmu: ', x_brmu.shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_br)
            basis_br = self.branch(x_br)
            
        dbasis_br = tape.batch_jacobian(basis_br,x_br)
        # print('Output shape basis brmu: ', basis_brmu.shape)
        # print('Output shape dbasis brmu: ', dbasis_brmu.shape)
        del tape

        # y_pred = self.linear_layer(basis_brmu) #Necessary to initialize the layer
        
        x_tr = self.sort_state(inputs, self.fields_tr) #(e,2)
        # print('Input shape tr: ', x_tr.shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_tr)
            basis_tr = self.trunk(x_tr) #(e,j)

        dbasis_tr = tape.batch_jacobian(basis_tr,x_tr) #(e,j,2)

        # print('Output shape basis tr: ', basis_tr.shape)
        # print('Output shape dbasis tr: ', dbasis_tr.shape)
        del tape

        return (basis_br, basis_tr, dbasis_br, dbasis_tr)
    
    
    def call(self, inputs):
        '''Applies the linear layer to both the basis and the dbasis'''
        
        Umatrix_br, Umatrix_tr, dUmatrix_br, dUmatrix_tr = self.construct_matrix(inputs)

        basis = tf.einsum('bj,ej->bej', Umatrix_br, Umatrix_tr)
        # db_vx = tf.einsum('bj,ej->bej', dUmatrix_br[:,:,2], Umatrix_tr)
        # db_vy = tf.einsum('bj,ej->bej', dUmatrix_br[:,:,3], Umatrix_tr)
        db_mu = tf.einsum('bj,ej->bej', dUmatrix_br[:,:,0], Umatrix_tr)
        db_x = tf.einsum('bj,ej->bej', Umatrix_br, dUmatrix_tr[:,:,0])
        db_y = tf.einsum('bj,ej->bej', Umatrix_br, dUmatrix_tr[:,:,1])

        U = self.linear_layer(basis)
        # dU_vx = self.linear_layer(db_vx)
        # dU_vy = self.linear_layer(db_vy)
        dU_mu = self.linear_layer(db_mu)
        dU_x = self.linear_layer(db_x)
        dU_y = self.linear_layer(db_y)
        
        result = {'T': U,
                  'grad(T)_x': dU_x,
                  'grad(T)_y': dU_y,
                  # 'jacUx(T)': dU_vx,
                  # 'jacUy(T)': dU_vy,
                  'jacMu(T)': dU_mu}
        
        return result
    
    
class NeuralOperatorModel(keras.Model):
    
    def __init__(self, net, grid, system_constructor, quadrature = 'centroids',
                 LS = True, regularizer = 10**(-3), GPU = True, **kwargs):
        
        super().__init__()
        self.net = net
        self.grid = grid
        self.quadrature = tf.constant(quadrature, dtype=tf.string)
        self.n_points_quad = grid.n_points_by_method[quadrature]
        self.LS_activation = LS
        self.regularizer = regularizer
        self.GPU = GPU
        self.dtypeid = net.dtypeid
                
        if system_constructor == 'vanilla':
            self.system = self.construct_LS_vanilla
            self.keys_phy = ['loss_u']
            self.keys_der = []
        elif system_constructor == 'H1':
            self.system = self.construct_LS_H1
            self.keys_phy = ['loss_u', 'loss_gradu_x', 'loss_gradu_y']
            self.keys_der = []
        elif system_constructor == 'phy':
            self.system = self.construct_LS_phy
            self.keys_phy = ['loss_u', 'loss_vgradu', 'loss_mugradu']
            self.keys_der = []
        elif system_constructor == 'van+der':
            self.system = self.construct_LS_vander
            self.keys_phy = ['loss_u']
            self.keys_der = ['loss_gradu_mu']
        elif system_constructor == 'H1+der':
            self.system = self.construct_LS_H1der
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
          
        # # Initialize branch and trunk nets
        # self.net.trunk.build(input_shape=(None, self.net.dim))
        # self.net.branch_mu.build(input_shape=(None, 1))
        # # self.net.branch_v.build(input_shape=(None, self.net.num_rand_sampling,self.net.dim))
        # self.net.linear_layer.build(input_shape=(None, self.net.num_basis_func))
              
        # Generate spatial evaluation points
        self.integration_points, _ = self.points_and_weights()
        
        # Initialize loss and metrics
        self.train_loss_tracker = myCustomLoss()
        self.train_partialLosses_tracker = {k:myCustomLoss() for k in self.keys_phy+self.keys_der}
        self.val_loss_tracker = myCustomLoss()
        self.val_partialLosses_tracker = {k:myCustomLoss() for k in self.keys_phy+self.keys_der}
        self.train_L2re_u_tracker = relativeL2error()
        self.train_L2re_grad_u_tracker = relativeL2error()
        self.val_L2re_u_tracker = relativeL2error()
        self.val_L2re_grad_u_tracker = relativeL2error()
        
        self.input_signature = [({'DT': tf.TensorSpec(shape=(None,1), dtype=self.dtypeid),
                                  'v_x': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid),
                                  'v_y': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid)},
                                 {'T': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid),
                                  'grad(T)_x': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid),
                                  'grad(T)_y': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid),
                                  'jacMu(T)': tf.TensorSpec(shape=(None,self.grid.ncells,1), dtype=self.dtypeid)})]
        
        self._train_step_fn = tf.function(
            self._train_step_core,
            input_signature=self.input_signature,
            jit_compile=True)
        
        self._test_step_fn = tf.function(
            self._test_step_core,
            input_signature=self.input_signature,
            jit_compile=True)
        
    def points_and_weights(self):
        '''Generates new integration points and weights'''
        
        # Generate integration points and weights
        outputs = tf.numpy_function(
            func = self.grid.generate_quadrature_tf,
            inp  = [self.quadrature, tf.constant(1), 
                    tf.constant(self.n_points_quad)], # Inputs as TensorFlow tensors
            Tout = [self.dtypeid, self.dtypeid, self.dtypeid]  # Data types of the outputs
            )
        
        points = {'coord_x': tf.squeeze(tf.ensure_shape(outputs[0], (1, self.grid.ncells, self.n_points_quad)), axis=0), 
                  'coord_y': tf.squeeze(tf.ensure_shape(outputs[1], (1, self.grid.ncells, self.n_points_quad)), axis=0)}
        weights = tf.squeeze(tf.ensure_shape(outputs[2], (1, self.grid.ncells, self.n_points_quad)), axis=0)
        
        return points, weights
    
    def construct_LS_vanilla(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the vanilla DeepONets'''

        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
               
        Amatrix = Umatrix #LHS of the LS system
        bvector = y_dict['T'] #RHS of the LS system
  
        return Amatrix, bvector
   
    def construct_LS_H1(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
       
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
        dUmatrix_x = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,0])
        dUmatrix_y = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,1])

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, dUmatrix_x, dUmatrix_y], axis=1)
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], y_dict['grad(T)_x'], y_dict['grad(T)_y']], axis=1)
        
        return Amatrix, Bvector

    
    def construct_LS_phy(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
        dUmatrix_x = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,0])
        dUmatrix_y = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y
        mudUmatrix_x = tf.einsum('bp,bej->bej', x_dict['DT'], dUmatrix_x)
        mudUmatrix_y = tf.einsum('bp,bej->bej', x_dict['DT'], dUmatrix_y)
        mudUmatrix = mudUmatrix_x + mudUmatrix_y

        # LHS of the LS system
        Amatrix = tf.concat([Umatrix, vdUmatrix, mudUmatrix], axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bp,bep->bep', x_dict['DT'], y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bp,bep->bep', x_dict['DT'], y_dict['grad(T)_y'])
        mudUmatrix_fv = mudUmatrix_x_fv + mudUmatrix_y_fv
        
        # RHS of the LS system
        Bvector = tf.concat([y_dict['T'], vdUmatrix_fv, mudUmatrix_fv], axis=1)
        
        return Amatrix, Bvector

   
    def construct_LS_vander(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
        dUmatrix_mu = tf.einsum('bj,ej->bej', dcoeffs_br[:,:,0], basis_tr) 
                
        # LHS of the LS system
        Alist = [Umatrix, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # RHS of the LS system
        Blist = [y_dict['T'], y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_H1der(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
        dUmatrix_mu = tf.einsum('bj,ej->bej', dcoeffs_br[:,:,0], basis_tr)
        dUmatrix_x = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,0])
        dUmatrix_y = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,1])
        
        # LHS of the LS system
        Alist = [Umatrix, dUmatrix_x, dUmatrix_y, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # RHS of the LS system
        Blist = [y_dict['T'], y_dict['grad(T)_x'], y_dict['grad(T)_y'], y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
    def construct_LS_phyder(self, x_dict, y_dict):
        '''Constructs the Least-Squares system A x = b for the enhanced DeepONets'''
    
        #Contruct the matrix for LS system
        coeffs_br, basis_tr, dcoeffs_br, dbasis_tr = self.net.construct_matrix(x_dict) 
        
        Umatrix = tf.einsum('bj,ej->bej', coeffs_br, basis_tr)
        dUmatrix_mu = tf.einsum('bj,ej->bej', dcoeffs_br[:,:,0], basis_tr)
        dUmatrix_x = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,0])
        dUmatrix_y = tf.einsum('bj,ej->bej', coeffs_br, dbasis_tr[:,:,1])
        
        # Build the terms for the loss
        vdUmatrix_x = tf.einsum('bep,bej->bej', x_dict['v_x'], dUmatrix_x)
        vdUmatrix_y = tf.einsum('bep,bej->bej', x_dict['v_y'], dUmatrix_y)
        vdUmatrix = vdUmatrix_x + vdUmatrix_y #(b,e,j)
        mudUmatrix_x = tf.einsum('bp,bej->bej', x_dict['DT'], dUmatrix_x)
        mudUmatrix_y = tf.einsum('bp,bej->bej', x_dict['DT'], dUmatrix_y)
        mudUmatrix = mudUmatrix_x + mudUmatrix_y #(b,e,j)
        
        # LHS of the LS system
        Alist = [Umatrix, vdUmatrix, mudUmatrix, dUmatrix_mu]
        Amatrix = tf.concat(Alist, axis=1)
        
        # Build data terms of the loss function
        vdUmatrix_x_fv = tf.einsum('bep,bep->bep', x_dict['v_x'], y_dict['grad(T)_x'])
        vdUmatrix_y_fv = tf.einsum('bep,bep->bep', x_dict['v_y'], y_dict['grad(T)_y'])
        vdUmatrix_fv = vdUmatrix_x_fv + vdUmatrix_y_fv
        mudUmatrix_x_fv = tf.einsum('bp,bep->bep', x_dict['DT'], y_dict['grad(T)_x'])
        mudUmatrix_y_fv = tf.einsum('bp,bep->bep', x_dict['DT'], y_dict['grad(T)_y'])
        mudUmatrix_fv = mudUmatrix_x_fv + mudUmatrix_y_fv
        
        # RHS of the LS system
        Blist = [y_dict['T'], vdUmatrix_fv, mudUmatrix_fv, y_dict['jacMu(T)']]
        Bvector = tf.concat(Blist, axis=1)
        
        return Amatrix, Bvector
    
            
    def resolve_LS(self, x_dict, y_dict):
        '''Resolves the Least-Squares system and updates the weights of the 
        linear layer'''

        # Construct LS system
        A, b = self.system(x_dict, y_dict) #(b,c*e,j) (b,c*e,1)

        # Solve LS system (op1)
        A_flat = tf.reshape(A, [-1,A.shape.as_list()[-1]])
        b_flat = tf.reshape(b, [-1,b.shape.as_list()[-1]])
        alpha_new = tf.linalg.lstsq(A_flat, b_flat, l2_regularizer=self.regularizer)
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
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **self.points_and_weights}
        
        # Run the model
        A, b = self.system(x_curated, y_dict)
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
        # self.kappas = {'loss_u': tf.convert_to_tensor(([1.]), dtype=self.dtype), 
        #                'loss_vgradu': tf.convert_to_tensor(([1.]), dtype=self.dtype),
        #                'loss_mugradu': tf.convert_to_tensor(([1.]), dtype=self.dtype),
        #                'loss_gradu_vx': tf.convert_to_tensor(([1.]), dtype=self.dtype),
        #                'loss_gradu_vy': tf.convert_to_tensor(([1.]), dtype=self.dtype), 
        #                'loss_gradu_mu': tf.convert_to_tensor(([1.]), dtype=self.dtype),
        #                }
        self.kappas = {k:tf.convert_to_tensor(([1.]), dtype=self.dtypeid) for k in meanLpartial.keys()}
        
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
        
    def train_step(self, data):
        return self._train_step_fn(data)

    def _train_step_core(self, data):
        ''' Training loop'''

        x_dict, y_dict = data 
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **self.integration_points}
        
        if self.LS_activation == True:
            # Assign training values
            if self.GPU == True:
                trainable_vars = [weight.value for weight in self.net.trainable_weights[:-1]] #GPU
            else:
                trainable_vars = self.net.trainable_weights[:-1] #CPU  
        else:
            # Assign training values
            if self.GPU == True:
                trainable_vars = [weight.value for weight in self.net.trainable_weights] #GPU
            else:
                trainable_vars = self.net.trainable_weights #CPU
        
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainable_vars)
            
            A, b = self.system(x_curated, y_dict)
            unweighted_loss = tf.square(self.net.linear_layer(A)-b)
            loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        grad = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grad, trainable_vars))

        if self.LS_activation == True:
            # Construct and resolve LS and update weights of linear layer
            self.resolve_LS(x_curated, y_dict)
            A, b = self.system(x_curated, y_dict)
            unweighted_loss = tf.square(self.net.linear_layer(A)-b)
            loss, partialLosses = self.weighted_loss(unweighted_loss)
            
        result = self.call(x_dict) 
        
        # Update loss and metrics
        self.train_loss_tracker.update_state(loss)
        for k,v in self.train_partialLosses_tracker.items():
            v.update_state(partialLosses[k])
        self.train_L2re_u_tracker.update_state(y_dict['T'], result['T'])
        self.train_L2re_grad_u_tracker.update_state(
            tf.sqrt(tf.square(y_dict['grad(T)_x']) + tf.square(y_dict['grad(T)_y'])),
            tf.sqrt(tf.square(result['grad(T)_x']) + tf.square(result['grad(T)_y'])))

        return {'loss': self.train_loss_tracker.result(),
                **{k:v.result() for k,v in self.train_partialLosses_tracker.items()},
                'L2re_u': self.train_L2re_u_tracker.result(),
                'L2re_grad_u': self.train_L2re_grad_u_tracker.result()}
    
    def test_step(self, data):
        return self._test_step_fn(data)
    
    def _test_step_core(self, data):
        ''' Validation loop'''
        
        x_dict, y_dict = data 
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **self.integration_points}
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        result = self.call(x_dict)    

        # Update loss and metrics
        self.val_loss_tracker.update_state(loss)
        for k,v in self.val_partialLosses_tracker.items():
            v.update_state(partialLosses[k])
        self.val_L2re_u_tracker.update_state(y_dict['T'], result['T'])
        self.val_L2re_grad_u_tracker.update_state(
            tf.sqrt(tf.square(y_dict['grad(T)_x']) + tf.square(y_dict['grad(T)_y'])),
            tf.sqrt(tf.square(result['grad(T)_x']) + tf.square(result['grad(T)_y'])))


        return {'loss': self.val_loss_tracker.result(),
                **{k:v.result() for k,v in self.val_partialLosses_tracker.items()},
                'L2re_u': self.val_L2re_u_tracker.result(),
                'L2re_grad_u': self.val_L2re_grad_u_tracker.result()}
   
    def reset_metrics(self):
        '''Resets the tracker for the loss and metrics'''
        self.train_loss_tracker.reset_state()
        {k:v.reset_state() for k,v in self.train_partialLosses_tracker.items()}
        self.val_loss_tracker.reset_state()
        {k:v.reset_state() for k,v in self.val_partialLosses_tracker.items()}
        self.train_L2re_u_tracker.reset_state()
        self.train_L2re_grad_u_tracker.reset_state()
        self.val_L2re_u_tracker.reset_state()
        self.val_L2re_grad_u_tracker.reset_state()
          
    def evaluate_loss(self, data):
        ''' Calls the model and return the loss'''
        x_dict, y_dict = data 
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **self.integration_points}
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        return loss
    
    def train_step_LSBFGS(self, data):
        
        x_dict, y_dict = data 
        
        # Prepare data for DeepONet
        x_curated = {**x_dict, **self.integration_points}
        
        # Solve LS and update lineal layer
        self.resolve_LS(x_curated, y_dict)
        
        A, b = self.system(x_curated, y_dict)
        unweighted_loss = tf.square(self.net.linear_layer(A)-b)
        loss, partialLosses = self.weighted_loss(unweighted_loss)
        
        return loss
    
    def train_LBFGS(self, data_training, data_validation, epochs_LBFGS):
        
        print('~~~~~BEGINNING OF LBFGS OPTIMIZATION~~~~~')
        metrics_LBFGS = []
        
        if self.LS_activation == True:
            # Assign training values
            if self.GPU == True:
                trainable_vars = [weight.value for weight in self.net.trainable_weights[:-1]] #GPU
            else:
                trainable_vars = self.net.trainable_weights[:-1] #CPU  
        else:
            # Assign training values
            if self.GPU == True:
                trainable_vars = [weight.value for weight in self.net.trainable_weights] #GPU
            else:
                trainable_vars = self.net.trainable_weights #CPU
        
        for i in range(epochs_LBFGS):
            # Traininig lineal layer with LS and the rest of the weights with LBFGS
            _ = lbfgs_minimize(trainable_vars, self.evaluate_loss, data_training)
            if self.LS_activation == True:
                
                # Prepare data for DeepONet
                x_curated = {**data_training, **self.integration_points}
                # Construct and resolve LS and update weights of linear layer
                self.resolve_LS(x_curated, data_training[1])
                
            metrics_LBFGS.append(get_model_performance(self.test_step, data_training, data_validation, print_results=True))

        LBFGS_history = {k: [d[k].numpy() for d in metrics_LBFGS] for k in metrics_LBFGS[0].keys()}

        return LBFGS_history

    def call(self, data):
        '''Prepares the data and calls to the DeepONet'''
        
        # Prepare data for DeepONet
        data_curated = {**data, **self.integration_points}
        
        # Call DeepONet and linear layer
        net_output = self.net(data_curated) 
        
        result = {'T': net_output['T'],
                   'grad(T)_x': net_output['grad(T)_x'],
                   'grad(T)_y': net_output['grad(T)_y'],
                   # 'jacUx(T)': net_output['jacUx(T)'],
                   # 'jacUy(T)': net_output['jacUy(T)'],
                   'jacMu(T)': net_output['jacMu(T)']}
        
        return result