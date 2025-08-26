import numpy as np

from scipy.optimize import minimize, LinearConstraint
import time

class my_mpc():
    # emit_optimal_input_ = pyqtSignal(bool)
    def __init__(self,model_params=None,
                 predict_length=6,
                 control_length=5,
                 ref=None,
                 control_strategy='Tracking',
                 control_channel=None,
                 is_WC=False,
                 min_input=np.array([0,0,0,0]),
                 max_input=np.array([1,1,1,1]),
                 control_type='closed_loop'):
        super(my_mpc, self).__init__()

        self.hidden_size=model_params['B'].shape[0]
        self.input_size= model_params['B'].shape[1]
        self.output_size=model_params['C'].shape[0]
        self.lasttime = time.time()

        print('input_dim:%d  system_dim:%d'%(self.input_size,self.output_size))
        model_type = 'discrete' # either 'discrete' or 'continuous'
        # self.model = do_mpc.model.Model(model_type)
        self.u = np.zeros(shape=(self.input_size,1))
        self.predict_length = predict_length
        
        if model_params==None:
            # load model parameter from checkpoint
            self.Wrec=np.abs(np.random.randn(self.hidden_size,self.hidden_size))
            self.Wrec_bias=np.abs(np.random.randn(self.hidden_size,1))

            self.B = np.abs(np.random.randn(self.hidden_size,self.input_size))
            self.B_bias= np.abs(np.random.randn(self.hidden_size,1))

            self.C=np.abs(np.random.randn(self.output_size,self.hidden_size))
            self.C_bias=np.abs(np.random.randn(self.output_size,1))
        else:
            self.Wrec=model_params['Wrec']
            self.Wrec_bias=model_params['Wrec_bias']

            self.B = model_params['B']
            self.B_bias= model_params['B_bias']

            self.C=model_params['C']
            self.C_bias=model_params['C_bias']
        
        self.is_WC = is_WC
        self.rnn_type=model_params['rnn_type']
        self.predict_length = predict_length
        self.control_length= control_length
        self.optimal_loss = np.zeros((4,self.predict_length))
        self.max_us = max_input 
        self.min_us = min_input 

        self.control_strategy=control_strategy 
        self.control_channel=control_channel
        self.control_type = control_type
        self.ref = np.reshape(ref,(self.output_size,1))
        self.u0 = np.zeros((self.control_length,4))
        self.start_optimize=False

    def objective(self,u):
        x = self.init_x.copy()
        
        u = u.reshape((self.control_length,-1))
        cost = 0
        
        for t in range(self.predict_length):
            if t<self.control_length:
                if self.rnn_type=='EI-RNN':
                    if self.is_WC:
                        x = np.tanh(self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias)
                    else:
                        x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                elif self.rnn_type=='VAR':
                    x = self.Wrec @ x + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
            else:
                if self.rnn_type=='EI-RNN':
                    if self.is_WC:
                        x = np.tanh(self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias)
                    else:
                        x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                elif self.rnn_type=='VAR':
                    x = self.Wrec @ x + self.Wrec_bias + self.B_bias

            if self.rnn_type=='EI-RNN':
                y = np.log(1+np.exp((self.C @ np.maximum(x,0))+self.C_bias))
            elif self.rnn_type=='VAR':
                y = np.exp(self.C@ x+self.C_bias)

            if self.control_strategy=='Tracking': 
                cost += np.sum((y - self.ref)**2)

            elif self.control_strategy=='Activating':
                cost += -(y[self.control_channel,0]**2)
            elif self.control_strategy=='Inhibiting':
                cost += (y[self.control_channel,0]**2)

            cost += 0.01*np.sum(u**2)
        
        return cost
    
    def constraint(self,u):
        u = u.reshape((self.control_length,-1))
        constraints = []
        for t in range(self.control_length):
            constraints.extend(self.max_us-u[t])
            constraints.extend(u[t]-self.min_us) #-self.min_us)
        return constraints


    def set_initial_state(self,init_x):
        self.init_x=init_x
        self.start_optimize=True

    def run(self):
        while True:
            if self.start_optimize:
                self.start_optimize=False
                self.lasttime= time.time()
                # self._signal.emit(list(self.obtain_optimal_control()))
                self.obtain_optimal_control()
                cost_time = time.time() - self.lasttime
                # print('cost_time:',cost_time*1000)
            time.sleep(0.02)


    def obtain_optimal_control(self,epochs=20):
        import time
        start = time.time()
        
        init_input_U=np.random.uniform(-1,1,(self.control_length, self.input_size))

        self.u0=init_input_U

        constraints = [{'type': 'ineq', 'fun': self.constraint}]
        # print(constraints)
        result = minimize(self.objective, 
                          np.reshape(self.u0,-1), 
                          method='COBYLA',  # 'L-BFGS-B', #'trust-constr', #'Powell',# 'Nelder-Mead',#'COBYLA', #'Powell', # 'SLSQP', #
                          constraints=constraints,#self.constraint(self.u0), 
                          options={'maxiter': 1000})
        self.u0 = result.x.reshape((self.control_length,4)) # 'trust-constr' #

       

        if self.control_type=='closed_loop':
            self.u = self.u0[:1,:].T
        elif self.control_type=='open_loop':
            self.u = self.u0[:,:].T