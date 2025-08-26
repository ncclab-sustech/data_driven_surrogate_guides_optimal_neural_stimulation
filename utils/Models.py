import torch
from torch.nn import functional as F
from torch import nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.model_utils import EI_Sparse_Mask_
import numpy as np
import math

class EI_RNN(nn.Module):
    def __init__(self, 
                 input_size,    # dimension of external input 
                 hidden_size,   # hidden dimension
                 output_size,   # observe dimension
                 fr_type=False, # transform neural state to neural firing rate by r(t)=relu(x(t))
                 ei_ratio=4,  # the ratio between the size of excitatory neurons and inhibitory neurons.
                 sparsity=0,
                 e_clusters=2,
                 i_clusters=1,
                 inter_or_intra='Inter',
                 output_layers=2,
                 with_Tanh=True, 
                 with_bias=True,
                 with_wrec_perturbation=False,
                 with_b_perturbation=False,
                 device='cuda:0'):
        super(EI_RNN, self).__init__()

        self.fr_type=fr_type
        self.multiplier = None

        self.hidden_size = hidden_size
        

        
        self.device=device
        probability=1.0*ei_ratio/(ei_ratio+1)
        

        self.W_rec_fix=EI_Sparse_Mask_(hidden_dim=hidden_size,e_clusters=e_clusters,i_clusters=i_clusters,sparsity=sparsity,ei_ratio=4,module_type=inter_or_intra).to(torch.device(device))
        
        ones_tensor=torch.ones(int(hidden_size*probability),device=device)
        minus_ones_tensor=-1*torch.ones(hidden_size-int(hidden_size*probability),device=device)
        self.ei_constraints=torch.diag(torch.cat((ones_tensor,minus_ones_tensor)))

        self.w_rec = nn.Parameter(torch.empty((hidden_size,hidden_size)),requires_grad=True)
        self.bias_rec = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)

        nn.init.kaiming_uniform_(self.w_rec, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_rec)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_rec, -bound, bound)
        
        self.w_rec_diag0=torch.ones((hidden_size,hidden_size),device=device)
        
        self.output_layers=output_layers

        self.B = nn.Parameter(torch.empty((hidden_size,input_size)),requires_grad=True)
        self.B_bias = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.B)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.B_bias, -bound, bound)


        self.C = nn.Parameter(torch.empty((output_size,hidden_size)),requires_grad=True)
        self.C_bias = nn.Parameter(torch.empty((output_size,1)),requires_grad=True)
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.C)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.C_bias, -bound, bound)
            
        # others
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus=nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        self.scaling_matrix = 1.0*torch.ones((self.hidden_size,self.hidden_size), device=self.device)
        self.with_wrec_perturbation = with_wrec_perturbation
        self.with_b_perturbation = with_b_perturbation
        self.with_Tanh=with_Tanh
        self.with_bias=with_bias

    def set_w_rec_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_wee = noise_.to(self.device)

    def set_b_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_b = noise_.to(self.device)
        
    def get_w_rec_fix(self):
        return self.W_rec_fix.detach().cpu().numpy()
        
    def set_w_rec_fix(self,w_rec_fix):
        self.W_rec_fix=torch.from_numpy(w_rec_fix).to(torch.device(self.device)).type(torch.float32)
        
    def get_w_rec(self):
        if self.with_wrec_perturbation:
            new_A = torch.multiply(self.scaling_matrix,torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec+self.noise_wee),self.ei_constraints))))
            return new_A
        else:
            return torch.multiply(self.scaling_matrix,torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))))
    
    def set_w_rec_scaling(self, e_magnification=1, i_magnification=1, ei_magnification=1, ie_magnification=1,):
        self.scaling_matrix[:int(self.hidden_size * self.probability), :int(self.hidden_size * self.probability)] = e_magnification
        self.scaling_matrix[int(self.hidden_size * self.probability):,int(self.hidden_size * self.probability):] = i_magnification
        self.scaling_matrix[int(self.hidden_size * self.probability):,:int(self.hidden_size * self.probability)] = ei_magnification
        self.scaling_matrix[:int(self.hidden_size * self.probability),int(self.hidden_size * self.probability):] = ie_magnification


    def get_B(self):
        if self.with_b_perturbation:
            return self.B+self.noise_b
        else:
            return self.B

    def forward(self, inputs, hidden):
        hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
        inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

        u_x = torch.matmul(self.get_B(),inputs_)+self.B_bias

        if self.fr_type:
            h2h= torch.matmul(self.get_w_rec(),self.relu(hidden_))+self.bias_rec
        else:
            h2h = torch.matmul(self.get_w_rec(),hidden_)+self.bias_rec
        
        if self.with_Tanh:
            hidden_new = self.tanh(h2h + u_x)
        else:
            hidden_new = h2h + u_x

        output = self.softplus(torch.matmul(self.C,self.relu(hidden_new))+self.C_bias)

        return torch.permute(output,(0,2,1)), torch.permute(hidden_new,(0,2,1))

class VAR_model(nn.Module):
    def __init__(self, 
                 A,B,C,train_model=False,
                 with_wrec_perturbation=False,
                 with_b_perturbation=False,
                 device='cuda:0'):
        super(VAR_model, self).__init__()
        self.with_wrec_perturbation = with_wrec_perturbation
        self.with_b_perturbation = with_b_perturbation
        self.device=device
        if train_model:
            self.A=nn.Parameter(torch.normal(0,0.01,A.shape),requires_grad=True)
            self.B=nn.Parameter(torch.normal(0,0.01,B.shape),requires_grad=True)
            self.C=nn.Parameter(torch.normal(0,0.01,C.shape),requires_grad=True)
            # self.C=torch.eye(C.shape[0]).to(self.device)
        else:
            self.A=nn.Parameter(A,requires_grad=False)
            self.B=nn.Parameter(B,requires_grad=False)
            self.C=nn.Parameter(C,requires_grad=False)

    def set_w_rec_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_wee = noise_.to(self.device)
    def set_b_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_b = noise_.to(self.device)
        
    def get_B(self):
        if self.with_b_perturbation:
            return self.B+self.noise_b
        else:
            return self.B
        
    def get_A(self):
        if self.with_wrec_perturbation:
            new_A = self.A+self.noise_wee
            L,V=torch.linalg.eig(new_A)
            if torch.max(torch.abs(L))>=1:
                return new_A/(torch.max(torch.abs(L)))#-1e-50)
            else:
                return new_A
            # return new_A
        else:
            return self.A
        
    def forward(self,inputs, hidden):
        hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
        inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

        hidden = self.get_A()@hidden_+self.get_B()@inputs_

        output = torch.log(1+torch.exp(self.C@hidden))
        return torch.permute(output,(0,2,1)), torch.permute(hidden,(0,2,1))

class VAR_model_ori(nn.Module):
    def __init__(self, 
                 A,B,C,train_model=False,
                 with_wrec_perturbation=False,
                 with_b_perturbation=False,
                 device='cuda:0'):
        super(VAR_model_ori, self).__init__()
        self.device=device
        self.with_wrec_perturbation = with_wrec_perturbation
        self.with_b_perturbation = with_b_perturbation
        # self.C=torch.eye(C.shape[0]).to(self.device)
        if train_model:
            self.A=nn.Parameter(torch.normal(0,0.01,A.shape),requires_grad=True)
            self.B=nn.Parameter(torch.normal(0,0.01,B.shape),requires_grad=True)
            self.C=nn.Parameter(torch.normal(0,0.01,C.shape),requires_grad=True)
            # self.C=torch.eye(C.shape[0]).to(self.device)
        else:
            self.A=nn.Parameter(A,requires_grad=False)
            self.B=nn.Parameter(B,requires_grad=False)
            self.C=nn.Parameter(C,requires_grad=True)
    
    def set_w_rec_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_wee = noise_.to(self.device)
    def set_b_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_b = noise_.to(self.device)
        
    def get_B(self):
        if self.with_b_perturbation:
            return self.B+self.noise_b
        else:
            return self.B
        
    def get_A(self):
        if self.with_wrec_perturbation:
            new_A = self.A+self.noise_wee
            L,V=torch.linalg.eig(new_A)
            if torch.max(torch.abs(L))>=1:
                return new_A/(torch.max(torch.abs(L))+1e-2)
            else:
                return new_A
        else:
            return self.A
        
    def forward(self,inputs, hidden):
        hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
        inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

        hidden = self.get_A()@hidden_+self.get_B()@inputs_

    # def forward(self,inputs, hidden):
    #     hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
    #     inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

    #     hidden = self.A@hidden_+self.B@inputs_

        output = torch.exp(self.C@hidden)
        return torch.permute(output,(0,2,1)), torch.permute(hidden,(0,2,1)) 

class nVAR_model(nn.Module):
    def __init__(self, 
                 A,B,C,train_model=False,
                 with_wrec_perturbation=False,
                 with_b_perturbation=False,
                 device='cuda:0'):
        super(nVAR_model, self).__init__()
        self.device=device
        self.with_wrec_perturbation = with_wrec_perturbation
        self.with_b_perturbation = with_b_perturbation

        if train_model:
            self.A=nn.Parameter(torch.normal(0,0.01,A.shape),requires_grad=True)
            self.B=nn.Parameter(torch.normal(0,0.01,B.shape),requires_grad=True)
            self.C=nn.Parameter(torch.normal(0,0.01,C.shape),requires_grad=True)
            # self.C=torch.eye(C.shape[0]).to(self.device)
        else:
            self.A=nn.Parameter(A,requires_grad=True)
            self.B=nn.Parameter(B,requires_grad=True)
            self.C=nn.Parameter(C,requires_grad=True)
    
    def set_w_rec_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_wee = noise_.to(self.device)
    def set_b_perturbation(self,noise_): # torch.normal(0, 0.01, size=self.w_rec.shape).to(self.device)
        self.noise_b = noise_.to(self.device)
        
    def get_B(self):
        if self.with_b_perturbation:
            return self.B+self.noise_b
        else:
            return self.B
        
    def get_A(self):
        if self.with_wrec_perturbation:
            new_A = self.A+self.noise_wee
            # L,V=torch.linalg.eig(new_A)
            # if torch.max(torch.abs(L))>=1:
            #     return new_A/(torch.max(torch.abs(L))+1e-2)
            # else:
            #     return new_A
            return new_A
        else:
            return self.A
        
    def forward(self,inputs, hidden):
        hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
        inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

        hidden = torch.tanh(self.get_A()@hidden_+self.get_B()@inputs_)
        output = torch.log(1+torch.exp(self.C@hidden))
        return torch.permute(output,(0,2,1)), torch.permute(hidden,(0,2,1))

class Wilson_Cowan(nn.Module):
    def __init__(self, 
                 W_EE,W_EI,W_IE,W_II,B,C,train_model=False,dt=0.05,
                 with_wrec_perturbation=False,
                 with_b_perturbation=False,
                 device='cuda:0'):
        super(Wilson_Cowan, self).__init__()
        if train_model:
            self.W_EE=nn.Parameter(torch.normal(0,0.01,W_EE.shape),requires_grad=True)
            self.W_EI=nn.Parameter(torch.normal(0,0.01,W_EI.shape),requires_grad=True)
            self.W_IE=nn.Parameter(torch.normal(0,0.01,W_IE.shape),requires_grad=True)
            self.W_II=nn.Parameter(torch.normal(0,0.01,W_II.shape),requires_grad=True)
            self.B=nn.Parameter(torch.normal(0,0.01,B.shape),requires_grad=True)
            self.C=nn.Parameter(torch.normal(0,0.01,C.shape),requires_grad=True)
        else:
            self.W_EE=torch.from_numpy(W_EE).to(device)
            self.W_EI=torch.from_numpy(W_EI).to(device)
            self.W_IE=torch.from_numpy(W_IE).to(device)
            self.W_II=torch.from_numpy(W_II).to(device)
            self.B=torch.from_numpy(B).to(device)
            self.C=torch.from_numpy(C).to(device)
        self.num_regions=W_EE.shape[0]
        self.dt=torch.tensor(dt)
        self.r=torch.tensor(1)
        self.E_tau=torch.tensor(1)
        self.E_a=torch.tensor(1.2)
        self.E_theta=torch.tensor(0.1)
        self.I_tau=torch.tensor(1)
        self.I_a=torch.tensor(1)
        self.I_theta=torch.tensor(0.1)
        self.softplus = torch.nn.Softplus()
    def sigmoid(self,x, a, theta):
        return 1 / (1 + torch.exp(-a * (x - theta))) - 1 / (1 + torch.exp(a * theta))
    def forward(self,inputs, hidden):
        hidden_ = torch.permute(hidden,(0,2,1)) # batch*hidden*time
        inputs_ = torch.permute(inputs,(0,2,1)) # batch*input_size*time

        E = hidden_[:,:self.num_regions,:]  # activity of excitatory neurons
        I = hidden_[:,self.num_regions:,:]  # activity of inhibitory neurons
        
       
        input_E = self.W_EE @ E - self.W_EI @ I + self.B@inputs_
        E = E+self.dt*(-E + (1 - self.r * E) * self.sigmoid(input_E, self.E_a, self.E_theta)) / self.E_tau

        input_I =self.W_IE @ E - self.W_II @ I
        I =I+self.dt* (-I + (1 - self.r * I) * self.sigmoid(input_I, self.I_a, self.I_theta)) / self.I_tau


        output = 3*self.softplus(self.C@E)
        hidden_[:,:self.num_regions,:] = E
        hidden_[:,self.num_regions:,:] = I
        # print(output,hidden_)
        return torch.permute(output,(0,2,1)), torch.permute(hidden_,(0,2,1))
 

class Latent_Model(nn.Module):
    def __init__(self, params,non_negative_var=True):
        super(Latent_Model,self).__init__()

        self.Tp = params['Tp']  # Tp step Prediction
        self.input_dim = int(params['IN_DIM'])
        self.hidden_dim = int(params['HIDDEN_DIM'])  # Mid Dimension
        self.latent_dim = int(params['LATENT_DIM'])  # Approximate High Dimension

        self.ext_input_dim = params['ext_input_dim']
        self.run_device = params['device']
        self.data_len = params['data_length']  # The data length for parameter estimation
        if params['activation']=='ReLU':
            self.activate=nn.ReLU()
        elif  params['activation']=='Tanh':
            self.activate=nn.Tanh()
        elif params['activation']=='Softplus':
            self.activate=nn.Softplus()
            
        self.with_wrec_perturbation = params['with_wrec_perturbation']
        self.with_b_perturbation = params['with_b_perturbation']
        self.sparsity=params['sparsity']
            
        self.e_clusters=params['e_clusters']
        self.i_clusters=params['i_clusters']
        self.inter_or_intra=params['inter_or_intra']
        
        if params['final_activation']=='ReLU':
            self.final_activate=nn.ReLU()
        elif params['final_activation']=='Softplus':
            self.final_activate=nn.Softplus()

        self.rnn_cell=None
        self.RNN_Type=params['RNN_Type']

        if self.RNN_Type=='EI-RNN':
            self.encode = nn.Sequential(
                torch.nn.Linear(self.input_dim,self.latent_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(self.latent_dim,self.latent_dim),
            )

            self.ei_ratio=params['EI_ratio']
            self.rnn_cell=EI_RNN(input_size=self.ext_input_dim, 
                                hidden_size=self.latent_dim, 
                                output_size=self.input_dim, 
                                output_layers=1,
                                fr_type=params['fr_type'], 
                                ei_ratio=self.ei_ratio,
                                sparsity=self.sparsity,
                                e_clusters=self.e_clusters,
                                i_clusters=self.i_clusters,
                                inter_or_intra=self.inter_or_intra,
                                with_Tanh=params['with_Tanh'],
                                with_wrec_perturbation=self.with_wrec_perturbation,
                                with_b_perturbation=self.with_b_perturbation,
                                device=self.run_device)
            print('run_device:',self.run_device)
        elif self.RNN_Type=='my_VAR':
            self.rnn_cell = VAR_model(A=params['A'],B=params['B'],C=params['C'],train_model=params['Train_model'],
                                    with_wrec_perturbation=self.with_wrec_perturbation,
                                    with_b_perturbation=self.with_b_perturbation,
                                    device=self.run_device)
        
        elif self.RNN_Type=='my_VAR_ori':
            self.rnn_cell = VAR_model_ori(A=params['A'],B=params['B'],C=params['C'],train_model=params['Train_model'],
                                        with_wrec_perturbation=self.with_wrec_perturbation,
                                    with_b_perturbation=self.with_b_perturbation,
                                    device=self.run_device)
    
        elif self.RNN_Type=='my_nVAR':
            self.rnn_cell = nVAR_model(A=params['A'],B=params['B'],C=params['C'],train_model=params['Train_model'],
                                        with_wrec_perturbation=self.with_wrec_perturbation,
                                    with_b_perturbation=self.with_b_perturbation,
                                    device=self.run_device)
        # elif self.RNN_Type=='my_nVAR_ori':
        #     self.rnn_cell = nVAR_model_ori(A=params['A'],B=params['B'],C=params['C'],train_model=params['Train_model'],
        #                             with_wrec_perturbation=self.with_wrec_perturbation,
        #                             with_b_perturbation=self.with_b_perturbation,
        #                             device=self.run_device)

        elif self.RNN_Type=='WC_model':
            self.rnn_cell = Wilson_Cowan(W_EE=params['W_EE'],W_EI=params['W_EI'],W_IE=params['W_IE'],W_II=params['W_II'],B=params['B'],C=params['C'],
                                            train_model=params['Train_model'],dt=params['dt'],device=self.run_device)
   
    def forward(self, X, ext_in, hidden_state=None):
        loss_hid = 0
        loss_pred = 0

        batch, length, ori_dim = X.size()

        loss_fun = torch.nn.SmoothL1Loss(reduction='mean',beta=0.1)

        Tp_prediction = torch.zeros((batch, length-1, self.input_dim), device=self.run_device)
        hidden_prediction = torch.zeros((batch, length-1, self.latent_dim), device=self.run_device)
        if self.RNN_Type=='EI-RNN':
            hidden_ = self.encode(X) # batch * time * hidden
        elif self.RNN_Type=='WC_model' and self.Train_model==False: 
            hidden_ = hidden_state[:, :self.data_len, :].to(self.run_device)
            hidden_prediction = torch.zeros((batch, self.data_len-1, self.latent_dim*2), device=self.run_device)
        else:
            if hidden_state is None:
                if self.RNN_Type=='my_VAR' or self.RNN_Type=='my_nVAR':
                    hidden_ = torch.permute(torch.linalg.pinv(self.rnn_cell.C)@torch.log(torch.exp(torch.permute(X,(0,2,1)))-1),(0,2,1))
                elif self.RNN_Type=='my_VAR_ori':
                    #print(X_.shape,self.rnn_cell.C.shape)
                    hidden_ = torch.permute(torch.linalg.pinv(self.rnn_cell.C)@torch.permute(X,(0,2,1)),(0,2,1))
                else:
                    hidden_ = X
                # if self.non_negative_var==False:
                #     hidden_ = X
            else:
                hidden_ = hidden_state[:, :self.data_len, :]
        
        T = hidden_.shape[1]
        hidden_state = hidden_[:,0:1,:]
        loss=0
        
        for t in range(1,T):
            # teacher forcing
            if (t%self.Tp==0):
                hidden_state = hidden_[:,t-1:t,:]
            
            # forward prediction
            inputs = ext_in[:, t-1:t, :]

            output, hidden_state = self.rnn_cell(inputs, hidden_state)

            Tp_prediction[:, t-1:t, :] = output
            hidden_prediction[:, t-1:t, :] = hidden_state

        loss_hid = loss_fun(hidden_[:,1:,:],hidden_prediction)
        loss_pred = loss_fun(X[:,1:,:],Tp_prediction)

        return loss_hid+loss_pred,loss_hid,loss_pred,Tp_prediction,X[:,1:,:] 

