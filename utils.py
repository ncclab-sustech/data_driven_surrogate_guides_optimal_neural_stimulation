import os
import numpy as np
import torch
from scipy.io import savemat
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, explained_variance_score, matthews_corrcoef
import matplotlib.pyplot as plt
from pytorch_tool.pytorchtools import EarlyStopping

def l2_regularization(model, l2_lambda=0.01):

    l2_reg = torch.tensor(0., device=next(model.parameters()).device)

    for param in model.parameters():

        if param.requires_grad:

            l2_reg += torch.norm(param, p=2)  # L2范数

    return l2_lambda * l2_reg

def train_model(model,train_dataloader,validation_dataloader,optimizer,epochs,ckpts,device='cuda:0'):
    early_stopping = EarlyStopping(patience=20,path=ckpts, verbose=True)
    
    for epoch in range(1, epochs + 1):
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        hid_losses = []
        pred_losses = []   

        model.train() # prep model for training
        for batch, (X, ext_in,list_) in enumerate(train_dataloader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            loss,loss_hid,loss_pred,prediction,gt = model(X.to(device), ext_in.to(device))
            parameters = model.parameters()
            # loss+= l2_regularization(model,1e-4) #sum(p.pow(2).sum() for p in parameters) * 0.0001 
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            hid_losses.append(loss_hid.item())
            pred_losses.append(loss_pred.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for batch, (X, ext_in,list_) in enumerate(validation_dataloader, 1):
            # forward pass: compute predicted outputs by passing inputs to the model
            loss,loss_hid,loss_pred,prediction,gt = model(X.to(device), ext_in.to(device))

            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        hid_loss = np.average(hid_losses)
        pred_loss = np.average(pred_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(epochs))
        
        if epoch%20==0:
            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f} ' +
                        f'hid_loss: {hid_loss:.5f} '+
                        f'pred_loss: {pred_loss:.5f}')
            print(print_msg)
        if epoch%1==0:
            torch.save(model.state_dict(), './checkpoints/nonlinear_model_epoch_%d.pth'%(epoch))
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if epoch>200:
            early_stopping(valid_loss, model)
            #ckpts=ckpt_suffix+'_epoch_%d.pth'%(epoch)
            #torch.save(model.state_dict(), ckpts)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    #del early_stopping
    # load the last checkpoint with the best model
    # evaluate the performance
    print('load model and testing\n')

    return model


def model_prediction_(model,X,ext_in,Tp_length=5,hidden_=None,device='cuda:0'):
    #print(X.shape,ext_in.shape)
    # if hidden_ is not None:
    #     print(hidden_.shape)
    loss,loss_hid,loss_pred,prediction,gt = model(X.to(device), ext_in.to(device),hidden_state=hidden_)
    # print(prediction.shape)
    return prediction.detach().cpu().numpy()[0,-Tp_length:,:],gt.detach().cpu().numpy()[0,-Tp_length:,:]


def model_prediction_forloop(model,input_data,start_index,prediction_len,Tp_length=5,device='cuda:0'):

    prediction_result = np.zeros([prediction_len, 4])
    gt_ = np.zeros([prediction_len, 4])
    # print(input_data.X.shape,start_index)
    # 4model.ar_order=0
    if start_index>input_data.X.shape[0]-prediction_len:
        start_index=input_data.X.shape[0]-prediction_len
    if start_index<prediction_len:
        start_index=prediction_len+10
    hidden_state=None
    if not model.RNN_Type=='DeepKoopman':
        for steps in range(0,prediction_len,Tp_length):
            X = torch.reshape(torch.from_numpy(input_data.X[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, 4)).type(torch.float)

            Inputs = torch.reshape(torch.from_numpy(input_data.ext_input[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, model.ext_input_dim)).type(torch.float)
            #print(X,Inputs)
            # print(prediction_result.shape)
            # print(X.shape,Inputs.shape,steps)
            if model.RNN_Type=='WC_model':
                hidden_state = torch.reshape(torch.from_numpy(input_data.hidden_data[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, -1)).type(torch.float).to(device)
            if model.RNN_Type == 'my_nVAR' and model.Train_model==False:
                hidden_state = torch.reshape(torch.from_numpy(input_data.hidden_data[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, -1)).type(torch.float).to(device)
            
            prediction_result[steps:steps+Tp_length,:],gt_[steps:steps+Tp_length,:]=model_prediction_(model,X,Inputs,Tp_length,hidden_state,device=device)
            # del X, Inputs
    else:
        for steps in range(0,prediction_len,model.data_len):
            X = torch.reshape(torch.from_numpy(input_data.X[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, 4)).type(torch.float)

            Inputs = torch.reshape(torch.from_numpy(input_data.ext_input[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, model.ext_input_dim)).type(torch.float)
            #print(X,Inputs)
            # print(prediction_result.shape)
            # print(X.shape,Inputs.shape,steps)
            print(model.RNN_Type)
            if model.RNN_Type=='WC_model':
                hidden_state = torch.reshape(torch.from_numpy(input_data.hidden_data[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, -1)).type(torch.float)
            
            if model.RNN_Type == 'my_nVAR' and model.Train_model==False:
                hidden_state = torch.reshape(torch.from_numpy(input_data.hidden_data[start_index - model.data_len-Tp_length+steps:(start_index+steps),:].astype('float32')),
                                (1, model.data_len + Tp_length, -1)).type(torch.float)
            # else:
            #     hidden_state = X
            prediction_result[steps:steps+Tp_length,:],gt_[steps:steps+Tp_length,:]=model_prediction_(model,X,Inputs,Tp_length,hidden_state,device=device)
            # del X, Inputs
    return prediction_result,gt_


def model_evaluation(model,input_data,start_indexs,Tp_length,device):
    print('total_params:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    statistic_result1 = {
                'R2': np.zeros((1, 30)),
                'mse': np.zeros((1, 30)),
                'EV': np.zeros((1, 30)),
                'CC': np.zeros((1, 30)),
                'R2_by_channel': np.zeros((1, 30, 4)),
                'EV_by_channel': np.zeros((1, 30, 4)),
                'mse_by_channel': np.zeros((1, 30, 4)),
            }
    time_length = 200 
    prediction_result = np.zeros([30, time_length, 4])
    gt_ = np.zeros([30, time_length, 4])


    for run_index in range(30):
        # print('step %d'%run_index)
        if start_indexs[-run_index] < input_data.X.shape[0] :
            start_index = start_indexs[-run_index]
        else:
            start_index = input_data.X.shape[0]
        # print('start_index:',start_index)
        #from sklearn.feature_selection import r_regression
        prediction_result[run_index,:,:],gt_[run_index,:,:]=model_prediction_forloop(model,input_data,start_index,time_length,Tp_length,device)
        statistic_result1['R2'][0, run_index] = r2_score(gt_[run_index,:, :], prediction_result[run_index, :, :],multioutput='variance_weighted')
        statistic_result1['mse'][0, run_index] =+ mean_squared_error(gt_[run_index,:, :],prediction_result[run_index, :, :])
        statistic_result1['EV'][0, run_index] = explained_variance_score(gt_[run_index,:, :],prediction_result[run_index, :, :],multioutput='variance_weighted')
        
        corr_ = np.zeros((4,))
        for k in range(4):
            #print(np.corrcoef(gt_[run_index,:, k],prediction_result[run_index,:,k]))
            corr_[k]=np.corrcoef(gt_[run_index,:, k],prediction_result[run_index,:,k])[0,1]
        statistic_result1['CC'][0, run_index] = np.mean(corr_)
        
        if run_index%15==0:
            plt.figure(figsize=(6,2.5))
            for i in range(4):
                plt.plot(gt_[run_index,:200, i]-2*i)
                plt.plot(prediction_result[run_index,:200, i]-2*i,'--')


    print("R2:", np.mean(statistic_result1['R2'][0]), "_std:",
            np.std(statistic_result1['R2'][0]), "\n")
    print("MSE:", np.mean(statistic_result1['mse'][0]), "_std:",
            np.std(statistic_result1['mse'][0]), "\n")  # ,
    print("EV:", np.mean(statistic_result1['EV'][0]), "_std:",
            np.std(statistic_result1['EV'][0]), "\n")
    print("CC:", np.mean(statistic_result1['CC'][0]), "_std:",
            np.std(statistic_result1['CC'][0]), "\n")
    
    return statistic_result1


def return_var_model_params(hidden_dim,Tp_length,data_length,device):
    var_model_params={'L_factors': 0.05,
                    'EI_ratio': 4,
                    'l1_reg': False,
                    'learning_rate': 0.0005,
                    'with_Tanh': False,
                    'sparsity': 0,
                    'e_clusters': 2,
                    'i_clusters': 1,
                    'inter_or_intra': 'Inter',
                    'l2_norm': 0.001,
                    'Tp': Tp_length,
                    'Sample_Size': 20000,
                    'data_length': data_length,  # Hyperparameter
                    'HIDDEN_DIM': hidden_dim,
                    'LATENT_DIM': hidden_dim,
                    'ext_input_dim':4,
                    'AR_order':1,
                    'IN_DIM':4,
                    'activation': 'ReLU',
                    'final_activation': 'Softplus',
                    'RNN_Type': 'my_VAR',#'EI-RNN', #  'AR', #'GRU' ,# 'AR' #
                    'l1_reg': False,
                    'train_day': '05-22/',
                    'cross_validation': False,
                    'init_ckpt_path':None,
                    'data_type':'nonlinear',
                    'num_folder': 1,
                    'device':device,
                    'Train_model':True,
                    'with_wrec_perturbation':False,
                    'with_b_perturbation':False,

                    'with_Tanh': False ,
                    'fr_type': True ,
                    'with_sigmoid': False,
                
                    'A':torch.randn((hidden_dim,hidden_dim)),
                    'B':torch.randn((hidden_dim,4)),
                    'C':torch.randn((4,hidden_dim)),
                    'epochs':500}
    return var_model_params

def return_eirnn_model_params(hidden_dim,Tp_length,data_length,with_tanh=True,device='cuda:0'):
    model_params={'L_factors': 0.05,
                    'EI_ratio': 4,
                    'l1_reg': False,
                    'learning_rate': 0.001,
                    #'with_Tanh': False,
                    'sparsity': 0,
                    'e_clusters': 2,
                    'i_clusters': 1,
                    'inter_or_intra': 'Inter',
                    'l2_norm': 0.001,
                    'Tp': Tp_length,
                    'Sample_Size': 20000,
                    'data_length': data_length, #101,  # Hyperparameter
                    'HIDDEN_DIM': hidden_dim,
                    'LATENT_DIM': hidden_dim,
                    'ext_input_dim':4,
                    'IN_DIM':4,
                    'AR_order':1,
                    'activation': 'ReLU',
                    'final_activation': 'Softplus',
                    'RNN_Type': 'EI-RNN', # 'my_VAR',# 'AR', #'GRU' ,# 'AR' #
                    'l1_reg': False,
                    'with_Tanh': with_tanh,
                    'fr_type':True,
                    'with_sigmoid':False,
                    'train_day': '05-22/',
                    'cross_validation': False,
                    'init_ckpt_path':None,
                    'data_type':'nonlinear',
                    'num_folder': 1,
                    'device':device,
                    'Train_model':True,
                    'with_wrec_perturbation':False,
                    'with_b_perturbation':False,
                    'A':torch.randn((hidden_dim,hidden_dim)),
                    'B':torch.randn((hidden_dim,4)),
                    'C':torch.randn((4,hidden_dim)),
                    'epochs':500}
    return model_params

def return_perturb_model_params(model,Tp_length,data_length,with_wrec_perturbation,with_b_perturbation,device):
    hidden_dim = model.hidden_dim
    model_params={'L_factors': 0.05,
                    'EI_ratio': 4,
                    'l1_reg': False,
                    'learning_rate': 0.001,
                    #'with_Tanh': False,
                    'sparsity': 0,
                    'e_clusters': 2,
                    'i_clusters': 1,
                    'inter_or_intra': 'Inter',
                    'l2_norm': 0.001,
                    'Tp': Tp_length,
                    'Sample_Size': 20000,
                    'data_length': data_length, #101,  # Hyperparameter
                    'HIDDEN_DIM': hidden_dim,
                    'LATENT_DIM': hidden_dim,
                    'ext_input_dim':4,
                    'IN_DIM':4,
                    'AR_order':1,
                    'activation': 'ReLU',
                    'final_activation': 'Softplus',
                    'RNN_Type': model.RNN_Type, # 'my_VAR',# 'AR', #'GRU' ,# 'AR' #
                    'l1_reg': False,
                    'with_Tanh': model.rnn_cell.with_Tanh,
                    'fr_type': model.rnn_cell.fr_type,
                    'with_sigmoid':model.rnn_cell.with_sigmoid,
                    'train_day': '05-22/',
                    'cross_validation': False,
                    'init_ckpt_path':None,
                    'data_type':'nonlinear',
                    'num_folder': 1,
                    'device':device,
                    'Train_model':True,
                    'with_wrec_perturbation':with_wrec_perturbation,
                    'with_b_perturbation':with_b_perturbation,
                    'A':torch.randn((hidden_dim,hidden_dim)),
                    'B':torch.randn((hidden_dim,4)),
                    'C':torch.randn((4,hidden_dim)),
                    'epochs':500}
    return model_params
