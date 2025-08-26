import torch
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding

def LLE_dim_reduction(data,n_components=3):
    params = {
        "n_neighbors": 12,
        "n_components": n_components,
        "eigen_solver": "auto",
        "random_state": 0,
    }
    lle=LocallyLinearEmbedding(method="modified", **params)
    lle.fit(data)
    return lle

def FA_dim_reduction(data,n_components=3):
    # initialize the factor analysis
    fa=FactorAnalysis(n_components)
    fa.fit(data)
    return fa

def PCA_dim_reduction(data,n_components=3):
    # initialize the pca
    pca=PCA(n_components)
    pca.fit(data)
    return pca


def TSNE_dim_reduction(data,n_components=3):
    # initialize the tsne
    tsne=TSNE(n_components)
    tsne.fit(data)
    return tsne

def Spectral_Embedding_dim_reduction(data,n_components=3):
    # initialize the spectral embedding
    se=SpectralEmbedding(n_components)
    se.fit(data)
    return se

def EI_Sparse_Mask_(hidden_dim=100,e_clusters=2,i_clusters=1,sparsity=0.0,ei_ratio=4,module_type='Inter'):
    probability=1.0*ei_ratio/(ei_ratio+1)
        
    ee_size=int(hidden_dim*probability)
    ii_size=hidden_dim-int(hidden_dim*probability)
    
    EE_fix=torch.ones((ee_size,ee_size))
    EI_fix=torch.ones((ee_size,ii_size))
    IE_fix=torch.ones((ii_size,ee_size))
    II_fix=torch.ones((ii_size,ii_size))
    
    if module_type=='Intra_with_E1_fixed_but_changed_E2':
        if e_clusters==2:
            e1_matrix_size=int(hidden_dim*probability/2)
            e2_matrix_size=int(hidden_dim*probability)-e1_matrix_size
            
            # E11
            e11_matrix=torch.ones((e1_matrix_size,e1_matrix_size))  # eclusters
            
            # E22
            e22_vector=torch.ones((e2_matrix_size*e2_matrix_size))
            zeros_num = int(e2_matrix_size*e2_matrix_size*sparsity)
            e22_index = [i for i in range(e2_matrix_size*e2_matrix_size)]
            if not zeros_num==0:
                e22_vector[:zeros_num]=0 # 将一部分元素置换为0
                e22_index=torch.randperm(e22_vector.shape[0])
            e22_matrix=torch.reshape(e22_vector[e22_index],(e2_matrix_size,e2_matrix_size))

            e12_matrix=torch.ones((e1_matrix_size,e2_matrix_size))
            e21_matrix=torch.ones((e2_matrix_size,e1_matrix_size))
            
            EE_fix=torch.cat((torch.cat((e11_matrix,e12_matrix),1),torch.cat((e21_matrix,e22_matrix),1)),0)
        
    if module_type=='Intra_with_E2_fixed_but_changed_E1':
        if e_clusters==2:
            e1_matrix_size=int(hidden_dim*probability/2)
            e2_matrix_size=int(hidden_dim*probability)-e1_matrix_size
            # E22
            e22_matrix=torch.ones((e2_matrix_size,e2_matrix_size))
            
            # E11
            e11_vector=torch.ones((e1_matrix_size*e1_matrix_size))  # eclusters
            zeros_num = int(e1_matrix_size*e1_matrix_size*sparsity)
            e11_index = [i for i in range(e1_matrix_size*e1_matrix_size)]
            if not zeros_num==0:
                e11_vector[:zeros_num]=0 # 将一部分元素置换为0
                e11_index=torch.randperm(e11_vector.shape[0])
            e11_matrix=torch.reshape(e11_vector[e11_index],(e1_matrix_size,e1_matrix_size))


            e12_matrix=torch.ones((e1_matrix_size,e2_matrix_size))
            e21_matrix=torch.ones((e2_matrix_size,e1_matrix_size))
            
            EE_fix=torch.cat((torch.cat((e11_matrix,e12_matrix),1),torch.cat((e21_matrix,e22_matrix),1)),0)
                
    if module_type=='Inter_with_E1_I2_E2_I1_sparsity':
        if e_clusters==2 and i_clusters==2:
            e1_matrix_size=int(hidden_dim*probability/2)
            e2_matrix_size=int(hidden_dim*probability)-e1_matrix_size
            e11_matrix=torch.ones((e1_matrix_size,e1_matrix_size))  # eclusters
            e22_matrix=torch.ones((e2_matrix_size,e2_matrix_size))
            
            
            e12_vector=torch.ones((e1_matrix_size*e2_matrix_size))
            e21_vector=torch.ones((e2_matrix_size*e1_matrix_size))
            
            zeros_num = int(e1_matrix_size*e2_matrix_size*sparsity)
            e12_index = [i for i in range(e1_matrix_size*e2_matrix_size)]
            e21_index = [i for i in range(e1_matrix_size*e2_matrix_size)]
            if not zeros_num==0:
                e12_vector[:zeros_num]=0 # 将一部分元素置换为0
                e21_vector[:zeros_num]=0 # 将一部分元素置换为0
                
                e12_index=torch.randperm(e12_vector.shape[0])
                e21_index=torch.randperm(e21_vector.shape[0])
            
            e12_matrix=torch.reshape(e12_vector[e12_index],(e1_matrix_size,e2_matrix_size))
            e21_matrix=torch.reshape(e21_vector[e21_index],(e2_matrix_size,e1_matrix_size))
        
            EE_fix=torch.cat((torch.cat((e11_matrix,e12_matrix),1),torch.cat((e21_matrix,e22_matrix),1)),0)
            
            i1_matrix_size=int((hidden_dim-int(hidden_dim*probability))/2)
            i2_matrix_size=(hidden_dim-int(hidden_dim*probability))-i1_matrix_size
            
            i11_matrix=torch.ones((i1_matrix_size,i1_matrix_size))  # eclusters
            i22_matrix=torch.ones((i2_matrix_size,i2_matrix_size))
            
            i12_matrix=torch.ones((i1_matrix_size,i2_matrix_size))
            i21_matrix=torch.ones((i2_matrix_size,i1_matrix_size))
        
            II_fix=torch.cat((torch.cat((i11_matrix,i12_matrix),1),torch.cat((i21_matrix,i22_matrix),1)),0)
            
            
            e1_i2_vector=torch.ones((e1_matrix_size*i2_matrix_size))
            i2_e1_vector=torch.ones((i2_matrix_size*e1_matrix_size))
            e2_i1_vector=torch.ones((e2_matrix_size*i1_matrix_size))
            i1_e2_vector=torch.ones((i1_matrix_size*e2_matrix_size))

            zeros_num1 = int(e1_matrix_size*i2_matrix_size*sparsity)
            e1_i2_index = [i for i in range(e1_matrix_size*i2_matrix_size)]
            if not zeros_num1==0:
                e1_i2_vector[:zeros_num1]=0
                e1_i2_index=torch.randperm(e1_i2_vector.shape[0])
                
            zeros_num2 = int(e2_matrix_size*i1_matrix_size*sparsity)
            e2_i1_index = [i for i in range(e2_matrix_size*i1_matrix_size)]
            if not zeros_num2==0:    
                e2_i1_vector[:zeros_num2]=0
                e2_i1_index=torch.randperm(e2_i1_vector.shape[0])
                
            zeros_num3 = int(i1_matrix_size*e2_matrix_size*sparsity)
            i1_e2_index = [i for i in range(i1_matrix_size*e2_matrix_size)]
            if not zeros_num3==0:    
                i1_e2_vector[:zeros_num3]=0
                i1_e2_index=torch.randperm(i1_e2_vector.shape[0])
            
            zeros_num4 = int(e1_matrix_size*i2_matrix_size*sparsity)
            i2_e1_index = [i for i in range(e1_matrix_size*i2_matrix_size)]
            if not zeros_num4==0:    
                i2_e1_vector[:zeros_num4]=0
                i2_e1_index=torch.randperm(i2_e1_vector.shape[0])
                
            e1_i2_matrix=torch.reshape(e1_i2_vector[e1_i2_index],(e1_matrix_size,i2_matrix_size))
            e2_i1_matrix=torch.reshape(e2_i1_vector[e2_i1_index],(e2_matrix_size,i1_matrix_size))
            i1_e2_matrix=torch.reshape(i1_e2_vector[i1_e2_index],(i1_matrix_size,e2_matrix_size))
            i2_e1_matrix=torch.reshape(i2_e1_vector[i2_e1_index],(i2_matrix_size,e1_matrix_size))    
            
            e1_i1_matrix=torch.ones((e1_matrix_size,i1_matrix_size))
            e2_i2_matrix=torch.ones((e2_matrix_size,i2_matrix_size))
            i1_e1_matrix=torch.ones((i1_matrix_size,e1_matrix_size))
            i2_e2_matrix=torch.ones((i2_matrix_size,e2_matrix_size))
            
            EI_fix=torch.cat((torch.cat((e1_i1_matrix,e1_i2_matrix),1),torch.cat((e2_i1_matrix,e2_i2_matrix),1)),0)
            IE_fix=torch.cat((torch.cat((i1_e1_matrix,i1_e2_matrix),1),torch.cat((i2_e1_matrix,i2_e2_matrix),1)),0)
            
                
    if module_type=='Inter':
        if e_clusters==2:
            e1_matrix_size=int(hidden_dim*probability/2)
            e2_matrix_size=int(hidden_dim*probability)-e1_matrix_size
            e11_matrix=torch.ones((e1_matrix_size,e1_matrix_size))  # eclusters
            e22_matrix=torch.ones((e2_matrix_size,e2_matrix_size))
            
            
            e12_vector=torch.ones((e1_matrix_size*e2_matrix_size))
            e21_vector=torch.ones((e2_matrix_size*e1_matrix_size))
            
            zeros_num = int(e1_matrix_size*e2_matrix_size*sparsity)
            e12_index = [i for i in range(e1_matrix_size*e2_matrix_size)]
            e21_index = [i for i in range(e1_matrix_size*e2_matrix_size)]
            if not zeros_num==0:
                e12_vector[:zeros_num]=0 # 将一部分元素置换为0
                e21_vector[:zeros_num]=0 # 将一部分元素置换为0
                
                e12_index=torch.randperm(e12_vector.shape[0])
                e21_index=torch.randperm(e21_vector.shape[0])
            
            e12_matrix=torch.reshape(e12_vector[e12_index],(e1_matrix_size,e2_matrix_size))
            e21_matrix=torch.reshape(e21_vector[e21_index],(e2_matrix_size,e1_matrix_size))
        
            EE_fix=torch.cat((torch.cat((e11_matrix,e12_matrix),1),torch.cat((e21_matrix,e22_matrix),1)),0)
        else:
            ee_size=int(hidden_dim*probability)
            EE_fix=torch.ones((ee_size,ee_size))
            
        if i_clusters==2:
            i1_matrix_size=int((hidden_dim-int(hidden_dim*probability))/2)
            i2_matrix_size=(hidden_dim-int(hidden_dim*probability))-i1_matrix_size
            i11_matrix=torch.ones((i1_matrix_size,i1_matrix_size))  # eclusters
            i22_matrix=torch.ones((i2_matrix_size,i2_matrix_size))
            
            
            i12_vector=torch.ones((i1_matrix_size*i2_matrix_size))
            i21_vector=torch.ones((i1_matrix_size*i2_matrix_size))
            
            zeros_num = int(i1_matrix_size*i2_matrix_size*sparsity)
            i12_index = [i for i in range(i1_matrix_size*i2_matrix_size)]
            i21_index = [i for i in range(i1_matrix_size*i2_matrix_size)]
            if not zeros_num==0:
                i12_vector[:zeros_num]=0 # 将一部分元素置换为0
                i21_vector[:zeros_num]=0 # 将一部分元素置换为0
            
                i12_index=torch.randperm(i12_vector.shape[0])
                i21_index=torch.randperm(i21_vector.shape[0])
            
            i12_matrix=torch.reshape(i12_vector[i12_index],(i1_matrix_size,i2_matrix_size))
            i21_matrix=torch.reshape(i21_vector[i21_index],(i2_matrix_size,i1_matrix_size))
        
            II_fix=torch.cat((torch.cat((i11_matrix,i12_matrix),1),torch.cat((i21_matrix,i22_matrix),1)),0)
        else:
            ii_size=hidden_dim-int(hidden_dim*probability)
            II_fix=torch.ones((ii_size,ii_size))
            
            
            
    return torch.cat((torch.cat((EE_fix,EI_fix),1),torch.cat((IE_fix,II_fix),1)),0)