# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 15:55:02 2022

@author: xiyaojin
"""
import numpy as np
import os
import imp
import tensorflow as tf
from scipy import ndimage
import torch
def get_patient_list(dataset):

    if dataset=='reference':
        f=open(r'Y:\chpc2.0\data\data.3D_2mm_fixed\test_normal_CL.txt')
        lines=f.readlines()
        patients=[l[7:].strip('\n') for l in lines] 
    elif dataset=='extra':
        f=open(r'Y:\chpc2.0\data\data.3D_2mm_fixed\thesis_other.txt')
        lines=f.readlines()
        patients=[l[7:].strip('\n') for l in lines] 
    elif dataset in ['av','ca','ma','para','prone','para-base']:
        mp={'av':'AV','ca':'CA','ma':'MA','para':'PARA','prone':'prone','para-base':'PARA'}
        f=open(os.path.join(r'Y:\irb_data',mp[dataset],'list.txt'))
        lines=f.readlines()
        patients=[l.strip('\n') for l in lines]
    return patients

def cal_dice(a,b,mode='all'):
    dsc=0
    labels={'all':np.arange(1,10),'small':np.arange(1,6),'large':np.arange(6,10)}
    for i in labels[mode]:
        x=np.zeros_like(a)
        y=np.zeros_like(a)
        x[a==i]=1
        y[b==i]=1
        dsc+=2*np.sum(x*y)/(np.sum(x)+np.sum(y)+1e-6)
    return dsc/len(labels[mode])

def cal_dist(ref,data,mode,fraction=1):
    dist=[]
    n=data.shape[0]
    for i in range(n):
        sample=data[i,:]
        if mode=='euclidean':
            d=np.mean(np.sqrt(np.sum((ref-sample)**2,axis=1)))
        elif mode=='cosine':
            temp=0
            for j in range(ref.shape[0]):
                temp+=np.sum(ref[j,:]*sample)/(np.sum(ref[j,:]**2)**0.5*np.sum(sample**2)**0.5)
            d=temp/ref.shape[0]
        elif mode=='cosine_knn':
            temp=np.zeros(ref.shape[0])
            for j in range(ref.shape[0]):
                temp[j]=np.sum(ref[j,:]*sample)/(np.sum(ref[j,:]**2)**0.5*np.sum(sample**2)**0.5)    
            d=np.sort(temp)[1:-1:].mean()
            
        elif mode=='cosine_avg':
            ref_avg=np.mean(ref,axis=0)
            d=np.sum(ref_avg*sample)/(np.sum(ref_avg**2)**0.5*np.sum(sample**2)**0.5)
        elif mode=='fraction':
            d=np.mean((np.sum(np.abs((ref-sample))**fraction,axis=1))**(1/fraction))
        dist.append(d)

            
    return np.array(dist)   

def rmse(a,b):
    n=len(a)
    return (np.sum((a-b)**2)/n)**0.5

noise_levels=[0, 3, 4, 5, 3, 4, 5, 0, 0, 0, 2, 2, 0, 4, 0, 0, 3, 5, 5, 0]

def load_labels(structures,model_id):
    labels={}
    datasets=['reference','av','ca','ma','para','prone','extra']
    if model_id<=17:
        if model_id>=0:
            labels['reference']=np.load(os.path.join(r'Y:\irb_test_results\normal',f"{structures}.npy"))[model_id,:]
            labels['extra']=np.load(os.path.join(r'Y:\irb_test_results\normal',f"{structures}-extra.npy"))[model_id,:]
            labels['av']=np.load(os.path.join(r'Y:\irb_test_results\AV',f"{structures}.npy"))[model_id,:]
            
            y_temp=np.load(os.path.join(r'Y:\irb_test_results\other',f"{structures}.npy"))[model_id,:]
        else:
            labels['reference']=np.load(os.path.join(r'Y:\irb_test_results\normal',f"{structures}.npy")).mean(axis=0)
            labels['extra']=np.load(os.path.join(r'Y:\irb_test_results\normal',f"{structures}-extra.npy")).mean(axis=0)
            labels['av']=np.load(os.path.join(r'Y:\irb_test_results\AV',f"{structures}.npy")).mean(axis=0)
            
            y_temp=np.load(os.path.join(r'Y:\irb_test_results\other',f"{structures}.npy")).mean(axis=0)      
            
        i=0
        for ds in datasets[2:-1]:
            
            if ds=='ma':
                i+=4
            n_samples=len(get_patient_list(ds))
            labels[ds]=y_temp[i:i+n_samples]
            i+=n_samples
    else: # for transformers
        for ds in datasets:
            labels[ds]=np.load(os.path.join(r'y:\irb_test_results\normal',model_seg_path[model_id],f'DCs-{ds}-{structures}.npy'))
    # overwite PARA
    DCs_noisy=np.load(r'Y:\irb_test_results\normal\DCs-noisy.npy')
    if model_id==19:
        model_id=18
    y_para=np.array([DCs_noisy[model_id,i,noise_levels[i]] for i in range(20)])
    labels['para']=y_para
    return labels


#%%    

def load_model(experiment_name, vae_type='VAE',checkpoint='',ret_cf = False):
    
    if checkpoint == '':
        model_name = 'final_model.pth'
    else:
        model_name = f'checkpoint{checkpoint}.pth'
    
    if vae_type=='VAE':
        config_path=os.path.join(r'y:\experiment_output\VAE',experiment_name,'config.py')
        cf=imp.load_source('config',config_path)
        cf.latent_dim=cf.z_dim
        
        models=imp.load_source('models',os.path.join(r'y:\experiment_output\VAE',experiment_name,'models.py'))
        model=models.GMVAE2_3D(cf)
        model.load_weights(os.path.join(r'y:\experiment_output\VAE',experiment_name,'saved_weights'))   
        if ret_cf:
            return model, cf
    
    elif vae_type in {'VQ_VAE', 'DAE', 'contrastive_learning'}:
        models=imp.load_source('models',os.path.join(r'y:\experiment_output',vae_type,experiment_name,'models.py'))
        model = torch.load(os.path.join(r'y:\experiment_output',vae_type,experiment_name,model_name),map_location=torch.device('cpu'))    
    
    return model

        

#%%   

def add_noise(img,noise):
    img=img/255.0
    img=np.random.poisson(img*255/noise)*noise
    return img
#%% --------------utils,constants----------------------------


datasets=['reference','av','ca','ma','para','prone']

data_path={
    'reference':r'Y:\chpc2.0\data\data.3D_2mm_fixed\images',
    'av':r'Y:\irb_data_preprocessed\nii\images',
    'ca':r'Y:\irb_data_preprocessed\nii\images',
    'ma':r'Y:\irb_data_preprocessed\nii\images',
    'para':r'Y:\irb_data_preprocessed\nii\images',
    'prone':r'Y:\irb_data_preprocessed\nii\images'
    }


label_path={
    'reference':r'Y:\chpc2.0\data\data.3D_2mm_fixed\labels',
    'extra':r'Y:\chpc2.0\data\data.3D_2mm_fixed\labels',
    'av':r'Y:\irb_data_preprocessed\nii\labels',
    'ca':r'Y:\irb_data_preprocessed\nii\labels',
    'ma':r'Y:\irb_data_preprocessed\nii\labels',
    'para':r'Y:\irb_data_preprocessed\nii\labels',
    'para-base':r'Y:\irb_data_preprocessed\nii\labels',
    'prone':r'Y:\irb_data_preprocessed\nii\labels'
    }

model_seg_path={0:'sizes\s4',1:'sizes\s2',2:'sizes\l0',3:'sizes\l2',4:'sizes\l6',
                5:'losses\ce',6:'losses\dice',7:'losses\dice-ce',8:'losses\mdl',9:'losses\mse',10:'losses\w-ce',11:'losses\w-dice',
                12:'aug\\00',13:'aug\\20',14:'aug\\40',15:'aug\\60',16:'aug\\80',17:'aug\\100',18:'transformer\\11-2-2',19:'transformer\\10-22-2-aug05-trans20-augPerWarm-lr1e4'}

experiment_name='4-17-1-gmvae2-3d-f256-d1024-down5True-k8-noAug'
vae_label_name='9-25-2-gmvae2-3d-label-f128-d256-k2-klMean'
model_id=19
#%%---------------------get_image_recon_error --------------
"""
does not work, because the vae was trained with all data
"""
def _get_image_recon(experiment_name,dataset):
    #nonlocal cf
    
    check_path=os.path.join(r'y:\experiment_output\VAE',experiment_name,f'recon_{dataset}.npy')
    if os.path.exists(check_path):
        M=np.load(check_path)
        return M
    model,cf=load_vae_model(experiment_name)
    patients=get_patient_list(dataset)
    M=np.zeros((len(patients)))
    for i,p in enumerate(patients):
        x=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))
        
        x=(x/255.0).astype(np.float32)
        x=tf.expand_dims(x,0)
        x=tf.expand_dims(x,-1)
        mean,var=model.encode(x)    
        x_rec=model.decode(mean)
        x_rec=tf.math.sigmoid(x_rec)
        error=np.mean((x_rec-x)**2)
        M[i]=error
    np.save(check_path,M)
    return M

def get_image_recon(experiment_name):
         
    feature_img_recon_error={}
    for ds in datasets:
        print(ds)
        feature_img_recon_error[ds]=_get_image_recon(experiment_name,ds)
    return feature_img_recon_error

#feature_img_recon_error=get_image_recon(experiment_name)
#%%---------------------get metal volume-----------------------------
"""
    Seems not working! VAE latent distance is better to represent metal
"""
def get_metal_volume(dataset,th=180):
    patients=get_patient_list(dataset)
    H=[]
    for i,p in enumerate(patients):
        img=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))    
        if dataset=='reference':
            label=np.load(os.path.join(r'Y:\chpc2.0\data\data.3D_2mm_fixed\labels',f"patient{p}.npy"))
        else:
            label=np.load(os.path.join(r'Y:\irb_data_preprocessed\nii\labels',f"patient{p}.npy"))
            
        #label=binary_erosion((label>0).astype(int),structure=np.ones((5,5,5)))
        img_heart=img*(label>0)
        
        volume=np.sum(img_heart>th)
        H.append(volume)
    return np.stack(H)   


# volumes={}
# for ds in datasets:
#     print(ds)
#     volumes[ds]=get_metal_volume(ds)
    

# feature_metal_volume={}
# for ds in datasets:
#     ref=volumes['reference']
#     ref=np.expand_dims(ref,axis=-1)
#     data=np.expand_dims(volumes[ds],axis=-1)
#     feature_metal_volume[ds]=cal_dist(ref,data,mode='euclidean')   

#%%---------------------get histogram features-----------------------------

def get_histogram(dataset,i_range=(100,151)):
    
    patients=get_patient_list(dataset)
    H=[]
    for i,p in enumerate(patients):
        img=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))

        n_bins=(i_range[1]-i_range[0])//3
        hist=np.histogram(img.flatten(),bins=n_bins,range=i_range,density=False)[0]  
        H.append(hist)
    return np.stack(H)

# histograms={}
# for ds in datasets:
#     histograms[ds]=get_histogram(ds,i_range=(100,151))
    
# feature_histogram_dist={}
# for ds in datasets:
#     ref=histograms['reference']
#     feature_histogram_dist[ds]=cal_dist(ref,histograms[ds],mode='cosine')
#%%-------------------get noise features----------------------------
def get_noise_level(dataset):
    patients=get_patient_list(dataset)
    
    check_path=os.path.join(r'y:\irb_data_features',dataset,'noise_level.npy')
    if os.path.exists(check_path):
        N=np.load(check_path)
        return N
    N=[]
    for i,p in enumerate(patients):
        img=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))       
        img_clean=ndimage.median_filter(img,3)
        mse=np.mean(img_clean-img)**2
        N.append(mse)
    np.save(check_path,N)
    return np.array(N)

# feature_noise_level={}
# for ds in datasets:
#     feature_noise_level[ds]=get_noise_level(ds)