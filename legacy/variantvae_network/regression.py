# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:52:56 2022

@author: xiyaojin
"""

import imp
import os
import json
import numpy as np
import tensorflow as tf

from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from scipy.stats import ranksums,f_oneway,wilcoxon,ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures,normalize
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage import gaussian_laplace
from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor
import torch
from regression_utils import get_patient_list,cal_dice,cal_dist,load_model,rmse,add_noise
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import json

#%% --------------utils,constants----------------------------


datasets=['reference','av','ca','ma','para','prone','extra']
noises=[0.01,1,2,4,6,8]
noise_name=['001','1','2','4','6','8']
noise_levels=[0, 3, 4, 5, 3, 4, 5, 0, 0, 0, 2, 2, 0, 4, 0, 0, 3, 5, 5, 0] #randomly generated for PARA
data_path={
    'reference':r'Y:\chpc2.0\data\data.3D_2mm_fixed\images',
    'extra':r'Y:\chpc2.0\data\data.3D_2mm_fixed\images',
    'av':r'Y:\irb_data_preprocessed\nii\images',
    'ca':r'Y:\irb_data_preprocessed\nii\images',
    'ma':r'Y:\irb_data_preprocessed\nii\images',
    'para':r'Y:\irb_data_preprocessed\nii\images',
    'para-base':r'Y:\irb_data_preprocessed\nii\images',
    'prone':r'Y:\irb_data_preprocessed\nii\images',
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

model2idx={'s4':0,'s2':1,'l0':2,'l2':3,'l6':4,
           'ce':5,'dice':6,'dice-ce':7,'mdl':8,'mse':9,'w-ce':10,'w-dice':11,
           '00':12,'20':13,'40':14,'60':15,'80':16,'100':17,'transformer':19}

model_seg_path={0:'sizes\s4',1:'sizes\s2',2:'sizes\l0',3:'sizes\l2',4:'sizes\l6',
                5:'losses\ce',6:'losses\dice',7:'losses\dice-ce',8:'losses\mdl',9:'losses\mse',10:'losses\w-ce',11:'losses\w-dice',
                12:'aug\\00',13:'aug\\20',14:'aug\\40',15:'aug\\60',16:'aug\\80',17:'aug\\100',18:'transformer\\11-2-2',19:'transformer\\10-22-2-aug05-trans20-augPerWarm-lr1e4',
                20:'siemens\chambers',21:'siemens\wh'}


#experiment_name='11-13-1-gmvae2-3d-f256-d1024-k8-heldoutTest'
experiment_name='11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE'
vae_label_name='9-25-2-gmvae2-3d-label-f128-d256-k2-klMean'
model_id=0

#%%---------------Load labels-------------------------
structures='ALL'

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
    if model_id<20:
        DCs_noisy=np.load(r'Y:\irb_test_results\normal\DCs-noisy.npy')
        if model_id==19:
            model_id=18
        y_para=np.array([DCs_noisy[model_id,i,noise_levels[i]] for i in range(20)])
        labels['para']=y_para
    return labels

labels=load_labels(structures,model_id)
    #%%------------table: test results-----------
# model_list=list(range(18))+[19]
model_list=[0]
datasets=['reference','av','ca','ma','para','prone','extra']

P=[]
for tmp_model_id in model_list:
    mean=[]
    std=[]
    p=[]
    labels=load_labels('ALL',tmp_model_id)
    #datasets=['ca']
    for ds in datasets:
        y=labels[ds]
        mean.append(np.mean(y))
        std.append(np.std(y))
        _,p_tmp=ranksums(labels['reference'],y)
        p.append(p_tmp)
    print(datasets)
    print(np.round(mean,3))
    #print(np.round(std,3))
    print(p)
    P.append(p[0])

    #%%----------Figure: models VS DSC different datasets----------

ds = 'para'

model_list=list(range(18))+[19]
DSC = []
for tmp_model_id in model_list:
    
    y = load_labels('ALL',tmp_model_id)[ds]
    DSC.append(y)
    
plt.figure(figsize=(18,5))
plt.boxplot(DSC)

    #%%--------Figure: datasets VS DSC, different models
tmp_datasets = ['reference','av','ca','ma','prone','para','extra']
tmp_model_id=20
DSC = []
for ds in tmp_datasets:
    y = load_labels('ALL',tmp_model_id)[ds]
    if ds == 'extra':
        DSC[1] = np.concatenate((DSC[1],y))
    else:
        DSC.append(y)
if tmp_model_id in {21,20}: #remove patient 185 (predicted nothing)
    DSC[-2] = np.delete(DSC[-2],4)     
if tmp_model_id in {20}: #move noncontrast from CE to normal
    tmp = np.sort(DSC[2])
    DSC[2] = tmp[3:]
    np.concatenate((DSC[0],tmp[:3]))

#plt.close('all')
plt.figure()
#plt.title('Segmentation Model=Transformer')
plt.title('Substructure = Heart Chambers')
plt.boxplot(DSC,showmeans=True)
plt.ylim((-0.05,1))
plt.grid(axis='y')
plt.xticks(np.arange(6)+1,['Reference','AV','CE','MA','Prone','$SV_{base}$'])#'$SV_{noisy}$'])
plt.xlabel('Domains')
plt.ylabel('DSC')

for i in range(1,5):
    _,p_tmp = ranksums(DSC[i],DSC[0])
    print(p_tmp)
    
_,p = f_oneway(DSC[0],DSC[1],DSC[2],DSC[3],DSC[4],DSC[5])
    
#%%----------------local inference (transformer)--------------
layers=imp.load_source('layers',r'C:\Users\xiyaojin\Documents\GitHub\ticker_transformer\layers.py')
models=imp.load_source('models',r'C:\Users\xiyaojin\Documents\GitHub\ticker_transformer\models.py')
transformer_name='11-2-2'

transformer_path=os.path.join(r'Y:\experiment_output\ticker_transformer',transformer_name)
output_path=os.path.join(r'Y:\irb_test_results\normal\transformer',transformer_name)
# encoder=torch.load(os.path.join(transformer_path,'final_encoder.pth'),map_location=torch.device('cpu'))   
# decoder=torch.load(os.path.join(transformer_path,'final_decoder.pth'),map_location=torch.device('cpu')) 

dsc_dataset={}

for ds in datasets:
    print(ds)
    patients=get_patient_list(ds)
    DCs_all=np.zeros(len(patients))
    DCs_small=np.zeros(len(patients))
    DCs_large=np.zeros(len(patients))
    for i,p in enumerate(patients):
        print(p)
        
        #img=np.load(os.path.join(data_path[ds],f'patient{p}.npy'))
        label=np.load(os.path.join(label_path[ds],f'patient{p}.npy'))
        label=np.round(label/np.max(label)*9.0).astype(np.uint8)
        
        # img=img/255.0
        
        # img=np.expand_dims(img,axis=0)
        # img=np.expand_dims(img,axis=0)
        
        # img=torch.tensor(img,dtype=torch.float)
        
        # output=decoder(encoder(img)).detach()
        # pred=np.squeeze(np.argmax(output,axis=1)).numpy()
        
        if not os.path.exists(os.path.join(output_path,'predicted_labels')):
            os.makedirs(os.path.join(output_path,'predicted_labels'))
        
        #np.save(os.path.join(output_path,'predicted_labels',f'patient{p}.npy'),pred.astype(np.uint8))
        
        pred=np.load(os.path.join(r'Y:\irb_test_results\normal\transformer',transformer_name,'predicted_labels',f'patient{p}.npy'))

        DCs_all[i]=cal_dice(pred,label,mode='all')
        DCs_small[i]=cal_dice(pred,label,mode='small')
        DCs_large[i]=cal_dice(pred,label,mode='large')
    np.save(os.path.join(output_path,f'DCs-{ds}-ALL.npy'),DCs_all)
    np.save(os.path.join(output_path,f'DCs-{ds}-SMALL.npy'),DCs_small)
    np.save(os.path.join(output_path,f'DCs-{ds}-LARGE.npy'),DCs_large)
    dsc_dataset[ds]=DCs_all
    
dsc=[dsc_dataset[ds] for ds in datasets]
plt.figure()
plt.boxplot(dsc)

plt.xticks(np.arange(len(datasets))+1,datasets)
plt.ylim((0,1))
plt.grid(axis='y')

#%%--------------get VAE latent for labels-----------------
def _cal_dice_avg(a,b,n_classes=2):
    
    dsc = []
    for i in range(1,n_classes):
        e = 1e-6
        tmp_a=(a==i).astype(int)
        tmp_b=(b==i).astype(int)
        num = np.sum(2*tmp_a*tmp_b)+e
        den = np.sum(tmp_a)+np.sum(tmp_b)+e
        dsc.append(num/den)
    return np.mean(dsc)

def load_seg_model(model_id):
   
    model_path=os.path.join(r'y:\irb_test_results\normal',model_seg_path[model_id])
    models_seg=imp.load_source('models',os.path.join('Y:\irb_test_results','models.py'))
    
    cf_seg=imp.load_source('config',os.path.join(model_path,'config.py'))
      
    seg=models_seg.Res_U(weight_decay=cf_seg.weight_decay,
                          input_shape=(100,100,100,1),
                          batch_momentum=cf_seg.batchnorm_momentum,
                          classes=cf_seg.classes,
                          nb_filters=cf_seg.nb_filters)
    
    seg.load_weights(os.path.join(model_path,'weights.hdf5'))
    return seg


def get_vae_label_latent(model_vae,cf_vae,experiment_name,dataset,model_id):
    
    check_path1=os.path.join(r'y:\experiment_output\VAE',experiment_name,f'M_{dataset}_model{model_id}.npy')
    check_path2=os.path.join(r'y:\experiment_output\VAE',experiment_name,f'DSC_{dataset}_model{model_id}.npy')
    
    patients=get_patient_list(dataset)

    M= []
    DCs=np.zeros((len(patients)))
    for i,p in enumerate(patients):

        seg_path=os.path.join(r'y:\irb_test_results\normal',model_seg_path[model_id],'predicted_labels')
        if dataset=='para':
            #pred=np.load(os.path.join(seg_path,f"patient{p}-noisy{noise_name[noise_levels[i]]}.npy")) 
            pred=np.load(os.path.join(seg_path,f"patient{p}.npy"))
        else:
            pred=np.load(os.path.join(seg_path,f"patient{p}.npy"))      
        
        pred=tf.one_hot(pred,cf_vae.out_dim)
        pred=tf.expand_dims(pred,0)
        mean,var=model_vae.encode(pred)    
  
        M.append(mean)
        
        recon=np.squeeze(np.argmax(model_vae.decode(mean),axis=-1))
        pred=np.squeeze(np.argmax(pred,axis=-1))
        DCs[i]=_cal_dice_avg(pred,recon,n_classes = cf_vae.out_dim)
    M = np.stack(M,axis=0)
    np.save(check_path1,M)
    np.save(check_path2,DCs)
    return M, DCs

def get_vae_label_dist(experiment_name,model_id):
    
    vae_latents={}
    feature_seg_recon_error={}
    model_vae=None
    for ds in datasets:
        print(ds,end=',')
        check_path1=os.path.join(r'y:\experiment_output\VAE',experiment_name,f'M_{ds}_model{model_id}.npy')
        check_path2=os.path.join(r'y:\experiment_output\VAE',experiment_name,f'DSC_{ds}_model{model_id}.npy')   
        if os.path.exists(check_path1) and os.path.exists(check_path2):
            vae_latents[ds]=np.load(check_path1)
            feature_seg_recon_error[ds]=np.load(check_path2)
        else:
            if model_vae is None:
                model_vae,cf_vae = load_model(experiment_name, ret_cf=True)  
            vae_latents[ds],feature_seg_recon_error[ds]=get_vae_label_latent(model_vae,cf_vae,experiment_name, ds,model_id)
    
    feature_dists_cosine={}
    feature_dists_euclid={}
    feature_dists_cosine_avg={}
   
    for ds in datasets:
        ref=vae_latents['reference'] 
        
        feature_dists_cosine[ds]=cal_dist(ref,vae_latents[ds],mode='cosine')
        feature_dists_cosine_avg[ds]=cal_dist(ref,vae_latents[ds],mode='cosine_avg')
        feature_dists_euclid[ds]=cal_dist(ref,vae_latents[ds],mode='euclidean')
    return feature_dists_cosine,feature_dists_cosine_avg,feature_dists_euclid,feature_seg_recon_error

tmp_vae_label_name='3-8-1-gmvae2-3d-labelchambers-prone'
#tmp_vae_label_name='3-8-2-gmvae2-3d-wh-prone'
tmp_model_id = 20
for model_id in [0]:
    feature_seg_dists_cosine,feature_seg_dists_cosine_avg,feature_seg_dists_euclid,feature_seg_recon_error = get_vae_label_dist(tmp_vae_label_name,tmp_model_id)


#%%---------------get individual vae label recon----------------
vae_label_name='3-8-1-gmvae2-3d-labelchambers-prone'
#vae_label_name = '3-7-3-gmvae2-3d-labelWH'
model_vae,cf_vae=load_model(vae_label_name,ret_cf=True)

seg_model_id=20
patient_id=183

noise_level=0


seg_path=os.path.join(r'y:\irb_test_results\normal',model_seg_path[seg_model_id],'predicted_labels')
if noise_level>0:    
    pred=np.load(os.path.join(seg_path,f"patient{patient_id}-noisy{noise_name[noise_level]}.npy"))
else:
    pred=np.load(os.path.join(seg_path,f"patient{patient_id}.npy"))

if patient_id>135:
    img=np.load(os.path.join(r'Y:\irb_data_preprocessed\nii\images',f'patient{patient_id}.npy'))
    gt=np.load(os.path.join(r'Y:\irb_data_preprocessed\nii\labels',f'patient{patient_id}.npy'))
else:
    img=np.load(os.path.join(r'Y:\chpc2.0\data\data.3D_2mm_fixed\images',f'patient{patient_id}.npy'))
    gt=np.load(os.path.join(r'Y:\chpc2.0\data\data.3D_2mm_fixed\labels',f'patient{patient_id}.npy'))

if seg_model_id == 20:
    gt=np.load(os.path.join(r'Y:\chpc3.0\data\vae\labels_chambers_prone',f'patient{patient_id}.npy'))
if seg_model_id == 21:
    gt=np.load(os.path.join(r'Y:\chpc3.0\data\vae\labels_wh_prone',f'patient{patient_id}.npy'))


if noise_level>0:
    img=add_noise(img, noises[noise_levels[noise_level]])

    

gt=np.round(gt/np.max(gt)*(cf_vae.out_dim-1)).astype(np.uint8)

pred_tf=tf.one_hot(pred,cf_vae.out_dim)
pred_tf=tf.expand_dims(pred_tf,0)
mean,var=model_vae.encode(pred_tf) 
recon=np.squeeze(np.argmax(model_vae.decode(mean),axis=-1))

dsc_pred=_cal_dice_avg(gt,pred,cf_vae.out_dim)
dsc_recon=_cal_dice_avg(pred,recon,cf_vae.out_dim)
dsc_recon_gt=_cal_dice_avg(gt,recon,cf_vae.out_dim)


plt.figure()
plt.suptitle(f'patient:{patient_id}')
plt.subplot(2,2,1)
plt.title('CT Image')
plt.imshow(img[:,50,:],cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Ground Truth')
plt.imshow(gt[:,50,:],cmap='gray')
plt.axis('off')
plt.subplot(2,2,3)
plt.title(f'seg,DSC(seg,gt)={dsc_pred:.2f}')
plt.imshow(pred[:,50,:],cmap='gray')
plt.axis('off')
plt.subplot(2,2,4)
plt.title(f'recon,DSC(rec,seg)={dsc_recon:.2f}')
plt.imshow(recon[:,50,:],cmap='gray')
plt.xlabel(f'DSC(rec,gt)={dsc_recon_gt:.2f}')

plt.figure()
plt.suptitle(f'patient:{patient_id}')
plt.subplot(2,2,1)
plt.title('CT Image')
plt.imshow(img[:,:,50],cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Ground Truth')
plt.imshow(gt[:,:,50],cmap='gray')
plt.axis('off')
plt.subplot(2,2,3)
plt.title(f'seg,DSC(seg,gt)={dsc_pred:.2f}')
plt.imshow(pred[:,:,50],cmap='gray')
plt.axis('off')
plt.subplot(2,2,4)
plt.title(f'recon,DSC(rec,seg)={dsc_recon:.2f}')
plt.imshow(recon[:,:,50],cmap='gray')
plt.xlabel(f'DSC(rec,gt)={dsc_recon_gt:.2f}')
plt.axis('off')



#%%---------------get vae latent for images---------------------

def get_vae_latent(experiment_name,dataset,vae_type='VAE',checkpoint=''):
    #nonlocal cf
    
    check_path=os.path.join(r'y:\experiment_output',vae_type,experiment_name,f'M_{dataset}.npy')
    if os.path.exists(check_path):
        M=np.load(os.path.join(r'y:\experiment_output',vae_type,experiment_name,f'M_{dataset}.npy'))
        V=np.load(os.path.join(r'y:\experiment_output',vae_type,experiment_name,f'V_{dataset}.npy'))
        return M,V

    model = load_model(experiment_name,vae_type,checkpoint)
    patients = get_patient_list(dataset)
    
    M = []
    V = []
    for i,p in enumerate(patients):
        x=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))
        
        if dataset=='para':
            x=add_noise(x, noises[noise_levels[i]])
        x=x/255.0
        x=x.astype(np.float32)
        
        if vae_type == 'VAE':
            x = tf.expand_dims(x,0)
            x = tf.expand_dims(x,-1)
            mean,var = model.encode(x)
            mean = tf.reshape(mean,shape=-1)
            var = tf.reshape(var,shape=-1)
        else:
            x = np.expand_dims(x,axis=0)
            x = np.expand_dims(x,axis=0)
            x = torch.tensor(x/255.0,dtype=torch.float)
            if vae_type == 'VQ_VAE':
                z_e = model.encode(x)
                _, z_q, _, _, _ = model.vector_quant(z_e)
                mean = z_q.flatten().detach().numpy()  
                var = np.ones_like(mean)
            elif vae_type in {'DAE', 'contrastive_learning'}:
                mean = model.encode(x).detach().numpy()
                var = np.ones_like(mean)
        
        M.append(mean)
        V.append(var)
    M = np.stack(M,axis=0)
    V = np.stack(V,axis=0)
    np.save(os.path.join(r'y:\experiment_output',vae_type,experiment_name,f'M_{dataset}.npy'),M)
    np.save(os.path.join(r'y:\experiment_output',vae_type,experiment_name,f'V_{dataset}.npy'),V)
    return M,V

def get_vae_dist(experiment_name,ref_idx,vae_type,checkpoint=''):
      
    vae_latents={'optimum':[[],[]]}

    for ds in datasets:
        print(ds,end=',')
        M,V=get_vae_latent(experiment_name, ds, vae_type,checkpoint)
        vae_latents[ds]=[M,V]
        vae_latents['optimum'][0].append(M[ref_idx[ds],:])
        vae_latents['optimum'][1].append(V[ref_idx[ds],:])
    vae_latents['optimum'][0]=np.concatenate(vae_latents['optimum'][0],axis=0)
    vae_latents['optimum'][1]=np.concatenate(vae_latents['optimum'][1],axis=0)
    feature_dists_cosine={}
    #feature_dists_euclid={}
    feature_dists_cosine_avg={}
    feature_dists_cosine_avg_var={}
    feature_dists_cosine_knn={}
    #feature_dists_fractional={}
    for ds in datasets:
        ref=vae_latents['reference']
        #ref=[np.concatenate((vae_latents['reference'][i],vae_latents['para'][i]),axis=0) for i in range(2)]
        feature_dists_cosine[ds]=cal_dist(ref[0],vae_latents[ds][0],mode='cosine')
        feature_dists_cosine_avg[ds]=cal_dist(ref[0],vae_latents[ds][0],mode='cosine_avg')
        feature_dists_cosine_avg_var[ds]=cal_dist(ref[1],vae_latents[ds][1],mode='cosine_avg')
        feature_dists_cosine_knn[ds]=cal_dist(ref[0],vae_latents[ds][0],mode='cosine_knn')
        #feature_dists_euclid[ds]=cal_dist(ref,vae_latents[ds],mode='euclidean')
        #feature_dists_fractional[ds]=cal_dist(ref,vae_latents[ds],mode='fraction',fraction=0.5)
    features_vae_img=[feature_dists_cosine_avg,
                      feature_dists_cosine_avg_var,
                      feature_dists_cosine,
                      feature_dists_cosine_knn]
    return features_vae_img

ref_idx={}
for ds in datasets:
    if True: #ds in {'reference','para'}:
        ref_idx[ds]=np.where(labels[ds]>0.6)[0]
    else:
        ref_idx[ds]=np.array([],dtype=np.int64)
#experiment_name='11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE'
#experiment_name = '12-28-01-nemb2048-zdim8-sigmaVAEMean-adam'

experiment_name = '1-23-2-dae-zdim256'
vae_type = 'DAE'
features_vae_img=get_vae_dist(experiment_name,ref_idx,vae_type=vae_type,checkpoint='')
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
tmp_experiment_name = '11-4-2-gmvae2-3d-f256-d1024-k1-trainWithRef'
feature_img_recon_error=get_image_recon(tmp_experiment_name)

#%%-------------------explore VAE latents--------------------------
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

experiment_name = '1-22-1-dae'

vae_latents={}
for ds in datasets:
    m,v=get_vae_latent(experiment_name,ds,vae_type='DAE')
    m = np.squeeze(m)
    vae_latents[ds]=m

x=[vae_latents[ds] for ds in datasets]
x=np.concatenate(x,axis=0)

tmp_labels=load_labels('ALL',0)
y=[tmp_labels[ds] for ds in datasets]
y=np.concatenate(y)


#reducer=PCA(n_components=2)
reducer=TSNE(n_components=2)
#reducer=umap.UMAP()
x_2d=reducer.fit_transform(x)

idx_list=[]
splits=[0.0,0.2,0.4,0.6,1.0]
legend=[]
for i in range(len(splits)-1):
    lo,hi=splits[i],splits[i+1]
    idx_list.append(np.where(np.logical_and(y<hi,y>lo))[0])
    legend.append(f'{lo}<y<{hi}')

plt.figure()
for idx in idx_list:
    plt.scatter(x_2d[idx,0],x_2d[idx,1])
plt.legend(legend)
#-------------find optimal k
S=[]
for k in range(2,10):
    kmeans=KMeans(n_clusters=k).fit(x)
    
    centroids=kmeans.cluster_centers_
    labels=kmeans.labels_
    scores=[]
    for i in range(k):
        center=centroids[i,:]
        idx=labels==i
        _x=x[idx,:]
        scores.append(np.mean((_x-center)**2))
    
    S.append(np.mean(scores))
plt.figure()
plt.plot(np.arange(2,10),S)

#%%================explore latent distance===============

tmp_experiment_name ='1-23-2-dae-zdim256' #'11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE'
vae_type = 'DAE'

tmp_datasets = ['reference','av']

#----------------------------

vae_latents={}
for ds in tmp_datasets:
    m,v=get_vae_latent(tmp_experiment_name,ds,vae_type)
    m = np.squeeze(m)
    vae_latents[ds]=m  
    
x=[vae_latents[ds] for ds in tmp_datasets]
x=np.concatenate(x,axis=0)
tmp_labels=load_labels('ALL',0)
y=[tmp_labels[ds] for ds in tmp_datasets]
y=np.concatenate(y)

N = x.shape[0]

D = np.zeros((N,N))

Dice_map = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        D[i,j] = np.sum((x[i,:]-x[j,:])**2)**0.5
        Dice_map[i,j] = np.abs(y[i]-y[j])
plt.figure(figsize=(12,5))

plt.suptitle(tmp_experiment_name)
plt.subplot(1,2,1)
plt.imshow(D)
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(Dice_map)
plt.colorbar()


    


#%%-------------------heart intensity features------------------

def get_heart_intensity(dataset,i_range=(0,255)):
    patients=get_patient_list(dataset)
    H=[]
    for i,p in enumerate(patients):
        img=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))    
        if dataset in ['reference','extra']:
            label=np.load(os.path.join(r'Y:\chpc2.0\data\data.3D_2mm_fixed\labels',f"patient{p}.npy"))
        else:
            label=np.load(os.path.join(r'Y:\irb_data_preprocessed\nii\labels',f"patient{p}.npy"))
        img_heart=img[label>0]
        
        
        intensity=np.mean(img_heart)
        H.append(intensity)
    return np.stack(H)   


histograms={}
for ds in datasets:
    print(ds)
    histograms[ds]=get_heart_intensity(ds,i_range=(0,255))
    

feature_heart_intensity={}
for ds in datasets:
    ref=histograms['reference']
    ref=np.expand_dims(ref,axis=-1)
    data=np.expand_dims(histograms[ds],axis=-1)
    feature_heart_intensity[ds]=cal_dist(ref,data,mode='euclidean')

# plt.figure()
# for i,(h,y) in  enumerate(zip(histograms['ca'],labels['ca'])):
#     plt.subplot(4,5,i+1)
#     plt.title(f"{np.around(y,2)}, "+f"{np.argmax(h)}")
#     plt.bar(np.arange(255),h)
    #%%----------Figure: DSC drop on contrast images------------
drop=[]
std=[]
P=[]
for k in list(range(18))+[19]:
    labels=load_labels('ALL',k)
    y_ref=labels['reference']
    y_ca=labels['ca']
    drop.append(np.mean(y_ref)-np.mean(y_ca))
    _,p=ranksums(y_ref, y_ca)
    P.append(p)
print(P)
labels=['CNN-'+str(i) for i in range(1,19)]+['Transformer']
plt.figure(figsize=(18,5))
plt.bar(np.arange(len(drop)),drop)
plt.xticks(np.arange(len(labels)),labels,rotation=5)
plt.ylim((0,1))
plt.xlabel('Segmentation Models')
plt.ylabel('DSC change')
plt.grid(axis='y')
    
#%%-------------------get noise features----------------------------


def get_noise_level(dataset):
    noises=[0.01,1,2,4,6,8]
    
    noise_levels=[0, 3, 4, 5, 3, 4, 5, 0, 0, 0, 2, 2, 0, 4, 0, 0, 3, 5, 5, 0]
    patients=get_patient_list(dataset)
    
    check_path=os.path.join(r'y:\irb_data_features',dataset,'noise_level.npy')
    if os.path.exists(check_path):
        N=np.load(check_path)
        return N
    N=[]
    for i,p in enumerate(patients):
        img=np.load(os.path.join(data_path[dataset],f"patient{p}.npy"))  
        if dataset=='para':
            img=add_noise(img,noises[noise_levels[i]])
        img_clean=ndimage.median_filter(img,3)
        mse=np.mean((img_clean-img)**2)
        N.append(mse)
    np.save(check_path,N)
    return np.array(N)

feature_noise_level={}
for ds in datasets:
    feature_noise_level[ds]=get_noise_level(ds)
    
#%%------------------Prepare Inputs--------------------

feature_list=[features_vae_img[0],
              feature_seg_recon_error,
              feature_heart_intensity,
              #feature_noise_level,
              ]

def load_inputs(feature_list):

    inputs={}
    
    for ds in datasets:
        
        features=[f[ds] for f in feature_list]
   
        inputs[ds]=np.transpose(np.stack(features))
    return inputs

inputs=load_inputs(feature_list)
      
#%%-------------------study correaltion-----------------
tmp_model_id=20
ds_clinic=['av','prone','extra','ca','ma','para']
ds_clinic=['ca']

feature=features_vae_img[0]
feature_name = 'Shape quality estimation'
labels=load_labels(structures,tmp_model_id)

x_list=[feature['reference']]+[feature[ds] for ds in ds_clinic]
y_list=[labels['reference']]+[labels[ds] for ds in ds_clinic]

x_list = [feature['ca']]
y_list = [labels['ca']]

x_tmp,y_tmp=np.concatenate(x_list),np.concatenate(y_list)

x_reg=np.arange(np.min(x_tmp),np.max(x_tmp),(np.max(x_tmp)-np.min(x_tmp))/100).reshape((-1,1))
reg=LinearRegression().fit(x_tmp.reshape((-1,1)),y_tmp)
y_reg=reg.predict(x_reg)
y_pred=reg.predict(x_tmp.reshape((-1,1)))
score=np.round(reg.score(x_tmp.reshape((-1,1)), y_tmp),3)
err_rmse=np.round(rmse(y_pred,y_tmp),3)
err_mae=np.round(np.abs(y_pred-y_tmp).mean(),3)

r,p0=pearsonr(x_tmp,y_tmp)
r,p0=np.round(r,3),np.round(p0,5)

plt.figure()
plt.title(f"P.C={r},p={p0},\n reg score={score}, rmse={err_rmse},mae={err_mae}")
for a,b in zip(x_list,y_list):
    plt.scatter(a,b)

plt.plot(x_reg,y_reg,color='green')
plt.legend(['reference']+ds_clinic)
plt.xlabel(feature_name)#+f', model_id={tmp_model_id}')
plt.ylabel('DSC')
plt.ylim((0,1))
#plt.xlim((0,1))

        #%%------Figure: X vs Y correlation---------------


tmp_model_id=21
ds_clinic=['av','ma','ca','para','prone','extra']
#ds_clinic=['para']

feature=features_vae_img[0]
labels=load_labels(structures,tmp_model_id)
if tmp_model_id in {21}:
    feature['prone']=np.delete(feature['prone'],4)
    labels['prone']=np.delete(labels['prone'],4)
    

x_list=[feature['reference']]+[np.concatenate([feature[ds] for ds in ds_clinic],axis=0)]
y_list=[labels['reference']]+[np.concatenate([labels[ds] for ds in ds_clinic],axis=0)]

x_tmp,y_tmp=np.concatenate(x_list),np.concatenate(y_list)

x_reg=np.arange(np.min(x_tmp),np.max(x_tmp),(np.max(x_tmp)-np.min(x_tmp))/100).reshape((-1,1))
reg=LinearRegression().fit(x_tmp.reshape((-1,1)),y_tmp)
y_reg=reg.predict(x_reg)
y_pred=reg.predict(x_tmp.reshape((-1,1)))
score=np.round(reg.score(x_tmp.reshape((-1,1)), y_tmp),3)
err_rmse=np.round(rmse(y_pred,y_tmp),3)
err_mae=np.round(np.abs(y_pred-y_tmp).mean(),3)

r,p0=pearsonr(x_tmp,y_tmp)
r,p0=np.round(r,3),p0

plt.figure()
name={19:'Transformer',0:'CNN-1'}
struct = {20:'Cardiac Chamebrs', 21:'Heart'}
#plt.title(f'Segmentation Model = {name[tmp_model_id]}')
plt.title(f'Structure = {struct[tmp_model_id]}')
print(f"P.C={r},p={p0},\n reg score={score}, rmse={err_rmse},mae={err_mae}")
for a,b in zip(x_list,y_list):
    plt.scatter(a,b)

plt.plot(x_reg,y_reg,color='green',linestyle='dashed')
plt.legend(['Domain=Common','Domain=Uncommon','Linear fit'])
plt.xlabel('$X_{Appr}$')
plt.ylabel('DSC')
plt.ylim((0,1))
#plt.xlim((0,1))

#%%--------------Dissertation FIgure, 6 panels

ds_lists = [
    ['av','ma','ca','para','prone','extra'],
    ['ca'],
    ['av','ma','ca','para','prone','extra']
    ]
legends = [
    ['Domain=Common','Domain=Uncommon','Linear fit'],
    ['Domain=Common','Domin=CE','Linear fit'],
    ['Domain=Common','Domain=Uncommon','Linear fit']
    ]
feature_candidates = [features_vae_img[0],
                      feature_heart_intensity,
                    
                      feature_seg_recon_error]
xlabels = ['$X_{Appr}$', '$X_{Intensity}$','$X_{Shape}$']
model_ids = [21,20]
plt.figure(figsize=(16,8))
tmp_vae_label_name = {20:'3-8-1-gmvae2-3d-labelchambers-prone',
                      21:'3-8-2-gmvae2-3d-wh-prone'}
for i in range(6):
    ds_clinic = ds_lists[i%3]
    tmp_model_id=model_ids[i//3]


    feature=feature_candidates[i%3].copy()
    if i%3==2:
        feature = get_vae_label_dist(tmp_vae_label_name[tmp_model_id],tmp_model_id)[3]
    labels=load_labels(structures,tmp_model_id)

    if tmp_model_id in {21} and i%3==0:
        feature['prone']=np.delete(feature['prone'],4)
        labels['prone']=np.delete(labels['prone'],4)
        

    x_list=[feature['reference']]+[np.concatenate([feature[ds] for ds in ds_clinic],axis=0)]
    y_list=[labels['reference']]+[np.concatenate([labels[ds] for ds in ds_clinic],axis=0)]
    

    x_tmp,y_tmp=np.concatenate(x_list),np.concatenate(y_list)

    x_reg=np.arange(np.min(x_tmp),np.max(x_tmp),(np.max(x_tmp)-np.min(x_tmp))/100).reshape((-1,1))
    reg=LinearRegression().fit(x_tmp.reshape((-1,1)),y_tmp)
    y_reg=reg.predict(x_reg)
    y_pred=reg.predict(x_tmp.reshape((-1,1)))
    score=np.round(reg.score(x_tmp.reshape((-1,1)), y_tmp),3)


    r,p0=pearsonr(x_tmp,y_tmp)
    r,p0=np.round(r,3),p0

    
    name={19:'Transformer',0:'CNN-1'}
    #plt.title(f'Segmentation Model = {name[tmp_model_id]}')
    plt.subplot(2,3,i+1)
    for a,b in zip(x_list,y_list):
        plt.scatter(a,b)


    plt.plot(x_reg,y_reg,color='green',linestyle='dashed')
    plt.legend(legends[i%3])

    plt.xlabel(xlabels[i%3])
    plt.ylabel('DSC')
    plt.ylim((0,1))
    if i%3==2:
        plt.xlim((0,1))



#%%--------------Figure: X vs Y, 8 panels------------------

ds_lists = [
    ['av','ma','para','prone','extra'],
    ['ca'],
    ['para'],
    ['av','ma','ca','para','prone','extra']
    ]
legends = [
    ['Domain=Common','Domain=Uncommon','Linear fit'],
    ['Domain=Common','Domin=CE','Linear fit'],
    ['Domain=Common','Domin=$SV_{noisy}$','Linear fit'],
    ['Domain=Common','Domain=Uncommon','Linear fit']
    ]
feature_candidates = [features_vae_img[0],
                      feature_heart_intensity,
                      feature_noise_level,
                      feature_seg_recon_error]
xlabels = ['$X_{Appr}$', '$X_{Intensity}$', '$X_{Noise}$', '$X_{Shape}$']
model_ids = [0,19]
plt.figure(figsize=(20,8))
for i in range(8):
    ds_clinic = ds_lists[i%4]
    tmp_model_id=model_ids[i//4]


    feature=feature_candidates[i%4]
    if i%4==3:
        feature = get_vae_label_dist(vae_label_name,tmp_model_id)[3]
    labels=load_labels(structures,tmp_model_id)

    x_list=[feature['reference']]+[np.concatenate([feature[ds] for ds in ds_clinic],axis=0)]
    y_list=[labels['reference']]+[np.concatenate([labels[ds] for ds in ds_clinic],axis=0)]
    
    x_ca = feature['ca']
    y_ca = labels['ca']

    x_tmp,y_tmp=np.concatenate(x_list),np.concatenate(y_list)

    x_reg=np.arange(np.min(x_tmp),np.max(x_tmp),(np.max(x_tmp)-np.min(x_tmp))/100).reshape((-1,1))
    reg=LinearRegression().fit(x_tmp.reshape((-1,1)),y_tmp)
    y_reg=reg.predict(x_reg)
    y_pred=reg.predict(x_tmp.reshape((-1,1)))
    score=np.round(reg.score(x_tmp.reshape((-1,1)), y_tmp),3)


    r,p0=pearsonr(x_tmp,y_tmp)
    r,p0=np.round(r,3),p0

    
    name={19:'Transformer',0:'CNN-1'}
    #plt.title(f'Segmentation Model = {name[tmp_model_id]}')
    plt.subplot(2,4,i+1)
    for a,b in zip(x_list,y_list):
        plt.scatter(a,b)


    plt.plot(x_reg,y_reg,color='green',linestyle='dashed')
    plt.legend(legends[i%4])
    if i%4==0:
        plt.scatter(x_ca,y_ca,c='C1',marker='x')
    plt.xlabel(xlabels[i%4])
    plt.ylabel('DSC')
    plt.ylim((0,1))
    if i%4==3:
        plt.xlim((0,1))

#%%----------------performing regression-----------------------
TEST_score=[]
TEST_mae=[]


kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3))  \
    + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-2, 1e1))

train_ratio=0.9
n_bags=50
#experiment_name='11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE' #'11-4-1-gmvae2-3d-f256-d1024-k1'
#experiment_name = '12-28-01-nemb2048-zdim8-sigmaVAEMean-adam'
#experiment_name = '1-22-1-dae'

experiments = [
               #['11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE' ,'VAE'],
               #['11-13-1-gmvae2-3d-f256-d1024-k8-heldoutTest','VAE'],
               #['12-28-01-nemb2048-zdim8-sigmaVAEMean-adam','VQ_VAE'],
               #['1-22-1-dae','DAE'],
               #['1-23-1-dae-fixed','DAE'],
               ['1-23-2-dae-zdim256','DAE'],
               #['1-25-1-dvae','VAE']
    ]

#vae_label_name='9-25-2-gmvae2-3d-label-f128-d256-k2-klMean'
vae_label_name ='3-8-2-gmvae2-3d-wh-prone'
#vae_label_name = '3-8-1-gmvae2-3d-labelchambers-prone'
#model_ids=list(range(18))+[19]
model_ids=[21]

labels=load_labels('ALL',-1)

tmp_datasets=['reference', 'av', 'ca', 'ma', 'para', 'prone','extra']
#tmp_datasets=['ca']

y=[labels[ds] for ds in tmp_datasets]
p=[[(line,ds) for line in get_patient_list(ds)] for ds in tmp_datasets]

y=np.concatenate(y,axis=0)
p=np.concatenate(p,axis=0)
idx=np.argsort(y)

regressors=[
            LinearRegression(),
            #BaggingRegressor(LinearRegression(),n_bags,max_samples=train_ratio,oob_score=True),
            #GaussianProcessRegressor(kernel=kernel,normalize_y=True),
            #RandomForestRegressor(),
            #GradientBoostingRegressor()
    ]

idx_test = list(idx[np.arange(0,len(idx),3)])#+[139]# include patient 185 into the test set
idx_test_set = set(idx_test)
#idx_train2= idx[list(set(np.arange(len(idx)))-set(idx_test))] # incorrect!

idx_train = []
for i in idx:
    if i not in idx_test_set and i!=139: ##exclude patietn 185!!
        idx_train.append(i)

regressor = regressors[0]
#experiment_name,vae_type = experiments[-1]
for experiment_name,vae_type in experiments:
#for regressor in regressors:
    
    
    #model,cf=load_vae_model(experiment_name)  
    train_scores,valid_scores,train_maes,valid_maes,valid_maes_std=[],[],[],[],[]
    feature_weights=[]
    accuracy=[]
    for model_id in model_ids:
        print(f'{model_id}',end=',')
        
        
        features_vae_img=get_vae_dist(experiment_name,ref_idx,vae_type)
        feature_seg_dists_cosine,feature_seg_dists_cosine_avg,feature_seg_dists_euclid,feature_seg_recon_error = get_vae_label_dist(vae_label_name,model_id)
        feature_list=[features_vae_img[0],
                      feature_seg_recon_error,
                      feature_heart_intensity,  
                      #feature_noise_level
                      ]
        inputs=load_inputs(feature_list) 
        x=[inputs[ds] for ds in tmp_datasets]
        x=np.concatenate(x,axis=0)
    
        labels=load_labels('ALL',model_id)
        y=[labels[ds] for ds in tmp_datasets]
        y=np.concatenate(y,axis=0)
        
            
        x_train,y_train=x[idx_train,:],y[idx_train]
        x_test,y_test=x[idx_test,:],y[idx_test]
        p_train,p_test=p[idx_train,:],p[idx_test,:]
         
        poly = PolynomialFeatures(degree = 2)
        #x_train=poly.fit_transform(x_train)
        #x_test=poly.fit_transform(x_test)
        

        reg=regressor.fit(x_train,y_train)
        
        #feature_weights.append(reg.coef_)
        
        #y_pred_oob=reg.oob_prediction_
        y_pred_train=reg.predict(x_train)
        y_pred_test=reg.predict(x_test)
        
        
        true_neg=np.logical_and(y_test>0.4,y_pred_test>0.4)
        true_pos=np.logical_and(y_test<0.4,y_pred_test<0.4)
        acc=np.logical_or(true_neg,true_pos)
        accuracy.append(np.sum(acc)/len(acc))
        
        train_score=np.round(reg.score(x_train,y_train),2)
        #valid_score=np.round(reg.oob_score_,2)
        test_score=np.round(reg.score(x_test,y_test),2)
        
        train_rmse=np.round(rmse(y_train,y_pred_train),3)
        test_rmse=np.round(rmse(y_test,y_pred_test),3)
        train_mae=np.round(np.abs(y_train-y_pred_train).mean(),3)
        test_mae=np.round(np.abs(y_test-y_pred_test).mean(),3)
        test_mae_std=np.round(np.std(np.abs(y_test-y_pred_test)),3)
        
    
        
        train_maes.append(train_mae)
        valid_maes.append(test_mae)
        valid_maes_std.append(test_mae_std)
        
        train_scores.append(train_score)
        valid_scores.append(test_score)
    print('\n')
    print('mean mae:',np.mean(valid_maes))
    print('std mae:',np.std(valid_maes))
    
    
    TEST_score.append(valid_scores)
    TEST_mae.append(valid_maes)
    
    
    
    # for i,(train_metric,valid_metric) in enumerate(zip([train_scores,train_maes],[valid_scores,valid_maes])):
    #     tr_mu,tr_std=np.mean(train_metric),np.std(train_metric)
    #     te_mu,te_std=np.mean(valid_metric),np.std(valid_metric)
    #     plt.figure(figsize=(12,4))
    #     plt.title(f'train:{tr_mu:.3f}'+u"\u00B1"+f'{tr_std:.3f}'+f', test:{te_mu:.3f}'+u"\u00B1"+f'{te_std:.3f}')
    #     plt.plot(np.arange(len(model_ids)),train_metric)
    #     plt.plot(np.arange(len(model_ids)),valid_metric)
    #     plt.ylim((0,1))
    #     if i==1:
    #         plt.ylim((0,0.1))
    #     plt.legend(['training','test'])
    #     idx2model={i:m for m,i in model2idx.items()}
    #     plt.xticks(np.arange(len(model_ids)),[idx2model[i] for i in model_ids])
    #     plt.xlabel('seg-models')
    #     plt.grid(axis='y')
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.boxplot([train_scores,valid_scores],showmeans=True)
    # plt.ylim((0,1))
    # plt.xticks([1,2],[f'Train\n{np.mean(train_scores):.3f}'+u"\u00B1"+f'{np.std(train_scores):.3f}',
    #                   f'Test\n{np.mean(valid_scores):.3f}'+u"\u00B1"+f'{np.std(valid_scores):.3f}'])
    # plt.title('R^2 score')
    # plt.grid(axis='y')

    # plt.subplot(1,2,2)
    # plt.boxplot([train_maes,valid_maes],showmeans=True)
    # plt.ylim((0,0.1))
    # plt.xticks([1,2],[f'Train\n{np.mean(train_maes):.3f}'+u"\u00B1"+f'{np.std(train_maes):.3f}',
    #                   f'Test\n{np.mean(valid_maes):.3f}'+u"\u00B1"+f'{np.std(valid_maes):.3f}'])
    # plt.title('MAE')
    # plt.grid(axis='y')
    #%%------------Figure: feature importances--------------
plt.figure()
plt.imshow((np.array(feature_weights)))
plt.colorbar()
   #%%------------Figure: comparison of VAEs-------------
plt.figure()
plt.boxplot(TEST_mae,showmeans=True)
plt.ylabel('MAE')
xtick = [x[0] for x in experiments]
#xtick = ['VAE','GM-VAE','VQ-VAE','DAE']
plt.xticks(np.arange(len(xtick))+1,xtick)
plt.grid(axis='y')
   #%%------------Figure: comparison of regression models-------------
plt.figure()
plt.boxplot(TEST_mae,showmeans=True)
plt.ylabel('MAE')
plt.xticks(np.arange(5)+1,['Linear', 'Bagging','GPR','RF','GBR'])
plt.grid(axis='y')
plt.ylim((0,0.1))
    
#plt.figure()
#plt.boxplot(TEST_score)

    #%%------------Figure Final results-----------------
    # xlabels=['Transformer','CNN-size1','CNN-size2','CNN-size3','CNN-size4','CNN-size5',
    #          'CNN-loss_CE','CNN-loss_Dice','CNN-loss_DiceCE','CNN-loss_MD','CNN-loss_MSE','CNN-loss_wCE','CNN-loss_wDice',
    #          'CNN-aug00','CNN-aug20','CNN-aug40','CNN-aug60','CNN-aug80','CNN-aug100']
    # xlabels=['Transformer','Size1','Size2','Size3','Size4','Size5',
    #          'Loss_CE','Loss_Dice','Loss_DiceCE','Loss_MD','Loss_MSE','Loss_wCE','Loss_wDice',
    #          'Aug00','Aug20','Aug40','Aug60','Aug80','Aug100']
xlabels=['CNN-'+str(i) for i in range(1,19)]+['Transformer']
plt.figure(figsize=(18,6))
#plt.scatter(np.arange(len(xlabels))+1,valid_maes)
plt.errorbar(np.arange(len(xlabels))+1,valid_maes,yerr=valid_maes_std,ecolor='lightgray',fmt='s',capsize = 4)
plt.xticks(np.arange(len(xlabels))+1,xlabels,rotation=8)
plt.ylabel('MAE')
plt.xlabel('Segmentation models')
    #%%---------Figure: results on specific seg models---------------
plt.figure()
plt.scatter(y_test,y_pred_test)
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),linestyle='dashed',color='green')
plt.xlabel('True DSC')
plt.ylabel('Predicted DSC')
plt.title('substructure  = Heart')
plt.xlim((0.7,1))
plt.ylim((0.7,1))
    #%%---------Figure: compare QA methods

maes_cnn = [0.088,0.105,0.102,0.093,0.089,
           0.079,0.069,0.111,0.084,0.096,
           0.102,0.094,0.090,0.077,0.120,
           0.113,0.146,0.087,0.092]

maes_cnn = np.array(maes_cnn)

#maes_ours = np.load(r'y:irb_test_results\maes_ours.npy')
maes_ours = TEST_mae[3]
maes_liu = np.load(r'y:irb_test_results\maes_liu.npy')

plt.figure()
plt.boxplot([maes_ours,maes_liu,maes_cnn],showmeans=True)
plt.ylabel('MAE')
plt.xticks(np.arange(3)+1,['Proposed', 'Liu et.al.', 'CNN Reg'],style='italic')
plt.grid(axis='y')
plt.ylim((0.03,0.125))
plt.xlabel('QA methods')
    
#%%
plt.figure()
plt.subplot(2,1,1)
plt.title(f'training, score={train_score},rmse={train_rmse},mae={train_mae}')
id_sort=np.argsort(y_train)
plt.scatter(np.arange(len(y_train)),y_train[id_sort])
plt.scatter(np.arange(len(y_pred_train)),y_pred_train[id_sort])
plt.ylim((0,1))
plt.ylabel('DSC')
plt.legend(['True DSC','Predicted DSC'])

id_sort=np.argsort(y_test)
plt.subplot(2,1,2)
plt.title(f'test, score={test_score},rmse={test_rmse},mae={test_mae}')
plt.scatter(np.arange(len(y_test)),y_test[id_sort])
plt.scatter(np.arange(len(y_pred_test)),y_pred_test[id_sort])
plt.legend(['True DSC','Predicted DSC'])
plt.ylim((0,1))
plt.ylabel('DSC')
plt.suptitle(f'model_id={model_id}')

# plt.figure()
# plt.scatter(y_test,y_pred_test)

# plt.subplot(1,2,2)
# plt.title(f'valid, score={valid_score}')
# plt.scatter(np.arange(len(y_train)),y_train[id_sort])
# plt.scatter(np.arange(len(y_pred_oob)),y_pred_oob[id_sort])
# plt.legend(['True DSC','Predicted DSC'])
# plt.ylim((0,1))
# plt.suptitle(f'model_id={model_id}')

#------------------visualization gaussian-----------------------
# pred_tr_m,pred_tr_std=reg.predict(x_train,return_std=True)
# pred_t_m,pred_t_std=reg.predict(x_test,return_std=True)
# xdata=np.arange(len(y_train))
# order=np.argsort(y_train)
# fig=plt.figure()
# ax=fig.add_subplot(2,1,1)
# plt.title('score_train:'+str(train_score))
# plt.plot(xdata,y_train[order])
# plt.scatter(xdata,pred_tr_m[order],color='#ff7f0e')
# plt.ylim((0,1))
# ax.fill_between( 
#         xdata, (pred_tr_m - pred_tr_std*1.96)[order],(pred_tr_m + pred_tr_std*1.96)[order], color="pink", alpha=0.5, label="predict std")

# xdata=np.arange(len(y_test))
# order=np.argsort(y_test)
# ax=fig.add_subplot(2,1,2)
# plt.title('score_test:'+str(test_score))
# plt.plot(xdata,y_test[order])
# plt.scatter(xdata,pred_t_m[order],color='#ff7f0e')
# plt.ylim((0,1))

# ax.fill_between( 
#         xdata, (pred_t_m - pred_t_std*1.96)[order],(pred_t_m + pred_t_std*1.96)[order], color="pink", alpha=0.5, label="predict std")
#------------------visualization gaussian 2 (diagnal)-----------------------
# pred_tr_m,pred_tr_std=reg.predict(x_train,return_std=True)
# pred_t_m,pred_t_std=reg.predict(x_test,return_std=True)
# xdata=np.arange(len(y_train))
# order=np.argsort(y_train)
# fig=plt.figure(figsize=(10,4))
# ax=fig.add_subplot(1,2,1)
# plt.title('score_train:'+str(train_score))
# #plt.plot(xdata,y_train[order])
# plt.scatter(y_train[order],pred_tr_m[order],color='#ff7f0e')
# plt.xlim((0,1))
# plt.ylim((0,1))
# ax.fill_between( 
#         y_train[order], (pred_tr_m - pred_tr_std*1.96)[order],(pred_tr_m + pred_tr_std*1.96)[order], color="pink", alpha=0.5, label="predict std")

# xdata=np.arange(len(y_test))
# order=np.argsort(y_test)
# ax=fig.add_subplot(1,2,2)
# plt.title('score_test:'+str(test_score))
# #plt.plot(xdata,y_test[order])
# plt.scatter(y_test[order],pred_t_m[order],color='#ff7f0e')
# plt.xlim((0,1))
# plt.ylim((0,1))

# ax.fill_between( 
#         y_test[order], (pred_t_m - pred_t_std*1.96)[order],(pred_t_m + pred_t_std*1.96)[order], color="pink", alpha=0.5, label="predict std")
#----------------------------


# plt.figure()
# plt.subplot(1,2,1)
# plt.title('train')
# plt.boxplot(TRAIN)
# plt.ylim((0,1))
# plt.grid(axis='y')
# plt.subplot(1,2,2)
# plt.title('valid')
# plt.boxplot(TEST)
# plt.ylim((0,1))
# plt.grid(axis='y')

#%%--------------------statistical test----------------

clinic_ds='ca'
ref_idx=[i for i,x in enumerate(p_test) if x[1]=='reference']
clinic_idx=[i for i,x in enumerate(p_test) if x[1]==clinic_ds]

y_test_good=y_test[ref_idx] 
y_test_bad=y_test[clinic_idx]
y_pred_test_good=y_pred_test[ref_idx]
y_pred_test_bad=y_pred_test[clinic_idx]

p_true=np.round(ranksums(y_test_bad, y_test_good)[1],5)
p_pred=np.round(ranksums(y_pred_test_bad,y_pred_test_good)[1],5)
    


plt.figure()
plt.subplot(1,2,1)
plt.title(f'True DSC, p={p_true}')
plt.boxplot([y_test_good,y_test_bad])
plt.xticks(np.arange(2)+1,['ref',f'"clinic":{clinic_ds}'])
plt.ylim((0,1))
plt.subplot(1,2,2)
plt.title(f'Predicted DSC, p={p_pred}')
plt.boxplot([y_pred_test_good,y_pred_test_bad])
plt.xticks(np.arange(2)+1,['ref',f'"clinic":{clinic_ds}'])
plt.ylim((0,1))

#%%-------------------clustering on VAE latent vectors-------------------------
from sklearn.cluster import KMeans
name='11-16-2-gmvae2-3d-f256-d256-k1-heldoutTest' 
M,V=get_vae_latent(name, 'reference')

n_clusters=4

kmeans=KMeans(n_clusters=n_clusters).fit(M)

clabels=kmeans.labels_
y_ref=[]
for i in range(n_clusters):
    y_ref.append(labels['reference'][clabels==i])
    

plt.figure()
plt.boxplot(y_ref,showmeans=True)

#%%----------calculate distances for reference dataset------------------
labels=load_labels('ALL', model_id=19)
idx=np.argsort(labels['reference'])[::-1]

embedding=M[idx,:]

n=embedding.shape[0]
dist_map=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        dist_map[i,j]=cal_dist(embedding[i:i+1,:],embedding[j:j+1,:],mode='cosine')

plt.figure()
plt.imshow(dist_map)
#%%---------------temp----------------

train_all=open(r'y:train_all.txt').readlines()
test={i for i,_ in p[idx_test]}
newlines=[]
for line in train_all:
    if line[:-2] not in test:
        newlines.append(line)
    
newfile=open(r'y:train_all_noTest.txt','w')
newfile.writelines(newlines)
newfile.close()
#%%------------------study image VAE latent features----------------
#name='11-19-1-gmvae2-3d-f256-d256-k1-heldoutTest-VanillaVAE' 
name = '1-23-2-dae-zdim256'
vae_type='DAE'
features_vae_img=get_vae_dist(name,ref_idx,vae_type=vae_type)

latents={ds:get_vae_latent(name,ds,vae_type) for ds in datasets+['para-base']}

tmp_ds=['reference','para-base','prone']
rep=30
img=[]
for ds in tmp_ds:
    img.append(np.repeat(latents[ds][0].mean(axis=0,keepdims=True),rep,axis=0))


img = np.squeeze(img)

plt.figure()
plt.imshow(np.concatenate(img,axis=0),cmap='bwr')
plt.yticks([])
plt.xlabel('Latent dimension')
plt.colorbar()

#%%------------------generate & save y_true-----------------

for tmp_mdl_id in list(range(18))+[19]:

    model_path=os.path.join(r'y:\irb_test_results\normal',model_seg_path[tmp_mdl_id])
    labels = load_labels('ALL',tmp_mdl_id)
    
    y_true_dict = {}
    
    for ds in datasets:
        patients = get_patient_list(ds)
        
        
        
        for i,patient in enumerate(patients):
            if ds !='para':
                name = f'patient{patient}'
            else:
                noise = noise_name[noise_levels[i]]
                name = f'patient{patient}-noisy{noise}'
    
            y_true_dict[name] = labels[ds][i]
    
    y_true_json = json.dumps(y_true_dict)
    f = open(os.path.join(model_path,'y_true_dict.json'),'w')
    f.write(y_true_json)
    f.close()

#load json
# f = open(os.path.join(model_path,'y_true_dict.json'))
# y_true_dict = json.load(f)



#%%----tmp-----
for tmp_mdl_id in list(range(18))+[19]:
    print(f'{tmp_mdl_id}',end=',')
    path = os.path.join(r'Y:\irb_test_results\normal',model_seg_path[tmp_mdl_id],'predicted_labels')
    files = os.listdir(path)
    
    for f in files:
        #print(f)
        img = np.load(os.path.join(path,f))
        img = img.astype(np.uint8)
        np.save(os.path.join(path,f),img)

#%%

p = []

for ds in datasets:
    patients = get_patient_list(ds)
    
    for i,patient in enumerate(patients):
        if ds !='para':
            name = f'patient{patient}'
        else:
            noise = noise_name[noise_levels[i]]
            name = f'patient{patient}-noisy{noise}'
        p.append(name+'\n')

p = np.array(p)

p_train_valid = list(p[idx_train])

idx_valid = list(np.array(idx_train)[np.arange(0,len(idx_train),4)])
idx_train_true =list(set(idx_train)-set(idx_valid)) 



p_train = list(p[idx_train_true])
p_valid = list(p[idx_valid])
p_test = list(p[idx_test])

p_train_all = list(p[idx_train])
f = open(r'y:train_all.txt','w')
f.writelines(p_train_all)
f.close()

f_train = open(r'y:train.txt','w')
f_train.writelines(p_train)
f_train.close()

f_valid = open(r'y:valid.txt','w')
f_valid.writelines(p_valid)
f_valid.close()

f_test = open(r'y:test.txt','w')
f_test.writelines(p_test)
f_test.close()

    
    