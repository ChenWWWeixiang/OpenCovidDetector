import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt
def inter_vecter(v):
    length=v.shape[0]
    x=np.linspace(0, 1, 40)
    xp = np.linspace(0, 1, length)
    new_v=np.interp(x, xp, v)
    return new_v

datas=open('val_slices_count.txt','r').readlines()
full_names=[da.split('\t')[0].split('/')[-1] for da in datas]
person_names=[da.split('_')[0] for da in full_names]
pres=[np.array(da.split('\t')[1].split(','),np.float) for da in datas]
abnormal_count=[da.split('\t')[2] for da in datas]
slice_id=[np.array(da.split('\t')[3].split(','),np.int) for da in datas]


present_names=list(set(person_names))
person_names=np.array(person_names)
full_names=np.array(full_names)
pres=np.array(pres)
a='figs_re/'
b='npys_re/'
os.makedirs(a,exist_ok=True)
os.makedirs(b,exist_ok=True)
for a_name in present_names:
    this_names=full_names[person_names==a_name]
    this_pred = pres[person_names == a_name]
    dates=[da.split('_')[-1][:-4] for da in this_names]
    idx=np.argsort(dates)## sorted idx
    this_names=this_names[idx]
    this_pred=this_pred[idx]
    this_pred=np.stack([inter_vecter(da) for da in this_pred])
    plt.figure(figsize =(4,8))
    sb.heatmap(this_pred.transpose(),vmin=0,vmax=1,annot=True, fmt=".3f",cmap='jet',xticklabels=this_names)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.29, top=0.94)
    plt.savefig(a+this_names[0].split('_')[0]+'.jpg')
    np.save(b+this_names[0].split('_')[0]+'.npy',np.concatenate([this_names[:,np.newaxis],this_pred],1))

