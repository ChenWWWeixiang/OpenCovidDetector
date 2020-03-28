import pandas as pd
import csv
import sklearn.cluster as cluster
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

name=['original_glszm_GrayLevelVariance',
'log-sigma-1-0-mm-3D_firstorder_Minimum',
'log-sigma-3-0-mm-3D_firstorder_Median',
'log-sigma-3-0-mm-3D_glszm_HighGrayLevelZoneEmphasis',
'log-sigma-3-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis',
'log-sigma-3-0-mm-3D_ngtdm_Complexity',
'log-sigma-5-0-mm-3D_glrlm_HighGrayLevelRunEmphasis',
'wavelet-LH_firstorder_90Percentile',
'wavelet-HL_firstorder_10Percentile',
'wavelet-HH_glcm_Autocorrelation',
'wavelet-LL_glrlm_GrayLevelNonUniformity',
'wavelet-LL_glszm_GrayLevelNonUniformity',
'wavelet-LL_gldm_SmallDependenceHighGrayLevelEmphasis',
'distance',
'hdf','hdf3d'
]
cls_name=['co','GGO','mix','paving']
with open('features.csv','r') as f:
    frac=open('HFD.txt','r')
    df_frac2=pd.read_csv(frac)
    df_frac3 = pd.read_csv(open('HFD3D.txt','r'))
    df_pos=pd.read_csv(open('distance.txt','r'))
    df = pd.read_csv(f)
    df=pd.merge(df,df_pos,on='id')
    df = pd.merge(df, df_frac2, on='id')
    df = pd.merge(df, df_frac3, on='id')
    #idname=df.pop('id')
    #df.pop('label')
    df=df[['id','label']+name]
    df.to_csv('only_left.csv')
    df=(df-df.mean())/df.std()
    print(df.keys())
    predict_labels=np.array([cls_name.index(item.split('_')[1]) for item in np.array(idname)])
    centers=[]
    for acluster in range(4):
        idx=np.where(predict_labels==acluster)
        clu_df=df.iloc[idx]
        clu_center=clu_df.mean()
        centers.append(clu_center)
    centers=np.array(centers)
    plt.style.use('ggplot')
    plt.figure(figsize=(10,10))
    #plt.plot(centers.transpose())
    x = np.linspace(1, 17, 16)-0.4
    plt.bar(x,centers[0,:],width=0.2, alpha=0.5)
    plt.bar(x+0.2, centers[1,:],width=0.2, alpha=0.5)
    plt.bar(x+0.4, centers[2,:],width=0.2, alpha=0.5)
    plt.bar(x+0.6, centers[3,:],width=0.2, alpha=0.5)
    #plt.xticks(x + bar_width / 2, tick_label)
    plt.xticks(x + 0.3, list(name), rotation=90,fontsize = 10)
    plt.subplots_adjust(left=0.05, wspace=0.2, hspace=0.2,
                        bottom=0.49, top=0.94)
    plt.title('Average Values of Selected Features on Manually Defined Sub-Classes')
    plt.legend(['Consolidation','Ground Glass Opacity','Mixture of Co. and GGO.','Crazy Paving'])

    plt.savefig('f_c.jpg')
    plt.show()