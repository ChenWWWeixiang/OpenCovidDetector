import os,random,datetime
import numpy as np
import sys
import seaborn as sb
import matplotlib.pyplot as plt
from numpy import linalg
import scipy
import SimpleITK as sitk
def inter_vecter(v):
    length=v.shape[0]
    x=np.linspace(0, 1, 100)
    xp = np.linspace(0, 1, length)
    new_v=np.interp(x, xp, v)
    return new_v
def get_feature_day(path):
    Data=[]
    Date=[]
    for file in os.listdir(path):
        data=sitk.ReadImage(os.path.join(path,file))
        data=sitk.GetArrayFromImage(data)
        data=data.mean(1).mean(1)
        Data.append(data)
        Date.append(file.split('.mha')[0])
    idx = np.argsort(Date)
    Date = np.array(Date)[idx]
    Date = [datetime.datetime.strptime('2020-' + date, '%Y-%m-%d') for date in Date]
    delta = np.array([(Date[i] - Date[0]).days for i in range(len(Data))])
    Data = np.array(Data)
    Data = Data[idx]
    this_pred = np.stack([inter_vecter(da) for da in Data])
    return this_pred,delta
def distance(gallary,query_data,query_delta,need_length=None):
    g=gallary[:,query_delta]
    q=query_data
    if need_length:
        if gallary.shape[1]<=need_length:
            return 1
        #if gallary[0,need_length]==-1:
        #    return 1
    cosV12 = 1-np.abs(np.dot(g.T, q.T) / (linalg.norm(g) * linalg.norm(q)+1e-5))
    #cosV12 = np.abs(g.T-q)
    return np.diagonal(cosV12).mean()
def similar_score(g,q,q_d,need_length=None):
    if need_length:
        if g[0,need_length]==-1:
            return 1,0
    al=np.where(g[0,:]==-1)
    al=np.min(al[0])
    S=[]
    d=[]
    for i in range(al-np.max(q_d)):
        S.append(distance(g[:,i:], q, q_d,need_length))
        d.append(i)
    return np.min(S),np.argmin(S)
l='x'
#inpath_train='/mnt/data9/mp_NCPs/reg_pt/gallary'
inpath_train='/mnt/data9/mp_NCPs/reg_pt/gallary'
inpath_query='/mnt/data9/mp_NCPs/reg_pt/query'
img_path='/mnt/data9/mp_NCPs/reg/images'
lesion_path='/mnt/data9/mp_NCPs/reg/lesions'
all_files=os.listdir(inpath_train)
all_files=[a for a in all_files if a.split('.')[-1]=='npy']
random.seed(2020)
random.shuffle(all_files)
query_file=os.listdir(inpath_query)
ll=[714]
klist=[1,3,5,10,20,30,40,-1]#,7,10,20,40,80,-1]
LLLL=[]
for k in klist:
    LLL = []
    for l in ll:
        gallary=all_files#[:l]
        query=query_file
        Loss=[]
        for item in query:
            name=os.path.join(lesion_path,item.split('.npy')[0])
            data,delta=get_feature_day(name)
            x=data[:-1,:]
            d_x=delta[:-1]
            y=data[-1,:]
            d_y=delta[-1]
            D=[]
            M=[]
            for to_find in gallary:
                dis=distance(np.load(os.path.join(inpath_train,to_find)),x,d_x,d_y)
                #dis,mv=similar_score(np.load(os.path.join(inpath_train,to_find)), x, d_x,d_y)
               # dis=1-sc
                #dis=1-dis
                #M.append(mv)
                D.append(dis)
            D=np.array(D)
            idx=np.argsort(D)
            G=np.array(gallary)[idx[:k]]
            #M=np.array(M)
            #M=M[idx[:k]]
            #if d_y>44:
            #    d_y=44
            k_closest = [np.load(os.path.join(inpath_train, data))[:, d_y] for data in G]
            #k_closest=[np.load(os.path.join(inpath_train,data))[:,d_y+m] for data,m in zip(G,M)]
            k_closest=[k*(k[0]>-1) for k in k_closest]
            if len(k_closest)==0:
                print('N.A. for query time:' +str(d_y))
                continue
            pre_scores=np.mean(np.stack(k_closest,0),0)
            loss=np.mean(np.abs(y-pre_scores))/(np.mean(data))
            Loss.append(loss)
            #plt.figure()
            #sb.heatmap([y,pre_scores], vmin=0, vmax=0.05, cmap='jet')
            #plt.show()
            #gallary[np.argmin(D)]
            a=1
        LLL.append(np.mean(Loss))
    LLLL.append(LLL)
LLLL=np.array(LLLL)
print(LLLL)