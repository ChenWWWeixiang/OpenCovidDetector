import os,datetime
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
inpath='/mnt/data9/mp_NCPs/lesions'
outpath='/mnt/data9/mp_NCPs/mp_analysis-45'
os.makedirs(outpath,exist_ok=True)
def inter_vecter(v):
    length=v.shape[0]
    x=np.linspace(0, 1, 50)
    xp = np.linspace(0, 1, length)
    new_v=np.interp(x, xp, v)
    return new_v
def inter_vecter_time(v,time):
    length=v.shape[0]
    x=np.linspace(0, 45, 45)
    #xp = np.linspace(0, 60, length)
    new_v=np.interp(x, time, v)
    return new_v
Del=[]
for patient in os.listdir(inpath):
    Data=[]
    Date=[]
    for file in os.listdir(os.path.join(inpath,patient)):
        data=sitk.ReadImage(os.path.join(inpath,patient,file))
        data=sitk.GetArrayFromImage(data)
        data=data.mean(1).mean(1)
        Data.append(data)
        Date.append(file.split('.mha')[0])

    idx = np.argsort(Date)
    Date=np.array(Date)[idx]
    Date = [datetime.datetime.strptime('2020-'+date, '%Y-%m-%d') for date in Date]
    delta = np.array([(Date[i]-Date[0]).days for i in range(len(Data))])
    Data=np.array(Data)
    Data=Data[idx]
    Del.append(delta.max())
    #print(patient,delta.max(),delta.min())
    this_pred = np.stack([inter_vecter(da) for da in Data])
    this_pred = np.stack([inter_vecter_time(this_pred[:,i],delta) for i in range(this_pred.shape[1])])
    #this_pred=np.concatenate([delta,this_pred],1)
    print(patient, this_pred.shape,delta.max())
    np.save(outpath+'/'+patient+'.npy',this_pred)
#plt.hist(Del)
#plt.show()
