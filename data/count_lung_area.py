import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric
import seaborn as sns
import pydicom
import SimpleITK as sitk
LIST=open('../data/ages.txt','r')
times=LIST.readlines()
tname=[n.split('\t')[0] for n in times]
sex=[n.split('\t')[1] for n in times]
ages=[int(n.split('\t')[2][:-1]) for n in times]
Re=[]
Rm=[]
if False:
    for tt,(a_name,a_sex,a_age) in enumerate(zip(tname,sex,ages)):
        temporalvolume = np.zeros(35)
        try:
            data=sitk.ReadImage((a_name+'.nii').replace('_data','_seg'))
            data=sitk.GetArrayFromImage(data)
            data = data[-300:-40, :, :]
            data=data[np.arange(data.shape[0]-40,45,-5),:,:]
            data=data.sum(1).sum(1)
            temporalvolume[:data.shape[0]]=data
            if a_sex == 'False':
                Re.append(temporalvolume)
            else:
                Rm.append(temporalvolume)
        except:
            continue

    Re=np.array(Re)
    Rm=np.array(Rm)
    Re=np.mean(Re,0)
    Rm=np.mean(Rm,0)
    A=np.stack([Re,Rm],0)
    np.save('A.npy',A)
    r1=[]
    r2=[]
    r3=[]
    for tt,(a_name,a_sex,a_age) in enumerate(zip(tname,sex,ages)):
        temporalvolume = np.zeros(35)
        try:
            data=sitk.ReadImage((a_name+'.nii').replace('_data','_seg'))
            data=sitk.GetArrayFromImage(data)
            data = data[-300:-40, :, :]
            data=data[np.arange(data.shape[0]-40,45,-5),:,:]
            data=data.sum(1).sum(1)
            temporalvolume[:data.shape[0]]=data
            if a_age //20 ==1:
                r1.append(temporalvolume)
            elif a_age//20==2:
                r2.append(temporalvolume)
            elif a_age//20==3:
                r3.append(temporalvolume)
        except:
            continue
#
    #r1=np.array(r1)
    #r2=np.array(r2)
    #r3=np.array(r3)

    r1=np.mean(r1,0)
    r2=np.mean(r2,0)
    r3=np.mean(r3,0)
    B=np.stack([r1,r2,r3],0)
B=np.load('B.npy')
A=np.load('A.npy')
Re=A[0,:]
Rm=A[1,:]
r1=B[0,:]
r2=B[1,:]
r3=B[2,:]
plt.figure(figsize=(6,6))
plt.style.use('ggplot')
plt.plot(Rm)
plt.plot(Re)
plt.ylim([7500,25000])
plt.legend(['male','female'])
print(Re,Rm)
#
plt.title('Averaged Number of Voxels of Lung')
plt.xticks(np.arange(0,40,5),np.arange(0,200,25))

plt.xlabel('Distance to Top of Lungs')
plt.ylabel('Voxels of Lungs ')
plt.savefig('sex distribution.jpg')
#plt.show()
a=1
plt.figure(figsize=(6,6))
plt.style.use('ggplot')
plt.plot(r1)
plt.plot(r2)
plt.plot(r3)
plt.legend(['20-39','40-59',">=60"])

plt.suptitle('Averaged Number of Voxels of Lung')
plt.xticks(np.arange(0,40,5),np.arange(0,200,25))
plt.ylim([7500,25000])
plt.xlabel('Distance to Top of Lungs')
plt.ylabel('Voxels of Lungs')
plt.savefig('age distribution.jpg')
plt.show()


