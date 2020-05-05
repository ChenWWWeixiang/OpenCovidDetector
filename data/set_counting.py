import os,glob
import numpy as np

f=open('3cls_test2.list','r')
data=f.readlines()
f2=open('train_covid2.list','r')
data2=f2.readlines()
data=data2
#f2=open('3cls_train2.list','r')
#data2=f2.readlines()
#data=data2+data
#f2=open('train_covid2.list','r')
#data2=f2.readlines()
#data=data2+data
cls=[]
data=[da.split(',')[0] for da in data]

person = [da.split('/')[-2] + '_' + da.split('/')[-1].split('_')[0] + '_' + da.split('/')[-1].split('_')[1] for da in
          data]

person = list(set(person))
gender=[]
age=[]
num=[]
for data_path in person:
    if 'healthy' in data_path:
        cls.append(0)

    elif 'cap' in data_path:
        cls.append(1)

    elif 'AB-in' in data_path:
        cls.append(2)  # covid
        try:
            gender.append(fullname[0].split('_')[-1].split('.nii')[0])
            age.append(int(fullname[0].split('_')[-2]))
            num.append(len(fullname))
        except:
            a = 1
    else:
        cls.append(3)
        fullname = glob.glob(
            '/home/cwx/extra/covid_project_segs/lungs/' + data_path.split('_')[0] + '/' + data_path + '*.nii')

nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
print('patient',nums)
age=np.array(age)
gender=np.array(gender)
num=np.array(num)
age=age//20
print(np.sum(gender=='M'),np.sum(gender=='F'))
print(np.sum(age==0),np.sum(age==1),np.sum(age==2),np.sum(age==3),np.sum(age>=4))
print(np.sum(num==1),np.sum(num==2),np.sum(num==3),np.sum(num>3))