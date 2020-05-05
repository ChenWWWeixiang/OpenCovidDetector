import os
import numpy as np

f=open('3cls_train2.list','r')
data=f.readlines()
f2=open('train_covid2.list','r')
data2=f2.readlines()
data=data2+data
f2=open('3cls_test2.list','r')
data2=f2.readlines()
data=data2+data
f2=open('test_covid2.list','r')
data2=f2.readlines()
data=data2+data
data=[da.split(',')[0] for da in data]
cls_stage = []
for data_path in data:
    if 'healthy' in data_path:
        cls_stage.append(0)
    elif 'cap' in data_path:
        cls_stage.append(1)
    elif 'AB-in' in data_path:
        cls_stage.append(2)  # covid
    else:
        cls_stage.append(3)
cls=[]
person = [da.split('/')[-2] + '_' + da.split('/')[-1].split('_')[0] + '_' + da.split('/')[-1].split('_')[1] for da in
          data]

person = list(set(person))
for data_path in person:
    if 'healthy' in data_path:
        cls.append(0)
    elif 'cap' in data_path:
        cls.append(1)
    elif 'AB-in' in data_path:
        cls.append(2)  # covid
    else:
        cls.append(3)
nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
print('patient',nums)
nums = [np.sum(np.array(cls_stage) == i) for i in range(np.max(cls_stage) + 1)]
print('stages', nums)