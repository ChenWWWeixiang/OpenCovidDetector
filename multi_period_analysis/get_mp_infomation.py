import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
root='/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill'
ill_root = '/home/cwx/extra/NCP_CTs/NCP_ill'
f=open('mp_ages_genders.txt','w')
for i in range(1,8):
    for person in os.listdir(root+str(i)):
        item=os.listdir(os.path.join(ill_root+str(i),person.split('_')[0]))
        n=item[0]
        datapath=os.path.join(ill_root+str(i),person.split('_')[0],n)
        a=1
        for phase in os.listdir(datapath):
            if not os.path.isdir(os.path.join(datapath, phase)):
                continue
            for inner in os.listdir(os.path.join(datapath, phase)):
                if not os.path.isdir(os.path.join(datapath, phase, inner)):
                    continue
                adicom = os.listdir(os.path.join(datapath, phase, inner))
                adicom = [a for a in adicom if a[0] == 'I']
                adicom = adicom[0]

                # print(os.path.join(root, patient, case, phase, inner, adicom))
                ds = pydicom.read_file(os.path.join(datapath, phase, inner, adicom))
                try:
                    age = int(ds['PatientAge'].value[:-1])
                    sex = ds['PatientSex'].value
                except:
                    age = 2020-int(ds['PatientBirthDate'].value[:4])
                    sex = ds['PatientSex'].value

                f.writelines('NCP_ill'+str(i)+'/'+person.split('_')[0]+'\t'+str(age)+'\t'+str(sex)+'\n')
                print('NCP_ill'+str(i)+'/'+person.split('_')[0]+','+str(age) + ',' + str(sex))
                # cnt+=1
                break
            break
f.close()