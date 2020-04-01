import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
root='/mnt/data7/resampled_data/'
ill_root = '/home/cwx/extra/NCP_CTs/NCP_ill'
control_root = '/home/cwx/extra/NCP_CTs/NCP_control/control'
f=open('all_ages_genders.txt','w')
for set_name in os.listdir(root):
    if set_name=='train3' or set_name=='readme':
        continue
    for person in os.listdir(os.path.join(root,set_name)):
        if person[0]=='c':
            set_id=int(person[1:].split('_')[0])//20+1
            inset_id=int(person[1:].split('_')[0])%20
            if inset_id==0:
                inset_id=20
                set_id=set_id-1
            datapath=os.path.join(control_root+str(set_id),str(inset_id),'1')
        else:
            inset_id = person.split('_')[0]
            phase_id = person.split('_')[1].split('.nii')[0]
            if set_name[:5]=='train':
                set_id=int(set_name[-1])*2-1
            else:
                set_id = int(set_name[-1]) * 2
                if set_id==6:
                    set_id=5
                if set_id==2:
                    inset_id=int(inset_id)+100
            datapath=os.path.join(ill_root+str(set_id),str(inset_id),phase_id)
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
                f.writelines(set_name+'/'+person+'\t'+str(age)+'\t'+str(sex)+'\n')
                print(set_name+'/'+person+','+str(age) + ',' + str(sex))
                # cnt+=1
                break
            break
f.close()