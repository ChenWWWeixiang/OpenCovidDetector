import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
root='/home/cwx/extra/CAP'

f=open('cap_ages_genders.txt','w')
for person in os.listdir(root):
    for phase in os.listdir(os.path.join(root,person)):
        if not os.path.isdir(os.path.join(root,person, phase)):
            continue
        for inner in os.listdir(os.path.join(root,person, phase)):
            if not os.path.isdir(os.path.join(root,person, phase, inner)):
                continue
            for innner in os.listdir(os.path.join(root, person, phase,inner)):
                if not os.path.isdir(os.path.join(root, person, phase, inner,innner)):
                    continue
                adicom = os.listdir(os.path.join(root,person, phase, inner,innner))
                adicom = [a for a in adicom if a[0] == 'I']
                adicom = adicom[0]
                # print(os.path.join(root, patient, case, phase, inner, adicom))
                ds = pydicom.read_file(os.path.join(root,person, phase, inner,innner, adicom))
                try:
                    age = int(ds['PatientAge'].value[:-1])
                    sex = ds['PatientSex'].value
                except:
                    age = int(ds['StudyDate'].value[:4])-int(ds['PatientBirthDate'].value[:4])
                    sex = ds['PatientSex'].value
                f.writelines('CAP/'+person+'\t'+str(age)+'\t'+str(sex)+'\n')
                print('CAP/'+person+','+str(age) + ',' + str(sex))
                # cnt+=1
                break
            break
        break
    #break
f.close()