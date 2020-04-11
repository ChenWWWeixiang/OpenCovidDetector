import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
root='/mnt/data7/ILD/raw/'

f=open('ill_ages_genders.txt','w')
for person in os.listdir(root):
    adicom = os.listdir(os.path.join(root, person))
    adicom = [a for a in adicom if a[-3:] == 'dcm']

    adicom = adicom[0]

    # print(os.path.join(root, patient, case, phase, inner, adicom))
    ds = pydicom.read_file(os.path.join(root, person, adicom))
    try:
        age = int(ds['PatientAge'].value[:-1])
        sex = ds['PatientSex'].value
    except:
        age = int(ds['StudyDate'].value[:4])-int(ds['PatientBirthDate'].value[:4])
        sex = ds['PatientSex'].value
    f.writelines('ILD/'+person+'\t'+str(age)+'\t'+str(sex)+'\n')
    print('ILD/'+person+','+str(age) + ',' + str(sex))

f.close()