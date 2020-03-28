import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
LIST=open('data_used.txt','w+')
ll=['109','106','111','113','114','119','120','122','126','104','176','142','127','180','158','161','144','182']
for idroot in range(1,6):
    root = '/home/cwx/extra/NCP_CTs/NCP_ill'+str(idroot)
    #for patient in os.listdir(root):
    for patient in ll:
        for case in os.listdir(os.path.join(root,patient)):
            if not os.path.isdir(os.path.join(root, patient, case)):
                continue
            for phase in os.listdir(os.path.join(root,patient,case)):
                if not os.path.isdir(os.path.join(root, patient, case, phase)):
                    continue
                for inner in os.listdir(os.path.join(root,patient,case,phase)):
                    if not os.path.isdir(os.path.join(root,patient,case,phase,inner)):
                        continue
                    adicom=os.listdir(os.path.join(root,patient,case,phase,inner))
                    adicom=[a for a in adicom if a[0]=='I']
                    adicom=adicom[0]
                    #print(os.path.join(root, patient, case, phase, inner, adicom))
                    ds = pydicom.read_file(os.path.join(root,patient,case,phase,inner,adicom))
                    try:
                        name=ds['PatientName']
                        id=ds['PatientID']
                        uct=ds['AccessionNumber']
                    except:
                        institution=ds['StudyDate']
                    #try:
                    #    age = int(ds['PatientAge'].value[:-1])
                    #except:
                    #    age = 2020 - int(ds['PatientBirthDate'].value[:3])
                    #print(os.path.join(root,patient,case,phase,inner,adicom))
                    #if ds['PatientSex'].value=='M':
                    #    cnt+=1
                    #else:
                    #    nn+=1
                    #date=ds['StudyDate'].value
                    #if idroot==2:
                    #    name='/mnt/data7/resampled_data/test1/'+str(int(patient)-100)+'_'+case
                    #elif idroot==4:
                    #    name = '/mnt/data7/resampled_data/test2/' + patient + '_' + case
                    #elif idroot==5:
                    #    name = '/mnt/data7/resampled_data/test3/' + patient + '_' + case
                    #else:
                    #    break
                    print(str(id.value)+','+str(name.value)+','+str(uct.value))
                    #cnt+=1
                    break
                break
            break

for idroot in range(15,14):
    root = '/home/cwx/extra/NCP_CTs/NCP_control/control'+str(idroot)
    for patient in os.listdir(root):
        for case in os.listdir(os.path.join(root,patient)):
            if not os.path.isdir(os.path.join(root, patient, case)):
                continue
            for phase in os.listdir(os.path.join(root,patient,case)):
                if not os.path.isdir(os.path.join(root, patient, case, phase)):
                    continue
                for inner in os.listdir(os.path.join(root,patient,case,phase)):
                    if not os.path.isdir(os.path.join(root,patient,case,phase,inner)):
                        continue
                    adicom=os.listdir(os.path.join(root,patient,case,phase,inner))
                    adicom=[a for a in adicom if a[0]=='I']
                    adicom=adicom[0]
                    #print(os.path.join(root, patient, case, phase, inner, adicom))
                    ds = pydicom.read_file(os.path.join(root,patient,case,phase,inner,adicom))
                    #try:
                    #    age = int(ds['PatientAge'].value[:-1])
                    #except:
                    #    age = 2020 - int(ds['PatientBirthDate'].value[:3])
                    #sex=ds['PatientSex'].value=='M'\
                    try:
                        institution=ds['StudyDate']
                    except:
                        institution=ds['StudyDate']
                    #age=int(ds['PatientAge'].value[:-1])
                    #date=ds['StudyDate'].value
                    cid=(int(idroot)-1)*20+int(patient)
                    if cid<100:
                        name='/mnt/data7/resampled_data/test1/c'+str(cid)+'_'+case
                    elif cid>=160 and cid<=200:
                        name = '/mnt/data7/resampled_data/test3/c' + str(cid) + '_' + case
                    else:
                        break
                    LIST.writelines(name+'\t'+str(institution)+'\n')
                    cnt+=1
                    break
                #break
            #break
