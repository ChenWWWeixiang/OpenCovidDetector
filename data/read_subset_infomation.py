import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
LIST=open('train_set_age_sex.txt','w+')
tofind=open('../reader_study/answer.txt','r').readlines()
tofind=list(set([re.split('test')[-1].split('_')[0] for re in tofind if re.split('/')[-1][1] !='.']))
AGES=[]
SEX=[]
for item in tofind:
    #if item[:4]=='3/c2':
    #    continue
    if item[0]=='1':
        if item[2]=='c':
            nums=int(item.split('_')[0].split('/')[1][1:])
            set_id=nums//20+1
            in_set_id=nums%20
            if in_set_id==0:
                in_set_id=20
                set_id-=1
            root='/home/cwx/extra/NCP_CTs/NCP_control/control'+str(set_id)+'/'+str(in_set_id)
            #t=os.listdir(name)
        else:
            num=int(item.split('_')[0].split('/')[1])+100
            root = '/home/cwx/extra/NCP_CTs/NCP_ill2/' + str(num) + '/'
    elif item[0]=='2':
        if item[2]=='c':
            nums=int(item.split('_')[0].split('/')[1][1:])
            set_id=nums//20+1
            in_set_id=nums%20
            if in_set_id==0:
                in_set_id=20
                set_id-=1
            root='/home/cwx/extra/NCP_CTs/NCP_control/control'+str(set_id)+'/'+str(in_set_id)
            #t=os.listdir(name)
        else:
            num=int(item.split('_')[0].split('/')[1])
            root = '/home/cwx/extra/NCP_CTs/NCP_ill4/' + str(num) + '/'
    else:
        continue
    for case in os.listdir(os.path.join(root)):
        if not os.path.isdir(os.path.join(root, case)):
            continue
        for phase in os.listdir(os.path.join(root, case)):
            if not os.path.isdir(os.path.join(root, case, phase)):
                continue
            for inner in os.listdir(os.path.join(root, case, phase)):
                if not os.path.isdir(os.path.join(root, case, phase, inner)):
                    continue
                adicom = os.listdir(os.path.join(root, case, phase, inner))
                adicom = [a for a in adicom if a[0] == 'I']
                adicom = adicom[0]
                #print(os.path.join(root, case, phase, inner, adicom))
                ds = pydicom.read_file(os.path.join(root, case, phase, inner, adicom))
                sex = ds['PatientSex'].value=='M'
                age = (2020-int(ds['PatientBirthDate'].value[:4]))//20
                a=1
                SEX.append(sex)
                AGES.append(age)
                break
            break
        break

SEX=np.array(SEX)
AGES=np.array(AGES)
#print(np.sum(SEX==1)*161//139+7,np.sum(SEX==0)*161//139-4)
#print(np.sum(AGES==0)*161//139,np.sum(AGES==1)*161//139,np.sum(AGES==2)*161//139,np.sum(AGES>=3)*161//139+3)