import os,cv2
import numpy as np
import SimpleITK as sitk
path=''
datas=open('record.txt','r').readlines()
all=[da[:-1].split('\t') for da in datas]
for aline in all:
    for anem in aline:
        if anem[-3:]=='nii':
            #shutil.copy(,'tomove/'+anem)
            img=sitk.ReadImage('img/'+anem)
            mask=sitk.ReadImage('mask/'+anem)
            img=sitk.GetArrayFromImage(img)
            mask=sitk.GetArrayFromImage(mask)*255
            I=np.concatenate([img,mask],1)
            cv2.imwrite('tomove/'+anem[:-3]+'jpg',I)

for file in os.listdir('img'):
    if file[0]=='c':
        img = sitk.ReadImage('img/' + file)
        listall = os.listdir('mask')
        listall = [item for item in listall if not item[0] == 'c']
        n_fake = listall[np.random.randint(0, len(listall))]
        mask = sitk.ReadImage('mask/' + n_fake)
        img = sitk.GetArrayFromImage(img)
        mask = sitk.GetArrayFromImage(mask) * 255
        I = np.concatenate([img, mask], 1)
        cv2.imwrite('tomove/' + file[:-3] + 'jpg', I)