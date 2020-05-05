import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os
#input_path='/home/cwx/extra/CAP'
#input_mask='/mnt/data6/CAP/resampled_seg'
set_name='cap-zs'
input_path='/home/cwx/extra/covid_project_data/'+set_name
input_mask='/home/cwx/extra/covid_project_segs/lungs/'+set_name
input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/'+set_name
#input_path='/mnt/data7/resampled_data/train3'
#input_mask='/mnt/data7/resampled_seg/train3'
output_path_slices='/mnt/data9/covid_detector_jpgs/crop_masked_'+set_name

os.makedirs(output_path_slices,exist_ok=True)

cnt=0

name_list=os.listdir(input_path)

reader = sitk.ImageSeriesReader()
for idx,name in enumerate(name_list):
    ##for case in os.listdir(os.path.join(input_path, name)):
     #   if not os.path.isdir(os.path.join(input_path, name, case)):
     #       continue
      #  for phase in os.listdir(os.path.join(input_path, name, case)):
       #     if not os.path.isdir(os.path.join(input_path, name, case, phase)):
        #        continue
         #   for inner in os.listdir(os.path.join(input_path, name, case, phase)):
          #      if not os.path.isdir(os.path.join(input_path, name, case, phase, inner)):
           #         continue
            #    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(input_path, name, case, phase, inner))
             #   reader.SetFileNames(dicom_names)
              #  volume = reader.Execute()
               # a=1
    set_id=int(name.split('_')[0])
    #person_id = int(name.split('_')[1].split('-')[0])
    if set_name=='healthy':
        if set_id>6:#1-6 train, 7-13 test
            continue
    if set_name=='covid' and set_id>5:
        continue
    volume = sitk.ReadImage(os.path.join(input_path,name))

    mask_name =  set_name+'_'+name
    #if not set_name=='healthy':
    lesion_name =  set_name+'_' + name.split('.nii')[0]+'_label.nrrd'
    L = sitk.ReadImage(os.path.join(input_lesion_mask, lesion_name))
    L = sitk.GetArrayFromImage(L)
    L[L>0]=1

    try:
        mask=sitk.ReadImage(os.path.join(input_mask,mask_name))
    except:
        print('e')
        continue
    M=sitk.GetArrayFromImage(mask)
    V = sitk.GetArrayFromImage(volume)
    #M = M[:V.shape[0], :, :]

    sums = M.sum(1).sum(1)
    idd=np.where(sums>500)
    iddx=np.where(M>0)
    M = M[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    V = V[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    L = L[idd[0], iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    if not set_name == 'healthy':
        sums2 = L.sum(1).sum(1)
        sums2=np.where(sums2>50)[0]
    #volume_box=sitk.GetImageFromArray(V)
    #sitk.WriteImage(volume_box,os.path.join(output_path_cropped,name))
    V_set=[]
    M[M>1]=1
    L[L>1]=1
    #for idx, i in enumerate(range(V.shape[0] - 40, 45, -5)):
    for idx, i in enumerate(range(0,V.shape[0],1)):
        if not set_name == 'healthy':
            if not i in sums2:
                continue
        data=V[i,:,:]
        data[data>500]=500
        data[data<-1200]=-1200
        data=data*255.0/1700

        data=data-data.min()
        #data_raw=np.stack([data,data,data],-1)
       # data_raw=data_raw.astype(np.uint8)
        #data=data/data.max()
        data=np.stack([data,M[i,:,:]*data,L[i,:,:]*data],-1)#mask one channel
        data = data.astype(np.uint8)
        #dst = cv2.equalizeHist(data)
        #V_set.append(data)
       # if os.path.exists(os.path.join(output_path_raw,name.split('.n')[0]+'_'+str(i)+'.jpg')):
        cv2.imwrite(os.path.join(output_path_slices,set_name+'_'+name.split('.n')[0]+'_'+str(int(i/(V.shape[0])*100))+'.jpg'),data)
        #cv2.imwrite(os.path.join(output_path_raw, set_name+'_'+name.split('.n')[0] + '_' + str(int(i/(V.shape[0])*100)) + '.jpg'), data_raw)
    #a=1
    #V_set=np.stack(V_set,0)
   # np.save(os.path.join(output_path_npy,name+'_'+str(idx)+'.npy'),V_set)