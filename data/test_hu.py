import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os
#input_path='/home/cwx/extra/CAP'
#input_mask='/mnt/data6/CAP/resampled_seg'
set_name='covid'
input_path='/home/cwx/extra/covid_project_data/'+set_name
input_mask='/home/cwx/extra/covid_project_segs/lungs/'+set_name
input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/'+set_name
#input_path='/mnt/data7/resampled_data/train3'
#input_mask='/mnt/data7/resampled_seg/train3'
output_path_slices='/mnt/data9/covid_detector_jpgs/masked_'+set_name
output_path_raw='/mnt/data9/covid_detector_jpgs/raw_'+set_name
#old_path='/mnt/data7/resampled_jpgs/masked_test3'
#output_path_cropped='/mnt/data6/lung_resample_lungbox'
#output_path_npy='/mnt/data6/lung_resample_npy'
os.makedirs(output_path_slices,exist_ok=True)
os.makedirs(output_path_raw,exist_ok=True)
#os.makedirs(output_path_cropped,exist_ok=True)
#os.makedirs(output_path_npy,exist_ok=True)
#data=np.stack([data,data,data],0)
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
    if set_name=='cap' and person_id>100:
        continue
    if set_name=='ild' and person_id>84:
        continue
    if set_name=='covid' and set_id>3:
        continue
    volume = sitk.ReadImage(os.path.join(input_path,name))

    mask_name =  set_name+'_'+name
    if set_name=='covid':
        lesion_name =  'covid_' + name.split('.nii')[0]+'_label.nrrd'
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

    if set_name=='covid':
        sums = L.sum(1).sum(1)
    else:
        sums = M.sum(1).sum(1)
    idd=np.where(sums>50)[0]
    M=M[idd,:,:]
    V=V[idd,:,:]
    #V = V[-300:-40,:,:]
    #M = M[-300:-40,:V.shape[1],:V.shape[2]]
    V=V[:M.shape[0],:M.shape[1],:M.shape[2]]
    M = M[:V.shape[0], :V.shape[1], :V.shape[2]]
    #volume_box=sitk.GetImageFromArray(V)
    #sitk.WriteImage(volume_box,os.path.join(output_path_cropped,name))
    V_set=[]
    #for idx, i in enumerate(range(V.shape[0] - 40, 45, -5)):
    for idx, i in enumerate(range(1,V.shape[0]-3,3)):

        data=V[i-1:i+2,:,:]
        data[data>700]=700
        data[data<-1200]=-1200
        data=data*255.0/1900

        data=data-data.min()
        data_raw=np.stack([data[1,:,:],data[1,:,:],data[1,:,:]],-1)
        data_raw=data_raw.astype(np.uint8)
        #data=data/data.max()
        data=np.concatenate([data[0:2,:,:],M[i:i+1,:,:]*255],0)#mask one channel
        data = data.astype(np.uint8).transpose(1,2,0)
        #dst = cv2.equalizeHist(data)
        #V_set.append(data)
       # if os.path.exists(os.path.join(output_path_raw,name.split('.n')[0]+'_'+str(i)+'.jpg')):
        cv2.imwrite(os.path.join(output_path_slices,name.split('.n')[0]+'_'+str(i)+'.jpg'),data)
        cv2.imwrite(os.path.join(output_path_raw, name.split('.n')[0] + '_' + str(i) + '.jpg'), data_raw)
    a=1
    #V_set=np.stack(V_set,0)
   # np.save(os.path.join(output_path_npy,name+'_'+str(idx)+'.npy'),V_set)