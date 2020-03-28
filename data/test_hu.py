import SimpleITK as sitk
import numpy as np
from PIL import Image
import cv2,os
input_path='/mnt/data7/ILD/train20_data'
input_mask='/mnt/data7/ILD/train20_seg'
#input_path='/mnt/data7/resampled_data/train3'
#input_mask='/mnt/data7/resampled_seg/train3'
output_path_slices='/mnt/data7/resampled_jpgs/masked_ild'
output_path_raw='/mnt/data7/resampled_jpgs/raw_ild'
#output_path_cropped='/mnt/data6/lung_resample_lungbox'
#output_path_npy='/mnt/data6/lung_resample_npy'
os.makedirs(output_path_slices,exist_ok=True)
os.makedirs(output_path_raw,exist_ok=True)
#os.makedirs(output_path_cropped,exist_ok=True)
#os.makedirs(output_path_npy,exist_ok=True)
#data=np.stack([data,data,data],0)
cnt=0
name_list=os.listdir(input_path)
for idx,name in enumerate(name_list):
    #if idx>=100:
    #    break
    volume = sitk.ReadImage(os.path.join(input_path,name))
    mask=sitk.ReadImage(os.path.join(input_mask,name))
    M=sitk.GetArrayFromImage(mask)
    V = sitk.GetArrayFromImage(volume)
    V = V[-300:-40,:,:]
    M = M[-300:-40,:V.shape[1],:V.shape[2]]
    V=V[:M.shape[0],:M.shape[1],:M.shape[2]]
    #volume_box=sitk.GetImageFromArray(V)
    #sitk.WriteImage(volume_box,os.path.join(output_path_cropped,name))
    V_set=[]
    for idx, i in enumerate(range(V.shape[0] - 40, 45, -5)):
    #for idx,i in enumerate(range(V.shape[1]-100,70,-5)):
    #for idx,i in enumerate(range(V.shape[2]-40,40,-5)):
        if idx>=60:
            break
        data=V[i-1:i+1,:,:]
        data[data>700]=700
        data[data<-1200]=-1200
        data=data*255.0/1900

        data=data-data.min()
        data_raw=np.concatenate([data[0:1,:,:],data[0:1,:,:],data[0:1,:,:]],0)
        data_raw=data_raw.astype(np.uint8).transpose(1,2,0)
        #data=data/data.max()
        data=np.concatenate([data,M[i:i+1,:,:]*255],0)#mask one channel
        data = data.astype(np.uint8).transpose(1,2,0)
        #dst = cv2.equalizeHist(data)
        #V_set.append(data)
       # if os.path.exists(os.path.join(output_path_raw,name.split('.n')[0]+'_'+str(i)+'.jpg')):
        cv2.imwrite(os.path.join(output_path_slices,name.split('.n')[0]+'_'+str(i)+'.jpg'),data)
        cv2.imwrite(os.path.join(output_path_raw, name.split('.n')[0] + '_' + str(i) + '.jpg'), data_raw)
    a=1
    #V_set=np.stack(V_set,0)
   # np.save(os.path.join(output_path_npy,name+'_'+str(idx)+'.npy'),V_set)