import SimpleITK as sitk
import numpy as np
from PIL import Image
import cv2,os
input_path='/mnt/data6/lung_test_resampled'
input_mask='/mnt/data7/lung_seg_test'
output_path_slices='/mnt/data7/lung_jpgs_with_SEG'
#output_path_cropped='/mnt/data6/lung_resample_lungbox'
#output_path_npy='/mnt/data6/lung_resample_npy'
os.makedirs(output_path_slices,exist_ok=True)
#os.makedirs(output_path_cropped,exist_ok=True)
#os.makedirs(output_path_npy,exist_ok=True)
#data=np.stack([data,data,data],0)

name_list=os.listdir(input_path)
for name in name_list:
    if not name[0]=='c':
        continue
    cid=int(name[1:].split('_')[0])
    if cid<100:
        continue
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
        data=data*255/1900

        #data=data-data.min()
        #data=data/data.max()
        data=np.concatenate([data,M[i:i+1,:,:]*255],0)#mask one channel
        data = data.astype(np.uint8).transpose(1,2,0)
        #dst = cv2.equalizeHist(data)
        #V_set.append(data)
        cv2.imwrite(os.path.join(output_path_slices,name[:-4]+'_'+str(i)+'.jpg'),data)
    a=1
    #V_set=np.stack(V_set,0)
   # np.save(os.path.join(output_path_npy,name+'_'+str(idx)+'.npy'),V_set)