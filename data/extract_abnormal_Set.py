import SimpleITK as sitk
import glob,os
import cv2
import numpy as np
mask_path='/mnt/data7/slice_test_seg/ct_lesion-type_20200311_177'
lung_path='/home/xzw/lung_seg'
img_path='/mnt/data6/lung_data'
output_path_slices='/mnt/data7/slice_test_seg/mask_jpgs_new'
output_path_raw='/mnt/data7/slice_test_seg/raw_jpgs_new'
os.makedirs(output_path_raw,exist_ok=True)
os.makedirs(output_path_slices,exist_ok=True)

all_files=glob.glob(mask_path+'/*mask-label.nii')

for item  in all_files:
    mask=sitk.ReadImage(item)
    mask=sitk.GetArrayFromImage(mask)
    zz,yy,xx=np.where(mask>0)

    name=item.split('/')[-1].split('-mask')[0]
    if int(name.split('_')[0])<100:
        full_name=os.path.join(img_path,'lung_1st',name.split('_')[0]+'_'+name.split('_')[1]+'.nii')
        full_name_lung=os.path.join(lung_path,'illPatient1_'+name+'_label.nii')
    else:
        full_name=os.path.join(img_path,'lung_2rd',str(int(name.split('_')[0])-100)+'_'+ name.split('_')[1]+'.nii')
        full_name_lung = os.path.join(lung_path, 'illPatient2_' + str(int(name.split('_')[0])-100)+
                                      '_'+ name.split('_')[1] + '_label.nii')
    data=sitk.ReadImage(full_name)
    data=sitk.GetArrayFromImage(data)
    lung=sitk.ReadImage(full_name_lung)
    lung=sitk.GetArrayFromImage(lung)
    for i in range(lung.shape[0]):
        if lung[i,:,:].sum()<50:
            continue
        I=data[i,:,:]
        I[I > 700] = 700
        I[I < -1200] = -1200
        I = I * 255.0 / 1900
        I = I - I.min()
        R=np.stack([I,I,I],-1).astype(np.uint8)
        I = np.stack([I, I, lung[i,:,:]*255], -1).astype(np.uint8)
        if mask[i,:,:].sum()<10:
            #continue
            cv2.imwrite(os.path.join(output_path_slices,'c--'+ name + '_' + str(i) + '.jpg'), I)
            cv2.imwrite(os.path.join(output_path_raw, 'c--'+name + '_' + str(i) + '.jpg'), R)
        else:
            cv2.imwrite(os.path.join(output_path_slices, name + '_' + str(i) + '.jpg'), I)
            cv2.imwrite(os.path.join(output_path_raw, name + '_' + str(i) + '.jpg'), R)

