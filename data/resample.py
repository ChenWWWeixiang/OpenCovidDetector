import SimpleITK as sitk
import numpy as np
import glob
import os,json


def get_resampled(input,resampled_spacing=[1,1,1],l=False):
    resampler = sitk.ResampleImageFilter()
    if l:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(resampled_spacing)
    resampler.SetOutputOrigin(input.GetOrigin())
    resampler.SetOutputDirection(input.GetDirection())
    #ratio=np.array(input.GetSize())*np.array(input.GetSpacing())/np.array(resampled_spacing)
    #ratio=list(ratio.astyp e(np.int))
    resampler.SetSize((1000,1000,1000))
    moving_resampled = resampler.Execute(input)

    return  moving_resampled

data_path='/mnt/data7/ILD/raw'
#data_path='/mnt/data7/'

all_names=os.listdir(data_path)
gt_dir='/home/xzw/lung_seg'

#all_names=os.listdir(data_path)

#OUTPUT_DIR = '/mnt/data6/test_set/'
#OUTPUT_PRED_LUNG='/mnt/data7/LIDC/resampled_seg'
#OUTPUT_RESAMPLE='/mnt/data7/LIDC/resampled_data'
OUTPUT_PRED_LUNG='/mnt/data7/ILD/resampled_seg'
OUTPUT_RESAMPLE='/mnt/data7/ILD/resampled_data'
#os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
os.makedirs(OUTPUT_PRED_LUNG,exist_ok=True)


for key in all_names:
    if key[0]=='c':
        cset=(int(key[1:].split('_')[0]))//20+1
        cid=(int(key[1:].split('_')[0]))%20
        if cid==0:
            cset=cset-1
            cid=20
        mask_name='control'+str(cset)+'_'+str(cid)+'_1_label.nii'
        mask_name = os.path.join(gt_dir, mask_name)
    else:
        mask_name='ILD_DB_txtROIs_'+key+'_1_label.nii'
        mask_name = os.path.join(gt_dir, mask_name)
    try:

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(data_path,key))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        #image = sitk.ReadImage(os.path.join(data_path,key), sitk.sitkFloat32)#data
        mask = sitk.ReadImage(mask_name, sitk.sitkFloat32)  ##pred
        mask_new = get_resampled(mask, resampled_spacing=[1, 1, 1],l=False)
        image_new = get_resampled(image, resampled_spacing=[1, 1, 1], l=True)
    except:
        print('e'+key)
        continue

    resampled_data_ar = sitk.GetArrayFromImage(image_new)
    resampled_mask_ar = sitk.GetArrayFromImage(mask_new)

    xx, yy, zz = np.where(resampled_data_ar > 0)

    resampled_data_ar = resampled_data_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    resampled_data = sitk.GetImageFromArray(resampled_data_ar)
    resampled_mask_ar = resampled_mask_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    resampled_mask = sitk.GetImageFromArray(resampled_mask_ar)
    print(resampled_mask.GetSize())
    #a=1
    sitk.WriteImage(resampled_mask, os.path.join(OUTPUT_PRED_LUNG, key+'.nii'))
    sitk.WriteImage(resampled_data, os.path.join(OUTPUT_RESAMPLE, key+'.nii'))

