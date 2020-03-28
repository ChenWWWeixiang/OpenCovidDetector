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

data_path='/home/xzw/LIDC'
gt_dir='/home/xzw/lung_seg'

all_names=os.listdir(data_path)

#OUTPUT_DIR = '/mnt/data6/test_set/'
OUTPUT_PRED_LUNG='/mnt/data7/LIDC/LIDC_resampled_seg'
OUTPUT_RESAMPLE='/mnt/data7/LIDC/LIDC_resampled'

os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
os.makedirs(OUTPUT_PRED_LUNG,exist_ok=True)


for key in all_names:
    data_name=key
    data_name = os.path.join(gt_dir, data_name)
    #if os.path.exists(os.path.join(OUTPUT_PRED_LUNG, key)):
    #    continue
    try:
        data = sitk.ReadImage(os.path.join(data_path,key), sitk.sitkFloat32)
        seg = sitk.ReadImage(os.path.join(gt_dir,key), sitk.sitkFloat32)  ##
        seg = get_resampled(seg, resampled_spacing=[1, 1, 1],l=False)
        data = get_resampled(data, resampled_spacing=[1, 1, 1], l=False)
    except:
        continue

    resampled_data_ar = sitk.GetArrayFromImage(seg)
    data_ar = sitk.GetArrayFromImage(data)

    xx, yy, zz = np.where(data_ar > 0)

    resampled_data_ar = resampled_data_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    data_ar = data_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    resampled_pred = sitk.GetImageFromArray(resampled_data_ar)
    resmapled_data = sitk.GetImageFromArray(data_ar)
    sitk.WriteImage(resampled_pred, os.path.join(OUTPUT_PRED_LUNG, key))
    sitk.WriteImage(resmapled_data, os.path.join(OUTPUT_RESAMPLE, key))
    print(key)

