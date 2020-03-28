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

#data_path='/mnt/data6/lung_data/lung_3rd'
data_path=os.listdir('/mnt/data7/filtered_mp_CT/images')
all_names=[]
for da in data_path:
    all_names+=glob.glob(os.path.join('/mnt/data7/filtered_mp_CT/images', da)+'/*.mha')
gt_dir='/mnt/data7/filtered_mp_CT/lungsegs'

#all_names=os.listdir(data_path)

#OUTPUT_DIR = '/mnt/data6/test_set/'
#OUTPUT_PRED_LUNG='/mnt/data7/LIDC/resampled_seg'
#OUTPUT_RESAMPLE='/mnt/data7/LIDC/resampled_data'
OUTPUT_PRED_LUNG='/mnt/data7/slice_test_seg/mp_seg'
OUTPUT_RESAMPLE='/mnt/data7/slice_test_seg/mp_data'
#os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
os.makedirs(OUTPUT_PRED_LUNG,exist_ok=True)


for key in all_names:
    #if os.path.exists(os.path.join(OUTPUT_PRED_LUNG, key)):
    #    continue
    mask_name=key.split('/')[-2]+'/'+key.split('/')[-1]
    if key[0]=='c':
        cset=(int(key[1:].split('_')[0])-200)//20+1
        cid=(int(key[1:].split('_')[0])-200)%20
        if cid==0:
            cset=cset-1
            cid=20
        mask_name='control1'+str(cset)+'_'+str(cid)+'_1_label.nii'
        mask_name = os.path.join(gt_dir, mask_name)
    else:
        #if int(key.split('_')[0])<40:
        #    mask_name = key.split('.nii')[0] + '-mask-label.nii'
        #else:
        #    mask_name = '1'+key.split('.nii')[0]+ '-mask-label.nii'
        mask_name = os.path.join(gt_dir, mask_name)
    try:

        #reader = sitk.ImageSeriesReader()
        #dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(data_path,key))
        #reader.SetFileNames(dicom_names)
        #image = reader.Execute()
        image = sitk.ReadImage(key, sitk.sitkFloat32)#data
        mask = sitk.ReadImage(mask_name, sitk.sitkFloat32)  ##pred
        #mask_new = get_resampled(mask, resampled_spacing=[1, 1, 1],l=False)
        #image_new = get_resampled(image, resampled_spacing=[1, 1, 1], l=False)
    except:
        print('e'+key)
        continue

    #resampled_data_ar = sitk.GetArrayFromImage(image_new)
    #resampled_mask_ar = sitk.GetArrayFromImage(mask_new)

    #xx, yy, zz = np.where(resampled_data_ar > 0)

    #resampled_data_ar = resampled_data_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    #resampled_data = sitk.GetImageFromArray(resampled_data_ar)
    #resampled_mask_ar = resampled_mask_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    #resampled_mask = sitk.GetImageFromArray(resampled_mask_ar)
    #print(resampled_data_ar.shape)
    #a=1
    sitk.WriteImage(mask, os.path.join(OUTPUT_PRED_LUNG, key.split('/')[-2]+'_'+key.split('/')[-1][:-4]+'.nii'))
    sitk.WriteImage(image, os.path.join(OUTPUT_RESAMPLE, key.split('/')[-2]+'_'+key.split('/')[-1][:-4]+'.nii'))

