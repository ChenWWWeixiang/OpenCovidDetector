import os
import xlrd
import SimpleITK as sitk
import cv2
import numpy as np

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


answer=open('answer.txt','r').readlines()
read_name=[an.split('\t')[0] for an in answer]
id1=[int(an.split('\t')[1][:-1]) for an in answer]
workbook=xlrd.open_workbook("reader_test 1.xlsx")
worksheet=workbook.sheet_by_index(0)
id2=np.array(worksheet.col_values(0,5),np.int)
significant=np.array([worksheet.col_values(2,5),
             worksheet.col_values(3,5),
             worksheet.col_values(4,5),
             worksheet.col_values(5,5)])
significant[significant=='']=0
significant=significant.astype(np.float)
slices=worksheet.col_values(6,5)
slices=np.array(slices)
slices[slices=='']=0
slices=slices.astype(np.float)
to_cut=np.where(np.sum(significant,0)>0)
def get_mask(read_name_this):
    data_bf_resample = sitk.ReadImage(read_name_this)
    read_name_this=read_name_this.replace('resampled_data','resampled_seg')
    seg_bf_resample=sitk.ReadImage(read_name_this)
    data = get_resampled(data_bf_resample, resampled_spacing=[0.7, 0.7, 5], l=False)
    seg = get_resampled(seg_bf_resample, resampled_spacing=[0.7, 0.7, 5], l=False)
    data_ar=sitk.GetArrayFromImage(data)
    seg_ar=sitk.GetArrayFromImage(seg)
    xx,yy,zz=np.where(data_ar>0)
    seg_ar = seg_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    #resampled_seg = sitk.GetImageFromArray(seg_ar)
    return seg_ar
name_sub=['co','mix','paving','GGO']
for i in to_cut[0]:
    id_to_find=id2[i]
    slice_idx=slices[i]
    cls=np.argmax(significant[:,i])
    idx=id1.index(id_to_find)
    read_name_this=read_name[idx]
    seg_mask=get_mask(read_name_this)
    #if os.path.exists('mask_img/'+str(id_to_find)+'_'+name_sub[cls-1]+'.jpg'):
    #    continue
    data=sitk.ReadImage(os.path.join('/mnt/data7/reader',str(id_to_find)+'.nii'))
    data=sitk.GetArrayFromImage(data)
    data=data[:seg_mask.shape[0],:seg_mask.shape[1],:seg_mask.shape[2]]
    seg_mask=seg_mask[:data.shape[0],:data.shape[1],:data.shape[2]]
    data=data[int(slice_idx)-1:int(slice_idx)+2,:,:]
    mask=seg_mask[int(slice_idx):int(slice_idx)+1,:,:]

    data[data > 700] = 700
    data[data < -1200] = -1200
    data = data * 255.0 / 1900
    data = data - data.min()
    I=np.concatenate([data[0:2,:,:],mask*255],0)
    I = I.astype(np.uint8).transpose(1,2,0)
    cv2.imwrite('mask_img/'+str(id_to_find)+'_'+name_sub[cls-1]+'.jpg',I)
    J = data[0,:,:]
    J = J.astype(np.uint8)#.transpose(1, 2, 0)
    cv2.imwrite('sig_img/' + str(id_to_find) + '_' + name_sub[cls - 1] + '.jpg', J)
