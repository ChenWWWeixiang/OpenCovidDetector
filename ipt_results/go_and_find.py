import SimpleITK as sitk
import os,glob
import numpy as np
import cv2
diction=['/mnt/data7/resampled_data/test1','/mnt/data7/resampled_data/test2',
            '/mnt/data7/resampled_data/test3','/mnt/data7/LIDC/resampled_data']
query='/mnt/data7/reader'
OUTPUT_RESAMPLE='/mnt/data7/reader_collections'
os.makedirs(OUTPUT_RESAMPLE,exist_ok=True)
dictionary=[]
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


for r1 in diction:
    D = glob.glob(r1 + '/*.n*')
    dictionary+=D

for c_item in dictionary:
    break
    if os.path.exists(os.path.join(OUTPUT_RESAMPLE, c_item.split('/')[-2]+'_'+c_item.split('/')[-1])):
        continue
    data_c=sitk.ReadImage(c_item)
    data_c=get_resampled(data_c, resampled_spacing=[0.7, 0.7, 5], l=False)
    resampled_data_ar = sitk.GetArrayFromImage(data_c)
    xx, yy, zz = np.where(resampled_data_ar > 0)
    data_c = resampled_data_ar[xx.min():xx.max(), yy.min():yy.max(), zz.min():zz.max()]
    data_c=sitk.GetImageFromArray(data_c)
    print(os.path.join(OUTPUT_RESAMPLE, c_item.split('/')[-2]+'_'+c_item.split('/')[-1]),data_c.GetSize())
    sitk.WriteImage(data_c, os.path.join(OUTPUT_RESAMPLE, c_item.split('/')[-2]+'_'+c_item.split('/')[-1]))

result=open('answer.txt','w+')
for item in os.listdir(query):
    data_q=sitk.ReadImage(os.path.join(query,item))
    data_q=sitk.GetArrayFromImage(data_q)
    #print(data_q.shape)
    m=100000000
    record='no find'
    for c_item in glob.glob(OUTPUT_RESAMPLE+'/*.n*'):
        data_c=sitk.ReadImage(c_item)
        data_c=sitk.GetArrayFromImage(data_c)
        #print(data_c.shape)
        if not data_q.shape[0]==data_c.shape[0]:
            continue
        mse=0
        for i in range(data_q.shape[0]):
            I=data_q[i,:,:]
            J=data_c[i,:,:]
            mse+=np.abs(I-cv2.resize(J,(I.shape[1],I.shape[0]))).sum()
        #mse=data_c-data_q
        if mse<m:
            m=mse
            record=c_item
    if record.split('/')[-1][:4]=='test':
        names = '/mnt/data7/resample_data/'+record.split('/')[-1].split('_')[0] + '/' + \
                record.split('/')[-1].split('_')[1]+'_'+record.split('/')[-1].split('_')[2]
    else:
        if record=='no find':
            continue
        names='/mnt/data7/LIDC/resampled_data/'+record.split('/')[-1].split('_')[-1]
    result.writelines(item.split('.nii')[0]+'\t'+names+'\n')
    print(item.split('.nii')[0]+'\t'+names+'\t'+str(m)+'\n')
