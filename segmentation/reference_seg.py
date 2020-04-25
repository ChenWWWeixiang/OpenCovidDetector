import SimpleITK as sitk 
import numpy as np
from lungmask import mask
import glob
import os 
from predict import predict,get_model
from unet import UNet
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
lung_dir = '/Extra/xuzhanwei/CoV19_data/Output/finial/reference/lung/'
leision_dir = '/Extra/xuzhanwei/CoV19_data/Output/finial/reference/leision/'
root_dir = '/Extra/xuzhanwei/CoV19_data/covid_project_data/cap_*'
filelist = glob.glob(root_dir)
model2 = './checkpoint_200000.pth'
model = get_model(model2)
print('get model done')
for filepath in filelist:
    imagelist = glob.glob(filepath+'/*.nii')
    for imagepath in imagelist:
        imagename = imagepath.split('/')[-1]
        batch_id = imagepath.split('/')[-2]
        if os.path.exists(leision_dir+batch_id+'_'+imagename):
            print(imagename)
            continue
        input_image = sitk.ReadImage(imagepath)
        segmentation = predict(input_image, model = model,batch_size=10)
        lung_image = sitk.ReadImage(lung_dir+batch_id+'_'+imagename)
        lung_data = sitk.GetArrayFromImage(lung_image)
        leision_seg = lung_data*segmentation
        result_out= sitk.GetImageFromArray(leision_seg)
        result_out.CopyInformation(input_image)
 
        sitk.WriteImage(result_out,leision_dir+batch_id+'_'+imagename)
        print(imagename)
