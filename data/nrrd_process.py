import SimpleITK as sitk
import numpy as np
import os,glob
output_path='/mnt/data7/lung_nrrd'
os.makedirs(output_path,exist_ok=True)
#input_path='/home/cwx/extra/NCP_ill'
#input_c='/home/cwx/extra/new_control/control/control'
#input_c='/home/cwx/extra/3rd/control/control '
input_path='/home/xzw/LIDC'


all_id = os.listdir(input_path)
for id in all_id:
    output_name = os.path.join(input_path, id)
    print(output_name)
    reader = sitk.ReadImage(output_name)
    data=sitk.GetArrayFromImage(reader)

    sitk.WriteImage(reader,os.path.join(output_path,id.split('.nrrd')[0]+'.nii'))