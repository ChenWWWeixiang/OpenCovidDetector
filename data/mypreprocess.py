import SimpleITK as sitk
import numpy as np
import os,glob
output_path='/mnt/data6/lung_data/lung_5th'
os.makedirs(output_path,exist_ok=True)
#input_path='/home/cwx/extra/NCP_ill'
#input_c='/home/cwx/extra/new_control/control/control'
#input_c='/home/cwx/extra/3rd/control/control '
input_path='/home/cwx/extra/NCP_CTs/NCP_control/control1'
for i in range(3):
    #
    path=input_path+str(i+1)
    #path=input_path
    all_id = os.listdir(path)
    for id in all_id:
        all_phase=os.listdir(os.path.join(path,id))
        for phase in all_phase:
            inner=os.listdir(os.path.join(path,id,phase))
            for itemsinnner in inner:
                if itemsinnner == "DICOMDIR" or itemsinnner == 'LOCKFILE' or itemsinnner == 'VERSION':
                    continue
                iinner=os.listdir(os.path.join(path,id,phase,itemsinnner))
                for iinn_item in iinner:
                    if iinn_item=='VERSION':
                        continue
                    cid = 200+int(id) + i * 20
                    output_name = os.path.join(output_path, 'c'+str(cid) + '_' + phase + '.nii')
                    #output_name = os.path.join(output_path, str(id) + '_' + phase + '.nii')
                    print(output_name)
                    case_path=os.path.join(path,id,phase,itemsinnner,iinn_item)
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(case_path)
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    if image.GetSize()[-1]<=10:
                        continue
                    sitk.WriteImage(image,output_name)




