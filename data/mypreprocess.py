import SimpleITK as sitk
import numpy as np
import os,glob
import sys
import pydicom
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/home/cwx/extra/covid_project_data/cap2')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    default='/home/cwx/extra/CAP/CAP')
                    #default='/home/cwx/extra/NCP_CTs/NCP_ill')
args = parser.parse_args()

output_path=args.output_path
os.makedirs(output_path,exist_ok=True)
#input_path='/home/cwx/extra/NCP_ill'
#input_c='/home/cwx/extra/new_control/control/control'
#input_c='/home/cwx/extra/3rd/control/control '
input_path=args.input_path
for i in range(2,5):
    #
    path=input_path+str(i)
    #path=input_path
    all_id = os.listdir(path)
    for id in all_id:
        all_phase=os.listdir(os.path.join(path,id))
        num_phase=len(all_phase)
        for phase in all_phase:
            inner=os.listdir(os.path.join(path,id,phase))
            for itemsinnner in inner:
                if itemsinnner == "DICOMDIR" or itemsinnner == 'LOCKFILE' or itemsinnner == 'VERSION':
                    continue
                iinner=os.listdir(os.path.join(path,id,phase,itemsinnner))
                for iinn_item in iinner:
                    if iinn_item=='VERSION':
                        continue
                    try:
                        case_path=os.path.join(path,id,phase,itemsinnner,iinn_item)
                        reader = sitk.ImageSeriesReader()
                        dicom_names = reader.GetGDCMSeriesFileNames(case_path)
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                    except:
                        continue
                    if image.GetSize()[-1]<=10:
                        continue
                    adicom = os.listdir(os.path.join(path,id,phase,itemsinnner,iinn_item))
                    adicom = [a for a in adicom if a[0] == 'I']
                    adicom = adicom[0]
                    # print(os.path.join(root, patient, case, phase, inner, adicom))
                    ds = pydicom.read_file(os.path.join(path,id,phase,itemsinnner,iinn_item, adicom))
                    date=ds['StudyDate'].value
                    try:
                        age = int(ds['PatientAge'].value[:-1])
                        sex = ds['PatientSex'].value
                    except:
                        try:
                            age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
                            sex = ds['PatientSex'].value
                        except:
                            continue
                    output_name = os.path.join(output_path, str(i) + '_' + str(id) + '_' + date+'_'+str(age)+'_'+sex+ '.nii')
                    print(output_name)
                    sitk.WriteImage(image,output_name)




