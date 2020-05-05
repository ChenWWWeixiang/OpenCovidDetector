import SimpleITK as sitk
import numpy as np
import os,glob
import sys
import pydicom
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/home/cwx/extra/covid_project_data/covid2')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    #default='/home/cwx/extra/CAP/CAP')
                    default='/home/cwx/baidunetdiskdownload/zj_covid/')
args = parser.parse_args()

output_path=args.output_path
os.makedirs(output_path,exist_ok=True)
#input_path='/home/cwx/extra/NCP_ill'
#input_c='/home/cwx/extra/new_control/control/control'
#input_c='/home/cwx/extra/3rd/control/control '
input_path=args.input_path
reader = sitk.ImageSeriesReader()
name_dict=dict()
for i in os.listdir(input_path):
    path=os.path.join(input_path,i)
    all_id = os.listdir(path)
    for id in all_id:
        all_phase=os.listdir(os.path.join(path,id))
        num_phase=len(all_phase)
        cts=glob.glob(os.path.join(output_path, '10_' + id.split('-')[0] + '_*.nii'))
        if len(cts)>=1:
            continue
        for phase in all_phase:
            adicom = os.listdir(os.path.join(path, id, phase))
            SID = dict()
            for adicom1 in adicom:
                ds = pydicom.read_file(os.path.join(path, id, phase, adicom1))
                sid = ds['SeriesInstanceUID'].value
                if not 'AXIAL' in ds['ImageType'].value:
                    continue
                if sid in SID.keys():
                    continue
                else:
                    date = ds['StudyDate'].value
                    SID[sid] = date
            try:
                age = int(ds['PatientAge'].value[:-1])
                sex = ds['PatientSex'].value
            except:
                age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
                sex = ds['PatientSex'].value
            case_path=os.path.join(path, id, phase)
            series = reader.GetGDCMSeriesIDs(case_path)
            I=[]
            max_size=0
            thisa=[]
            for _, ase in enumerate(series):
                if not ase in SID.keys():
                    continue
                dicom_names = reader.GetGDCMSeriesFileNames(case_path, ase)
                reader.SetFileNames(dicom_names)
                try:
                    image = reader.Execute()
                except:
                    continue
                if image.GetSize()[-1]<=10:
                    continue
                if image.GetSize()[-1]>max_size:
                    I=image
                    max_size=image.GetSize()[-1]
                    thisa=ase
            if not max_size==0:
                output_name = os.path.join(output_path,'10_'+id.split('-')[0]+ '_' + SID[thisa] + '_' + str(age) + '_' + sex + '.nii')
                print(output_name)
                sitk.WriteImage(I,output_name)




