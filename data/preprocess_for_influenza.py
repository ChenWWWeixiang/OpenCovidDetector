import SimpleITK as sitk
import numpy as np
import os,glob
import sys
import pydicom
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-o", "--output_path", help="path to output nii files",  type=str,
                    default='/home/cwx/extra/covid_project_data/AB-in')
parser.add_argument("-i", "--input_path", help="path to input dicom files",  type=str,
                    #default='/home/cwx/extra/NCP_CTs/NCP_control/control')
                    #default='/home/cwx/extra/CAP/CAP')
                    default='/home/cwx/baidunetdiskdownload/HXNX/')
args = parser.parse_args()

output_path=args.output_path
os.makedirs(output_path,exist_ok=True)
path=args.input_path
reader = sitk.ImageSeriesReader()
name_dict=dict()
cnt=0
for id in os.listdir(path):
    name=id.split('_')[0]
    if not name in name_dict:
        name_dict[name]=cnt
        cnt+=1
    adicom = os.listdir(os.path.join(path, id))
    SID = dict()
    for adicom1 in adicom:
        ds = pydicom.read_file(os.path.join(path, id, adicom1))
        sid = ds['SeriesInstanceUID'].value
        if not ds['ImageType'].value[-1] == 'AXIAL':
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
    case_path=os.path.join(path, id)
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
        output_name = os.path.join(output_path,'1_'+str(name_dict[id.split('_')[0]])+ '_' + SID[thisa] + '_' + str(age) + '_' + sex + '.nii')
        print(output_name)
        sitk.WriteImage(I,output_name)




