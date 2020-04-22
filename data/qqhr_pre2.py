import SimpleITK as sitk
import pydicom,os,shutil

root='/mnt/data9/cap_qqhr'
output_path='/home/cwx/extra/covid_project_data/cap_qqhr'
os.makedirs(output_path,exist_ok=True)
cnt=1
IDs=[]
reader = sitk.ImageSeriesReader()
for dir in os.listdir(root):
    inners=os.listdir(os.path.join(root,dir))
    if len(inners)<100:
        shutil.rmtree(os.path.join(root,dir))
    else:
        try:
            case_path = os.path.join(root,dir)
            dicom_names = reader.GetGDCMSeriesFileNames(case_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        except:
            continue
        ds=pydicom.read_file(os.path.join(root,dir, inners[0]))

        date = ds['StudyDate'].value
        name=ds['PatientID'].value
        if  name in IDs:
            shutil.rmtree(os.path.join(root, dir))
            continue
        IDs.append(name)
        try:
            age = int(ds['PatientAge'].value[:-1])
            sex = ds['PatientSex'].value
        except:
            try:
                age = int(ds['StudyDate'].value[:4]) - int(ds['PatientBirthDate'].value[:4])
                sex = ds['PatientSex'].value
            except:
                continue
        output_name = os.path.join(output_path,
                                   '1_' + str(cnt) + '_' + date + '_' + str(age) + '_' + sex + '.nii')
        cnt+=1
        print(output_name)
        sitk.WriteImage(image, output_name)