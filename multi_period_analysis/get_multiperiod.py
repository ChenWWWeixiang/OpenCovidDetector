import pydicom,os
import numpy as np
import csv,json
from pydicom.data import get_testdata_files
import SimpleITK as sitk

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)


outpath='/mnt/data9/mp_NCPs'
os.makedirs(outpath,exist_ok=True)
reader = sitk.ImageSeriesReader()
recording=open('multi_period_datas.txt','w')
Patient=dict()
for idroot in range(1,8):
    root = '/home/cwx/extra/NCP_CTs/NCP_ill'+str(idroot)
    for patient in os.listdir(root):
        for case in os.listdir(os.path.join(root,patient)):
            if not os.path.isdir(os.path.join(root, patient, case)):
                continue
            for phase in os.listdir(os.path.join(root,patient,case)):
                if not os.path.isdir(os.path.join(root, patient, case, phase)):
                    continue
                for inner in os.listdir(os.path.join(root,patient,case,phase)):
                    if not os.path.isdir(os.path.join(root,patient,case,phase,inner)):
                        continue
                    adicom=os.listdir(os.path.join(root,patient,case,phase,inner))
                    adicom=[a for a in adicom if a[0]=='I']
                    if len(adicom)>2:
                        adicom=adicom[0]
                        ds = pydicom.read_file(os.path.join(root,patient,case,phase,inner,adicom))
                        date=ds['StudyDate'].value
                        name=ds['PatientName'].value
                        try:
                            age = int(ds['PatientAge'].value[:-1])
                            sex = ds['PatientSex'].value
                        except:
                            age = 2020 - int(ds['PatientBirthDate'].value[:4])
                            sex = ds['PatientSex'].value
                        if not name in Patient.keys():
                            Patient[name]=dict()
                            Patient[name]['gender']=sex
                            Patient[name]['age'] = age
                            Patient[name]['serials']=[date]
                        else:
                            Patient[name]['serials'].append(date)

                        dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(root,patient,case,phase,inner))
                        reader.SetFileNames(dicom_names)
                        image = reader.Execute()
                        outfile=os.path.join(outpath,str(idroot)+'_'+patient+'_'+date[5:]+'.nii')
                        sitk.WriteImage(image,outfile)

save_dict('mp_p.json',Patient)




