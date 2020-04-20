import SimpleITK as sitk
import pydicom,os
root='/home/cwx/baidunetdiskdownload'
for item in os.listdir(root):
    inner_dicom=os.path.join(root,item,'dicom')
    ALL_V=[]
    for dcm in os.listdir(inner_dicom):
        ds = pydicom.Dataset(os.path.join(inner_dicom, dcm))

        a=1
