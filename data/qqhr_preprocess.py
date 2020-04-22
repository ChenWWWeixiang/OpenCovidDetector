import SimpleITK as sitk
import pydicom,os,shutil

root='/home/cwx/baidunetdiskdownload'
outpath='/mnt/data9/cap_qqhr'
os.makedirs(outpath,exist_ok=True)
reader = sitk.ImageSeriesReader()
for item in range(1,9):
    item=str(item)
    inner_dicom=os.path.join(root,item,'dicom')
    ALL_V=[]
    for dcm in os.listdir(inner_dicom):
        if not dcm[0]=='I':
            continue
        ds = pydicom.read_file(os.path.join(inner_dicom, dcm))
        id=ds['SeriesInstanceUID']._value
        os.makedirs(os.path.join(outpath,id),exist_ok=True)
        shutil.copy(os.path.join(inner_dicom, dcm),os.path.join(outpath,id,dcm))
        a=1
