import os,shutil,glob

i_path='/home/cwx/extra/NCP_ill3'
o_path='/mnt/data7/multiphase_CT_dicom'
#o_file='/mnt/data7/multiphase_CT.txt'

for item in os.listdir(i_path):
    phases=os.listdir(os.path.join(i_path,item))
    if len(phases)>1:
       # for phase in phases:
        newitem=str(int(item)+200)
        shutil.copytree(os.path.join(i_path,item),os.path.join(o_path,newitem))