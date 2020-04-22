import os,shutil
reader_root='/mnt/data9/reader_cap_vs_covid'
os.makedirs(reader_root,exist_ok=True)
data=open('reader_cap_vs_covid.list').readlines()

name=[da.split(',')[0] for da in data]
for i,j in enumerate(name):
    shutil.copy(j,os.path.join(reader_root,str(i)+'.nii'))
    a=1