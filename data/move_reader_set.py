import os,shutil
reader_root='/mnt/data9/reader_cap_vs_covid'
os.makedirs(reader_root,exist_ok=True)
data=open('reader_cap_vs_covid.list').readlines()
idx=[da.split(',')[0] for da in data]
name=[da.split(',')[1][:-1] for da in data]
for i,j in zip(name,idx):
    shutil.copy(i,os.path.join(reader_root,j+'.nii'))
    a=1