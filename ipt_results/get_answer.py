import os
data=open('result.txt','r').readlines()
op=open('answer.txt','w')
fl=[da.split('\t')[0].split('.nii')[0] for da in data]
sl=['/mnt/data7/resample_data/test'+da.split('test')[-1][0]+
    '/'+da.split('test')[-1].split('_')[1]+'_'+da.split('test')[-1].split('_')[2] for da in data]
for f,s in zip(fl,sl):
    op.writelines(f+'\t'+s+'\n')
