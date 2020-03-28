import os,shutil,glob

i_path='/mnt/data6/lung_3rd'
o_path='/mnt/data7/multiphase_CT'
o_file='/mnt/data7/multiphase_CT.txt'
f=open(o_file,'w')
all=os.listdir(o_path)
first_name=[ss.split('_')[0] for ss in all]
names=list(set(first_name))
c2=0
c3=0
c4=0
c5=0
for na in names:
    a=glob.glob(os.path.join(o_path,na+'_*.nii'))
    if len(a)>=5:
        c5+=1
    if len(a)>=4:
        c4+=1
    if len(a)>=3:
        c3+=1
    c2+=1
    f.writelines('id='+na+', num='+str(len(a))+'\n')
print(c2,c3,c4,c5)
f.writelines('number of 2/3/4/5+ data:'+str(c2)+'_'+str(c3)+'_'+str(c4)+'_'+str(c5))