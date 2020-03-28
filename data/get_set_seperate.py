import os,random,glob
path='/mnt/data6/lung_resample_npy'
files=os.listdir(path)
names_id=[file.split('_')[0] for file in files]
names_id=list(set(names_id))
random.shuffle(names_id)
train=names_id[:len(names_id)//2]
val=names_id[len(names_id)//2:-len(names_id)//4]
test=names_id[-len(names_id)//4:]
f=open('2d_train.txt','w')
for i in train:
    names=glob.glob(path+'/'+i+'_*')
    for name in names:
        f.writelines(name+'\n')
f=open('2d_val.txt','w')
for i in val:

    names=glob.glob(path+'/'+i+'_*')
    for name in names:
        f.writelines(name+'\n')
f=open('2d_test.txt','w')
for i in test:
    names=glob.glob(path+'/'+i+'_*')
    for name in names:
        f.writelines(name+'\n')