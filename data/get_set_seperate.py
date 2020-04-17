import os,random,glob
path=['/home/cwx/extra/covid_project_data/cap',
      '/home/cwx/extra/covid_project_data/ild',
      '/home/cwx/extra/covid_project_data/covid',
      '/home/cwx/extra/covid_project_data/healthy']
ftrain=open('txt/3d_train.txt','w')
ftest=open('txt/3d_test.txt','w')
for onep in path:
    files=os.listdir(onep)
    names_id=[onep+'/'+file.split('_')[0]+'_'+file.split('_')[1] for file in files]
    names_id=list(set(names_id))
    random.shuffle(names_id)
    train=names_id[:len(names_id)//10*7]
    val=names_id[len(names_id)//10*7:]
    for i in train:
        names=glob.glob(i+'_*.nii')
        for name in names:
            ftrain.writelines(name+'\n')
    for i in val:
        names=glob.glob(i+'_*.nii')
        for name in names:
            ftest.writelines(name+'\n')