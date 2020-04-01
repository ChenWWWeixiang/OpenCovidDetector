import os,random,glob
#path='/mnt/data7/lung_jpgs_with_SEG'
#path=['/mnt/data7/resampled_jpgs/masked_train1',
#      '/mnt/data7/resampled_jpgs/masked_train2',
#      '/mnt/data7/resampled_jpgs/masked_train_lidc',
#      '/mnt/data7/resampled_jpgs/masked_train3',
#      '/mnt/data7/resampled_jpgs/masked_ild']
path=['/mnt/data9/new_seg_set/masked_jpgs1','/mnt/data9/new_seg_set/masked_jpgs2',
      '/mnt/data7/resampled_jpgs/masked_train_lidc','/mnt/data7/resampled_jpgs/masked_train3',
      '/mnt/data7/resampled_jpgs/masked_ild']
#path=['/mnt/data7/slice_test_seg/jpgs2']
f1 = open('txt/train_xzw.txt', 'w')
f2 = open('txt/test_xzw.txt', 'w')
train_count=5000
c=0
for ipath in path:
    cnt = 0
    files=os.listdir(ipath)
    names_id=[file.split('_')[0] for file in files]
    names_id=list(set(names_id))
    random.shuffle(names_id)
    train=names_id[:-len(names_id)//4]
    #val=names_id[len(names_id)//2:-len(names_id)//4]
    test=names_id[-len(names_id)//4:]
    for _,i in enumerate(train):
        if cnt >= train_count:
            break
        names=glob.glob(ipath+'/'+i+'_*')
        for name in names:
            if cnt>=train_count:
                break
            cnt+=1
            c+=1
            f1.writelines(name+'\n')
#print(c)
    for i in test:
        names=glob.glob(ipath+'/'+i+'_*')
        for name in names:
            f2.writelines(name+'\n')