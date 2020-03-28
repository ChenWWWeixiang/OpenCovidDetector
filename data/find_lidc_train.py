import os,shutil
train_jpgs='/mnt/data7/resampled_jpgs/raw_train_lidc'
all_lidc='/mnt/data7/LIDC/resampled_data'
all_lidc_mask='/mnt/data7/LIDC/resampled_seg'
train_lidc='/mnt/data7/LIDC/resampled_data_train100'
train_mask='/mnt/data7/LIDC/resampled_seg_train100'
os.makedirs(train_lidc,exist_ok=True)
os.makedirs(train_mask,exist_ok=True)
filenames=os.listdir(train_jpgs)
filenames=[n.split('_')[0] for n in filenames]
filenames=list(set(filenames))
for i in filenames:
    shutil.move(os.path.join(all_lidc,i+'.nrrd'),os.path.join(train_lidc,i+'.nrrd'))
    shutil.move(os.path.join(all_lidc_mask, i + '.nrrd'), os.path.join(train_mask, i + '.nrrd'))

