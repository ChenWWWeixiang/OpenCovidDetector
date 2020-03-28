import os

this_path_slices='/mnt/data7/resampled_jpgs/masked_train1'
that_path_slices='/mnt/data7/resampled_jpgs/raw_train1'
for item in os.listdir(this_path_slices):
    name=item.split('_')
    thatname=name[0]+'_'+name[1]+'.nii_'+name[-1]
    if not os.path.exists(os.path.join(that_path_slices,thatname)):
        os.remove(os.path.join(this_path_slices,item))