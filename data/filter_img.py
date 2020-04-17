import os

this_path_slices='/mnt/data9/covid_detector_jpgs/masked_ild'
that_path_slices='/mnt/data9/covid_detector_jpgs/raw_ild'
for item in os.listdir(this_path_slices):
    #name=item.split('_')
    thatname=item
    if not os.path.exists(os.path.join(that_path_slices,thatname)):
        os.remove(os.path.join(this_path_slices,item))