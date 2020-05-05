import os,glob
files=['reader_cap_vs_covid.list','reader_healthy_vs_ill.list']
jpgdir='/mnt/data9/covid_detector_jpgs/training_jpgs'
for file in files:
    data=open(file,'r').readlines()
    ct=[da.split(',')[1].split('/')[-1].split('.nii')[0] for da in data]
    for act in ct:
        target=glob.glob(os.path.join(jpgdir,act+'*.jpg'))
        for item in target:
            os.remove(item)