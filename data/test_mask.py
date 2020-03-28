import cv2,os
import numpy as np
import SimpleITK as sitk
from skimage import measure
input_path='/mnt/data6/lung_resample_jpgs'
output_path='/mnt/data6/lung_resample_masked'
#img_path='/mnt/data6/lung_data'
os.makedirs(output_path,exist_ok=True)
for name in os.listdir(input_path):
    I=cv2.imread(os.path.join(input_path,name))
    J=((I>10)*(I<100)).astype(np.uint8)*255
    testa1 = measure.label(J, connectivity=1)
    props = measure.regionprops(testa1)
    #props=[pre for pre in props if pre.area>200 ]
    MAYBE=[]
    for k,pr in enumerate(props):
        if pr.bbox[0]<10 or pr.bbox[1]<10:
            continue
        if pr.bbox[3]>I.shape[0]-10 or pr.bbox[4]>I.shape[1]-10:
            continue
        if pr.area<200:
            continue
        MAYBE.append(pr.bbox)
        #print(pr.bbox,pr.area)
    if len(MAYBE)>=2:
        x=np.min([MAYBE[0][:3],MAYBE[1][:3]],0)
        T=np.max([MAYBE[0][-3:],MAYBE[1][-3:]],0)
        temp=I[x[0]-10:T[0]+10,x[1]-10:T[1]+10,:]
        if T[0]-x[0]<100:
            temp=I
        if T[1]-x[1]<170:
            temp=I
    else:
        temp=I
    cv2.imwrite(os.path.join(output_path,name),temp)