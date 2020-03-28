import os,cv2
import numpy as np
import SimpleITK as sitk
os.makedirs('check_jpgs',exist_ok=True)
pre_data=open('val_slices_count.txt','r').readlines()
fix='/mnt/data7/slice_test_seg/data_re'
ma='/mnt/data7/slice_test_seg/mask_re'
path='/mnt/data7/slice_test_seg/seg_re'
name=[pre.split('\t')[0].split('data_re/')[-1] for pre in pre_data]
pred=[pre.split('\t')[1] for pre in pre_data]
pred=[np.array(pr[1:-1].split(','),np.float) for pr in pred]
slice_name=[pre.split('\t')[-1] for pre in pre_data]
slice_name=[np.array(pr[1:-2].split(','),np.int) for pr in slice_name]
GT=[]
PR=[]
for pre,sli,idn in zip(pred,slice_name,name):
    pre=pre[:len(sli)]

    try:
        data = sitk.ReadImage(os.path.join(path, idn))
        data=sitk.GetArrayFromImage(data)

    except:
        continue

    GG=np.where(data.sum(1).sum(1)>0)[0]
    GG=GG[5:-5]
    cc = 0
    for a_p,a_n in zip(pre,sli):
      #  name_jpg=fix+idn+'_'+str(
      # a_n)+'.jpg'
        #name_jpg2 = fix + idn[:-4] + '_' + str(a_n) + '.jpg'
        #a_gt=os.path.exists(name_jpg) or os.path.exists(name_jpg2)
        a_pr=a_p

        a_gt=np.sum(GG==a_n)
        if not a_gt==(a_p>0.5):
            #t= sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, idn)))[a_n, :, :]
            tt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(fix, idn)))[a_n, :, :]
            tt=(tt+1200)*255/1900
            cc+=1
            mask=sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ma, idn)))[a_n, :, :]*255

            #cv2.imwrite('t.jpg',t*255)
            cv2.imwrite('check_jpgs/'+idn+'_'+str(cc)+'.jpg', tt)
            a=1

        GT.append(a_gt)
        PR.append(a_pr)
    if cc > 10:
        print(idn, cc)
        #print(name_jpg,a_gt)
import sklearn.metrics as metric
train_y=np.array(GT)
train_x=np.array(PR)
auc=metric.roc_auc_score(train_y,train_x)
print('AUC',auc)
acc=metric.accuracy_score(train_y, train_x > 0.5),
tnr=np.sum((train_y == 0) * (train_x < 0.5)) / np.sum(train_y == 0)
tpr=np.sum((train_y==1)*(train_x>0.5))/np.sum(train_y==1)
npv = np.sum((train_y == 0) * (train_x < 0.5)) / np.sum(train_x < 0.5)
print('TNR',tnr)
print('TPR',tpr)
prec=metric.precision_score(train_y, train_x > 0.5)
print('PPV',prec)
print('NPV',npv)
print('F1',metric.f1_score(train_y, train_x > 0.5))
print('YOUDEN',tpr + tnr - 1)
print('ACC',acc[0])
fpr, tpr, threshold = metric.roc_curve(train_y, train_x)
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()