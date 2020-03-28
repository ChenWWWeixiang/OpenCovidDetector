import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.calibration import calibration_curve


#res=np.load('ipt_results/results/train.npy')
res=np.load('re/lidc.npy')
pre=res[:,0]
gt=res[:,1]
AUC=[]
for i in range(0):
    train_x, test_x, train_y, test_y = train_test_split(pre, gt, test_size=0.2)
    auc=metric.roc_auc_score(train_y,train_x)
    AUC.append(auc)
sorted_scores=np.array(AUC)
#sorted_scores.sort()
#confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
#confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
TPR=np.sum((gt==1)*(pre>0.5))/np.sum(gt==1)
TNR=np.sum((gt==0)*(pre<0.5))/np.sum(gt==0)
FPR=np.sum((gt==0)*(pre>0.5))/np.sum(gt==0)
NPV=np.sum((gt==0)*(pre<0.5))/np.sum(pre<0.5)
#print('AUC(95%CI):',np.mean(AUC),"({:0.4f} - {:0.4})".format(
#        confidence_lower, confidence_upper))
print('ACC:',metric.accuracy_score(gt,pre>0.5))
print('specificity:',TNR)
print('recall/sensitivity:',TPR)
print('PPV:',metric.precision_score(gt,pre>0.5))
print('NPV:',NPV)
print('F1:',metric.f1_score(gt,pre>0.5))
print('YOUDEN:',TPR+TNR-1)

#prob_pos = (pre - pre.min()) / (pre.max() - pre.min())

fraction_of_positives, mean_predicted_value = calibration_curve(gt, pre, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, "s-",
         label="%s" % ('1',))

#plt.hist(prob_pos, range=(0, 1), bins=10, label='1',histtype="step", lw=2)
plt.show()





fpr,tpr,threshold = metric.roc_curve(gt, pre)

plt.figure()
plt.plot(fpr, tpr) ###¼ÙÕýÂÊÎªºá×ø±ê£¬ÕæÕýÂÊÎª×Ý×ø±ê×öÇúÏß
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve for Validation Set (Slice level)')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('roc_val.jpg')

precision, recall,t=metric.precision_recall_curve(gt,pre)
plt.figure()
plt.plot(recall,precision) ###¼ÙÕýÂÊÎªºá×ø±ê£¬ÕæÕýÂÊÎª×Ý×ø±ê×öÇúÏß
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR Curve for Validation Set (Slice level)')
plt.savefig('pr_val.jpg')