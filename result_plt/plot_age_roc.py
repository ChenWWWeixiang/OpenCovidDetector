import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
def get_CI(value,res):
    sorted_scores=np.array(value)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    res.append(str(np.mean(value)) + ' (' + str(confidence_lower) + '-' + str(confidence_upper) + ')')
    return res
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../re/4cls_gender.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='../saves/results_age.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name','all-AUC','all-ACC','healthy-AUC','healthy-recall','healthy-precision',
                'CAP-AUC','CAP-recall', 'CAP-precision','ILD-AUC','ILD-recall','ILD-precision',
                'COVID-AUC','COVID-recall', 'COVID-precision'])
    for a_res in ress:
        res = np.load(a_res)
        if res.shape[1]==5:
            pre=np.array(res[:,:-1],np.float)
            gt=np.array(res[:,-1],np.float)
        else:
            name=res[:, 0]
            pre = np.array(res[:, 1:-1], np.float)
            gt = np.array(res[:, -1], np.float)

        age=[int(item.split('_')[-2])//20 for item in name]
        age=np.array(age)
        for agei in range(1,4):
            AUC = []
            ACC = []
            REC = []
            PRE = []
            SAUC=[]
            y_one_hot = label_binarize(gt[age==agei], np.arange(4))
            norm_x=pre[age==agei,:]/ pre[age==agei,:].max(axis=0)
            for i in range(200):
                train_x, test_x, train_y, test_y = train_test_split(pre[age==agei,:], y_one_hot, test_size=0.2)
                train_x=train_x/train_x.max(axis=0)
                auc=metric.roc_auc_score(train_y,train_x,average='micro')
                AUC.append(auc)

                prediction=np.argmax(train_x,1)
                groundtruth=np.argmax(train_y,1)
                prediction[np.max(train_x[:,1:],1) > 0.80]=np.argmax(train_x[np.max(train_x[:,1:],1) > 0.80,1:],1) + 1
                ACC.append(np.mean(prediction==groundtruth))
                recall=[]
                precision=[]
                sauc=[]
                for cls in range(4):
                    recall.append(np.sum((prediction==cls)*(groundtruth==cls))/np.sum(groundtruth==cls))
                    precision.append(np.sum((prediction==cls)*(groundtruth==cls))/np.sum(prediction==cls))
                    sauc.append(metric.roc_auc_score(train_y[cls,:], train_x[cls,:]))
                SAUC.append(sauc)
                REC.append(recall)
                PRE.append(precision)
            PRE=np.array(PRE)
            REC=np.array(REC)
            SAUC=np.array(SAUC)
            s="{:d}-{:d}".format(agei*20+1,agei*20+20)
            Res=[s]
            Res=get_CI(AUC,Res)
            Res = get_CI(ACC, Res)
            Res = get_CI(SAUC[:, 0], Res)
            Res = get_CI(REC[:,0], Res)
            Res = get_CI(PRE[:,0], Res)
            Res = get_CI(SAUC[:, 1], Res)
            Res = get_CI(REC[:,1], Res)
            Res = get_CI(PRE[:,1], Res)
            Res = get_CI(SAUC[:, 2], Res)
            Res = get_CI(REC[:,2], Res)
            Res = get_CI(PRE[:,2], Res)
            Res = get_CI(SAUC[:, 3], Res)
            Res = get_CI(REC[:,3], Res)
            Res = get_CI(PRE[:,3], Res)
            f.writerow(Res)

            plt.figure(1)
            #fpr,tpr,threshold = metric.roc_curve(y_one_hot, norm_x)
            fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())

            plt.plot(fpr, tpr, label=s+', AUC={:.2f}'.format(metric.auc(fpr, tpr)))

plt.figure(1)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('../saves/roc_mc.jpg')
plt.show()
