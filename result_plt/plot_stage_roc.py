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
def plot_a_group(s,gt,pre):
    AUC = []
    ACC = []
    REC = []
    PRE = []
    SAUC = []
    y_one_hot = label_binarize(gt, np.arange(4))
    norm_x = pre / pre.max(axis=0)
    for i in range(200):
        train_x, test_x, train_y, test_y = train_test_split(pre, y_one_hot, test_size=0.2)
        train_x = train_x / train_x.max(axis=0)
        auc = metric.roc_auc_score(train_y, train_x, average='micro')
        AUC.append(auc)

        prediction = np.argmax(train_x, 1)
        groundtruth = np.argmax(train_y, 1)
        prediction[np.max(train_x[:, 1:], 1) > 0.80] = np.argmax(train_x[np.max(train_x[:, 1:], 1) > 0.80, 1:], 1) + 1
        ACC.append(np.mean(prediction == groundtruth))
        recall = []
        precision = []
        sauc = []
        for cls in range(4):
            recall.append(np.sum((prediction == cls) * (groundtruth == cls)) / np.sum(groundtruth == cls))
            precision.append(np.sum((prediction == cls) * (groundtruth == cls)) / np.sum(prediction == cls))
            sauc.append(metric.roc_auc_score(train_y[cls, :], train_x[cls, :]))
        SAUC.append(sauc)
        REC.append(recall)
        PRE.append(precision)
    PRE = np.array(PRE)
    REC = np.array(REC)
    SAUC = np.array(SAUC)
    Res = [s]
    Res = get_CI(AUC, Res)
    Res = get_CI(ACC, Res)
    Res = get_CI(SAUC[:, 0], Res)
    Res = get_CI(REC[:, 0], Res)
    Res = get_CI(PRE[:, 0], Res)
    Res = get_CI(SAUC[:, 1], Res)
    Res = get_CI(REC[:, 1], Res)
    Res = get_CI(PRE[:, 1], Res)
    Res = get_CI(SAUC[:, 2], Res)
    Res = get_CI(REC[:, 2], Res)
    Res = get_CI(PRE[:, 2], Res)
    Res = get_CI(SAUC[:, 3], Res)
    Res = get_CI(REC[:, 3], Res)
    Res = get_CI(PRE[:, 3], Res)
    f.writerow(Res)
    plt.figure(1)
    # fpr,tpr,threshold = metric.roc_curve(y_one_hot, norm_x)
    fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())
    plt.plot(fpr, tpr, label=s + ', AUC={:.2f}'.format(metric.auc(fpr, tpr)))
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../key_result/test.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='../saves/results_stage.csv')
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

        time=[int(item.split('_')[-3][-4:]) for item in name]
        time=np.array(time)
        person=[item.split('/')[-2]+'/'+item.split('/')[-1].split('_')[0]+'_'+
                item.split('/')[-1].split('_')[1] for item in name]

        unit_person=list(set(person))
        STAGEI=[]
        STAGEII=[]
        STAGE_MEAN=[]
        person=np.array(person)
        unit_person=np.array(unit_person)
        cnt2=0
        for iperson in unit_person:
            this_idx=np.where(person==iperson)[0]
            this_time=time[this_idx]
            if len(this_time)>=2:
                cnt2+=1
            else:
                continue
            sorted_idx=this_idx[np.argsort(this_time)]
            STAGEI.append([pre[sorted_idx[0],:],gt[sorted_idx[0]]])
            STAGEII.append([pre[sorted_idx[1], :], gt[sorted_idx[1]]])
            STAGE_MEAN.append([np.mean(pre[sorted_idx, :],0), gt[sorted_idx[1]]])
        print(cnt2)
        STAGEI=np.array(STAGEI)
        STAGEII=np.array(STAGEII)
        STAGE_MEAN=np.array(STAGE_MEAN)
        plot_a_group('stage I', np.array(STAGEI[:,1],np.int),np.stack(STAGEI[:,0]))
        plot_a_group('stage II', np.array(STAGEII[:,1],np.int), np.stack(STAGEII[:,0]))
        plot_a_group('stage mix', np.array(STAGE_MEAN[:, 1], np.int), np.stack(STAGE_MEAN[:, 0]))


plt.figure(1)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('../saves/roc_3stage.jpg')
plt.show()
