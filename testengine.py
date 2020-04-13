from torch.autograd import Variable
import torch
import time
import torch.optim as optim
from datetime import datetime, timedelta
#from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPDataset, NCP2DDataset, NCPJPGDataset,NCPJPGtestDataset
import os, cv2
import toml
from models.net2d import densenet121,densenet161,resnet152,resnet152_plus
import numpy as np
#from models.g_cam import GuidedPropo
import matplotlib as plt
KEEP_ALL=True
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--maskpath", help="A list of paths for lung segmentation data",#  type=list,
                    default=['/mnt/data6/CAP/seg_test',
                             '/mnt/data7/ILD/resampled_seg',
                            #'/mnt/data7/examples/seg',
                            #'/mnt/data7/reader_ex/resampled_seg',
                            '/mnt/data7/LIDC/resampled_seg',
                            '/mnt/data7/resampled_seg/test1', '/mnt/data7/resampled_seg/test2',
                            '/mnt/data7/resampled_seg/test3'
                            #'/mnt/data7/slice_test_seg/mask_re',
                                   # '/mnt/data7/resampled_seg/test3']
                            ])
parser.add_argument("-i", "--imgpath", help="A list of paths for image data",
                    default=['/mnt/data6/CAP/data_test',
                        '/mnt/data7/ILD/resampled_data',
                        #'/mnt/data7/examples/data',
                        #'/mnt/data7/reader_ex/resampled_data',
                        '/mnt/data7/LIDC/resampled_data',
                        '/mnt/data7/resampled_data/test1','/mnt/data7/resampled_data/test2',
                        '/mnt/data7/resampled_data/test3'
                        #'/mnt/data7/slice_test_seg/data_re',
                             #'/mnt/data7/resampled_data/resampled_test_3']
                        ])
parser.add_argument("-o", "--savenpy", help="A path to save record",  type=str,
                    default='re/5cls.npy')
parser.add_argument("-e", "--exclude_list", help="A path to a txt file for excluded data list. If no file need to be excluded, "
                                                 "it should be 'none'.",  type=str,
                    default='none')
parser.add_argument("-v", "--invert_exclude", help="Whether to invert exclude to include",  type=bool,
                    default=False)
parser.add_argument("-p", "--model_path", help="Whether to invert exclude to include",  type=str,
                    default='weights/model_5cls.pt')
parser.add_argument("-g", "--gpuid", help="gpuid",  type=str,
                    default='2')
args = parser.parse_args()


def _validate(modelOutput, labels, topn=1):
    modelOutput=list(np.exp(modelOutput.cpu().numpy())[:,-1])#for covid19
    pos_count=np.sum(np.array(modelOutput)>0.5)
    modelOutput.sort()
    averageEnergies = np.mean(modelOutput[-topn:])
    iscorrect = labels.cpu().numpy()==(averageEnergies>0.5)
    return averageEnergies,iscorrect,pos_count

def _validate_multicls(modelOutput, labels, topn=3):
    averageEnergies=[]
    for i in range(0,options['general']['class_num']):
        t = np.exp(modelOutput.cpu().numpy())[:, i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t))
        else:
            averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred

    return averageEnergies, iscorrect,pred
def _voting_validate_multicls(modelOutput, labels, topn=3):
    averageEnergies=np.exp(modelOutput.cpu().numpy()).mean(0)
    #cnt=np.zeros([options['general']['class_num']])
    idx_max=np.argmax(modelOutput.cpu().numpy(),axis=1)
    cnt=np.bincount(idx_max)
    pred=np.argmax(cnt)
    if pred==0:
        if cnt[0]>0.9*idx_max.shape[0]:
            pred=0
        else:
            pred=np.argmax(cnt[1:])+1
    iscorrect = labels.cpu().numpy() == pred
    return averageEnergies.tolist(), iscorrect,pred


class Validator():
    def __init__(self, options, mode):
        self.cls_num=options['general']['class_num']
        self.use_plus=options['general']['use_plus']
        self.use_3d = options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice = options['general']['use_slice']
        datalist = args.imgpath
        masklist =args.maskpath
        self.savenpy = args.savenpy
        if not args.exclude_list=="none":
            f=open(args.exclude_list,'r')
            #f = open('data/txt/val_list.txt', 'r')
            f=f.readlines()
            f=[da.split('\t')[-1] for da in f]
            if self.use_plus:
                self.validationdataset = NCPJPGtestDataset(datalist,
                                                           masklist,
                                                           options[mode]["padding"],
                                                           f,1-args.invert_exclude,age_list='all_ages_genders.txt',
                                                           cls_num=self.cls_num)
            else:
                self.validationdataset = NCPJPGtestDataset(datalist,
                                                           masklist,
                                                           options[mode]["padding"],f,1-args.invert_exclude,
                                                           cls_num=self.cls_num)
        else:
            if self.use_plus:
                self.validationdataset = NCPJPGtestDataset(datalist,
                                                           masklist,
                                                           options[mode]["padding"],cls_num=self.cls_num,age_list='all_ages_genders.txt',
                                                           )
            else:
                self.validationdataset = NCPJPGtestDataset(datalist,
                                                           masklist,
                                                           options[mode]["padding"],cls_num=self.cls_num
                                                           )
        self.topk=3
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
            self.validationdataset,
            batch_size=1,
            shuffle=options["input"]["shuffle"],
            num_workers=options["input"]["numworkers"],
            drop_last=False
        )
        self.mode = mode
        self.epoch = 0

    def __call__(self, model):
        self.epoch += 1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((self.cls_num + self.use_plus * 2))
            Matrix = np.zeros((self.cls_num, self.cls_num))
            if self.cls_num>2:
                validator_function=_validate_multicls
            else:
                validator_function = _validate
            model.eval()
            LL = []
            GG=[]
            AA=[]
            if (self.usecudnn):
                net = nn.DataParallel(model).cuda()
            #error_dir = 'error/'
            #os.makedirs(error_dir, exist_ok=True)
            cnt = 0
            cnt_for_wh=0
            cnt_for_lidc=0
            e_cnt_l=0
            e_cnt_w=0
            num_samples = np.zeros((self.cls_num + self.use_plus * 2))
            #elist=open('val_slices_count.txt','w+')
            #truth_list=truth_list.readlines()
            #names=[tl.split('\t')[0] for tl in truth_list]
            #cls = [int(tl.split('\t')[1]) for tl in truth_list]
            tic=time.time()
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                if self.use_plus:
                    age = Variable(sample_batched['age']).cuda()
                    gender = Variable(sample_batched['gender']).cuda()
                name =sample_batched['length'][0]
                slice_idx=sample_batched['length'][1]

                model = model.cuda()
                input=input.squeeze(0)
                input=input.permute(1,0,2,3)
                if not self.use_plus:
                    outputs = net(input)
                else:
                    outputs, out_gender, out_age = net(input)
                if KEEP_ALL:
                    all_numpy=np.exp(outputs.cpu().numpy()[:,1]).tolist()
                    a=1
                (vector, isacc,pos_count) = validator_function(outputs, labels,self.topk)
                _, maxindices = outputs.cpu().max(1)
                if self.use_plus:
                    _, maxindices_gender = out_gender.cpu().mean(0).max(0)
                    genderacc = gender.cpu().numpy().reshape(gender.size(0)) == maxindices_gender.numpy()
                    output_gender_numpy = np.exp(out_gender.cpu().numpy()[:, 1]).mean()
                    gender_numpy=gender.cpu().numpy()
                    age_numpy = age.cpu().numpy().reshape(age.size(0))
                    pre_age_numpy = (np.exp(out_age.cpu().numpy()) * np.array([10, 30, 50, 70, 90])).sum(1).mean()
                    #ages_mse,oa=self.age_function(out_age, age)
                #_, maxindices = vector.cpu().max(1)  ##vector--outputs

                output_numpy = vector
                label_numpy = labels.cpu().numpy()[0, 0]
                ####

                ####
                #LL.append([name[0],output_numpy, label_numpy])

                if self.use_plus and gender_numpy[0]>-1:
                    print(name[0], isacc, vector, 'sex:', genderacc)
                else:
                    print(name[0],isacc,vector)
                # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    if self.cls_num>2:
                        LL.append([name[0]]+ output_numpy+[label_numpy])
                    else:
                        LL.append([name[0],output_numpy,label_numpy])
                    Matrix[label_numpy, pos_count] += 1
                    #if isacc[i]==0:
                    #elist.writelines(name[0]+'\t'+str(all_numpy)+'\t'+str(pos_count)+'\t'+str(np.array(slice_idx).tolist())+'\n')
                    if isacc[i] == 1:
                        count[labels[i]] += 1
                    num_samples[labels[i]] += 1

                    if self.use_plus and gender_numpy[i]>-1:
                        GG.append([output_gender_numpy,gender_numpy])
                        AA.append(np.abs(pre_age_numpy-age_numpy))
                        if genderacc[i]==1 :
                            count[gender[i]+self.cls_num] += 1
                        num_samples[gender[i]+self.cls_num] += 1
                # print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                #                                                                       count[0],len(self.validationdataset),
                #                                                                       1.0*count[0]/num_samples))
        print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(),np.mean(AA))
        LL = np.array(LL)
        print(Matrix)
        np.save(self.savenpy, LL)
        if self.use_plus:
            GG = np.array(GG)
            AA=np.array(AA)
            np.save('gender.npy', GG)
            np.save('age.npy', AA)
        toc=time.time()
        print((toc-tic)/self.validationdataloader.dataset.__len__())
        return count / num_samples, count[:self.cls_num].sum() / num_samples[:self.cls_num].sum()

    def age_function(self, pre, label):
        pre=pre.cpu().numpy().mean()* 90
        label=label.cpu().numpy()
        return np.mean(pre-label),pre


print("Loading options...")
with open('test.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
if isinstance(args.imgpath,str):
    args.imgpath=eval(args.imgpath)
    args.maskpath=eval(args.maskpath)
torch.manual_seed(options["general"]['random_seed'])

# Create the model.
if options['general']['use_plus']:
    model = resnet152_plus(options['general']['class_num'])
else:
    model = resnet152(options['general']['class_num'])

pretrained_dict = torch.load(args.model_path)
# load only exists weights
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                   k in model_dict.keys() and v.size() == model_dict[k].size()}
print('matched keys:', len(pretrained_dict))
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

tester = Validator(options, 'test')

result, re_all = tester(model)
print (tester.savenpy)
print('-' * 21)
print('All acc:' + str(re_all))
print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
for i in range(result.shape[0]):
    print('{:<10}|{:>10}'.format(i, result[i]))
print('-' * 21)

