from torch.autograd import Variable
import torch
import time
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPDataset, NCP2DDataset, NCPJPGDataset,NCPJPGtestDataset,NCPJPGtestDataset_MHA
import os, cv2
import toml
from models.net2d import densenet121,densenet161,resnet152
import numpy as np
#from models.g_cam import GuidedPropo
import matplotlib as plt
KEEP_ALL=True
def _validate(modelOutput, labels, topn=1):
    modelOutput=list(np.exp(modelOutput.cpu().numpy())[:,1])
    pos_count=np.sum(np.array(modelOutput)>0.5)
    modelOutput.sort()
    averageEnergies = np.mean(modelOutput[-topn:])
    iscorrect = labels.cpu().numpy()==(averageEnergies>0.5)
    return averageEnergies,iscorrect,pos_count
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--maskpath", help="A list of paths for lung segmentation data",  type=list,
                    default=['/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill1',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill2',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill3',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill4',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill5',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill6',
                             '/mnt/data7/NCP_mp_CTs/reg/lungsegs/NCP_ill7'])
parser.add_argument("-i", "--imgpath", help="A list of paths for image data",  type=list,
                    default=['/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill1',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill2',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill3',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill4',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill5',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill6',
                             '/mnt/data7/NCP_mp_CTs/reg/images/NCP_ill7'])
parser.add_argument("-o", "--savenpy", help="A path to save record",  type=str,
                    default='scores_mp.npy')
parser.add_argument("-p", "--modelpath", help="Whether to invert exclude to include",  type=str,
                    default='../saves/trained_model.pt')
parser.add_argument("-g", "--gpuid", help="gpuid",  type=str,
                    default='2')
args = parser.parse_args()

class Validator():
    def __init__(self, options, mode):
        self.use_3d = options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice = options['general']['use_slice']
        datalist=args.imgpath
        #datalist=[os.path.join('/mnt/data7/filtered_mp_CT/crop_images',da) for da in datalist]
        masklist = args.maskpath
        #masklist = [os.path.join('/mnt/data7/filtered_mp_CT/crop_lungsegs', da) for da in masklist]
        self.savenpy = args.savenpy

        self.validationdataset = NCPJPGtestDataset_MHA(datalist,
                                                   masklist,
                                                   options[mode]["padding"],
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
            count = np.zeros((2))
            validator_function = _validate
            model.eval()
            LL = []
            if (self.usecudnn):
                net = nn.DataParallel(model).cuda()
            error_dir = 'error/'
            os.makedirs(error_dir, exist_ok=True)
            cnt = 0
            cnt_for_wh=0
            cnt_for_lidc=0
            e_cnt_l=0
            e_cnt_w=0
            num_samples = np.zeros((2))
            elist=open('val_slices_count.txt','w+')
            #truth_list=truth_list.readlines()
            #names=[tl.split('\t')[0] for tl in truth_list]
            #cls = [int(tl.split('\t')[1]) for tl in truth_list]
            tic=time.time()
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                name =sample_batched['length'][0]
                slice_idx=sample_batched['length'][1]

                #model = model.cuda()
                input=input.squeeze(0)
                input=input.permute(1,0,2,3)
                outputs = net(input)
                if KEEP_ALL:
                    all_numpy=np.exp(outputs.cpu().numpy()[:,1]).tolist()
                    a=1
                (vector, isacc,pos_count) = validator_function(outputs, labels,self.topk)
                #_, maxindices = vector.cpu().max(1)  ##vector--outputs

                output_numpy = vector
                label_numpy = labels.cpu().numpy()[0, 0]
                ####

                ####
                LL.append([name[0],output_numpy, label_numpy])
                print(name[0],isacc,vector)
                # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    #if isacc[i]==0:
                    elist.writelines(name[0]+'\t'+str(all_numpy)[1:-1]+'\t'+str(pos_count)+'\t'+str(np.array(slice_idx).tolist())[1:-1]+'\n')
                    if isacc[i] == 1 and labels[i] == 0:
                        count[0] += 1
                    if isacc[i] == 1 and labels[i] == 1:
                        count[1] += 1
                    if labels[i] == 0:
                        num_samples[0] += 1
                    else:
                        num_samples[1] += 1
                    if isacc[i] == 0 and False:
                        input = input.cpu().numpy()
                        for b in range(input.shape[2]):
                            I = (input[0, :, b, :, :].transpose(1, 2, 0) * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(error_dir, 'error' + str(cnt) + '_truecls:' + str(
                                label_numpy) + '_slice:' + str(b) + '.jpg'
                                                     ), I)
                        cnt += 1

                # print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                #                                                                       count[0],len(self.validationdataset),
                #                                                                       1.0*count[0]/num_samples))
        print(count.sum() / num_samples.sum())
        LL = np.array(LL)
        np.save(self.savenpy, LL)
        toc=time.time()
        print((toc-tic)/200)
        return count / num_samples, count.sum() / num_samples.sum()

    def validator_function(self, pre, label):
        pass

print("Loading options...")
with open('get_abnormal_slices.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

torch.manual_seed(options["general"]['random_seed'])

# Create the model.
model = resnet152(2)

pretrained_dict = torch.load(args.modelpath)
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
for i in range(2):
    print('{:<10}|{:>10}'.format(i, result[i]))
print('-' * 21)

