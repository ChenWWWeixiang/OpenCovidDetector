from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta

from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPDataset,NCP2DDataset,NCPJPGDataset
import os,cv2
import numpy as np
def _validate(modelOutput, length, labels, total=None, wrong=None):

    averageEnergies = torch.mean(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        averageEnergies[i] = modelOutput[i,:length[i]].mean(0)

    maxvalues, maxindices = torch.max(averageEnergies, 1)
    #print(maxindices.cpu().numpy())
    #print(labels.cpu().numpy())
    count = 0

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        else:
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1

    return (averageEnergies, count)

class Validator():
    def __init__(self, options, mode):
        self.use_3d=options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice=options['general']['use_slice']
        if options['general']['use_slice']:
            self.validationdataset = NCPJPGDataset(options[mode]["data_root"],
                                                    options[mode]["index_root"],
                                                    options[mode]["padding"],
                                                    False)
        else:
            if options['general']['use_3d']:
                self.validationdataset = NCPDataset(options[mode]["data_root"],
                                                      options[mode]["index_root"],
                                                      options[mode]["padding"],
                                                      False)
            else:
                self.validationdataset = NCP2DDataset(options[mode]["data_root"],
                                                        options[mode]["index_root"],
                                                        options[mode]["padding"],
                                                        False)
                                                
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        self.epoch=0
        
    def __call__(self, model,epoch):
        self.epoch+=1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((2))
            if self.use_3d:
                validator_function = model.validator_function()##TODO:
            if self.use_lstm:
                validator_function = _validate
            model.eval()
            LL=[]
            if(self.usecudnn):
                net = nn.DataParallel(model).cuda()
            error_dir='view_segs/'
            os.makedirs(error_dir,exist_ok=True)
            cnt=0
            num_samples = np.zeros((2))
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
                
                model = model.cuda()

                outputs,segs = net(input)
                segs=segs.cpu().numpy()
                if self.use_3d or self.use_lstm:
                    (outputs, top1) = validator_function(outputs, length,labels)
                    _, maxindices = outputs.cpu().max(1)##vector--outputs
                else:
                    _, maxindices = outputs.cpu().max(1)
                isacc=labels.cpu().numpy().reshape(labels.size(0))==maxindices.numpy()
                output_numpy=np.exp(outputs.cpu().numpy()[:,1])
                label_numpy=labels.cpu().numpy()[:,0]
                input_numpy=input.cpu().numpy()
               # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    LL.append([output_numpy[i], label_numpy[i]])
                    if isacc[i]==1 and labels[i]==0 :
                        count[0] += 1
                    if isacc[i]==1 and labels[i]==1 :
                        count[1] += 1
                    if labels[i]==0:
                        num_samples[0] += 1
                    else:
                        num_samples[1] += 1
                    if cnt<100:
                        I=input_numpy[i,:,:,:]
                        I=(I.transpose(1,2,0)*255).astype(np.uint8)
                        J=((segs[i,:,:,:].transpose(1,2,0)>0.5)*255).astype(np.uint8)
                        J=np.concatenate([J,J,J],-1)
                        cv2.imwrite(os.path.join(error_dir,str(epoch)+'_'+str(cnt)+'.jpg'
                                                 ),np.concatenate([I,J],1))
                        cnt+=1


                #print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                #                                                                       count[0],len(self.validationdataset),
                #                                                                       1.0*count[0]/num_samples))
        print(count.sum()/num_samples.sum())
        LL=np.array(LL)
        np.save('re/'+self.mode+'records'+str(self.epoch)+'.npy',LL)
        return count/num_samples,count.sum()/num_samples.sum()
    def validator_function(self,pre,label):
        pass
