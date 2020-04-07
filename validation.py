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
        self.cls_num=options['general']['class_num']
        self.use_plus = options['general']['use_plus']
        self.use_3d=options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice=options['general']['use_slice']
        if options['general']['use_slice']:
            self.validationdataset = NCPJPGDataset(options[mode]["data_root"],
                                                    options[mode]["index_root"],
                                                    options[mode]["padding"],
                                                    False,cls_num=self.cls_num)
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
        
    def __call__(self, model):
        self.epoch+=1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((3))
            if self.use_3d:
                validator_function = model.validator_function()##TODO:
            if self.use_lstm:
                validator_function = _validate
            model.eval()
            LL=[]
            GG=[]
            AA=0
            if(self.usecudnn):
                net = nn.DataParallel(model).cuda()
            error_dir='error/'
            os.makedirs(error_dir,exist_ok=True)
            cnt=0
            num_samples = np.zeros((3))
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
                if self.use_plus:
                    age = Variable(sample_batched['age']).cuda()
                    gender = Variable(sample_batched['gender']).cuda()
                model = model.cuda()

                if not self.use_plus:
                    outputs = net(input)
                else:
                    outputs, out_gender, out_age = net(input)
                if self.use_3d or self.use_lstm:
                    (outputs, top1) = validator_function(outputs, length,labels)
                    _, maxindices = outputs.cpu().max(1)##vector--outputs
                elif self.use_plus:
                    _, maxindices = outputs.cpu().max(1)
                    _, maxindices_gender = out_gender.cpu().max(1)
                    _, maxindices_age = out_age.cpu().max(1)
                    #pre_ages = out_age.cpu()*90
                    genderacc = gender.cpu().numpy().reshape(gender.size(0)) == maxindices_gender.numpy()
                    ageacc = age.cpu().numpy().reshape(age.size(0)) == maxindices_age.numpy()
                    #ages_mse=np.abs(pre_ages.numpy()-age.cpu().numpy())
                    output_gender_numpy = np.exp(out_gender.cpu().numpy()[:, 1])
                    gender_numpy = gender.cpu().numpy()[:, 0]

                    #out_age_numpy = np.exp(out_age.cpu().numpy()[:, 1])
                    #age_numpy = age.cpu().numpy()[:, 0]

                else:
                    _, maxindices = outputs.cpu().max(1)
                isacc=labels.cpu().numpy().reshape(labels.size(0))==maxindices.numpy()
                output_numpy=np.exp(outputs.cpu().numpy())
                label_numpy=labels.cpu().numpy()[:,0]

               # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    LL.append([output_numpy[i], label_numpy[i,:]])
                    if isacc[i]==1 :
                        count[labels[i]] += 1
                    num_samples[labels[i]]+=1
                    if isacc[i]==0 and False:
                        input=input.cpu().numpy()
                        for b in range(input.shape[2]):
                            I=(input[0,:,b,:,:].transpose(1,2,0)*255).astype(np.uint8)
                            cv2.imwrite(os.path.join(error_dir,'error'+str(cnt)+'_truecls:'+str(label_numpy)+'_slice:'+str(b)+'.jpg'
                                                     ),I)
                        cnt+=1
                    if self.use_plus and gender_numpy[i]>-1:
                        GG.append([output_gender_numpy[i],gender_numpy[i]])
                        #AA.append()
                        if genderacc[i]==1 and gender[i]==0 :
                            count[2] += 1
                        if genderacc[i]==1 and gender[i]==1 :
                            count[3] += 1
                        if gender[i]==0:
                            num_samples[2] += 1
                        else:
                            num_samples[3] += 1


                #print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                #                                                                       count[0],len(self.validationdataset),
                #                                                                       1.0*count[0]/num_samples))
        print(count[:3].sum()/num_samples[:3].sum())
        LL=np.array(LL)
        np.save('re/' + self.mode + 'records' + str(self.epoch) + '.npy', LL)
        if self.use_plus:
            GG=np.array(GG)
            np.save('re/' + self.mode + 'gender_records' + str(self.epoch) + '.npy', GG)

        return count/num_samples,count[:2].sum()/num_samples[:2].sum()
    def validator_function(self,pre,label):
        pass
