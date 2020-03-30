from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data.dataset import NCPDataset,NCP2DDataset,NCPJPGDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import pdb
import math
class NLLSequenceLoss(torch.nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """

    def __init__(self,w=[0.55, 0.45]):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = torch.nn.NLLLoss(reduction='none', weight=torch.Tensor(w).cuda())

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target).unsqueeze(1))
        loss = torch.cat(loss, 1)
        # print('loss:',loss)
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, L - 1] = 1.0
        # print('mask:',mask)
        # print('mask * loss',mask*loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss
class Cir_loss(torch.nn.Module):
    def __init__(self,lamda=[0.5,0.3,0.2], w=[0.4, 0.5]):
        super(Cir_loss, self).__init__()
        self.lamda=lamda
        self.cls_1st = torch.nn.NLLLoss(reduction='none', weight=torch.Tensor(w).cuda())
    def forward(self, output,gt,output_r,output_inv,mask,mask_input):
        l1=self.cls_1st(output,gt)+(self.cls_1st(output_r,gt)*0.5+self.cls_1st(output_inv,1-gt)*0.8)*gt
        l2=0
        l3=0
        for i in range(mask.shape[0]):
            if gt[i]==1:
                l2+=torch.sum((mask[i,:,:,:]>0.5)*1.0)/224/224
                temp=mask_input[i,:,:,:]
                l3+=torch.std(temp[:,mask[i,0,:,:]>0.5])
        return self.lamda[0]*l1.mean()+self.lamda[1]*l2+self.lamda[2]*l3
def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)

def output_iteration(loss, i, time, totalitems):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    
    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time), timedelta_string(estTime),loss))

class Trainer():

    tot_iter = 0
    writer = SummaryWriter()    
    
    def __init__(self, options):
        self.use_slice = options['general']['use_slice']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_3d=options['general']['use_3d']
        self.batchsize = options["input"]["batchsize"]
        self.use_lstm=options["general"]["use_lstm"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.learningrate = options["training"]["learningrate"]
        self.modelType = options["training"]["learningrate"]
        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        self.save_prefix = options["training"]["save_prefix"]
        if options['general']['use_slice']:
            self.trainingdataset =NCPJPGDataset(options["training"]["data_root"],
                                                options["training"]["index_root"],
                                                options["training"]["padding"],
                                                True)#
        else:
            if options['general']['use_3d']:
                self.trainingdataset = NCPDataset(options["training"]["data_root"],
                                                    options["training"]["index_root"],
                                                    options["training"]["padding"],
                                                    True)
            else:
                self.trainingdataset = NCP2DDataset(options["training"]["data_root"],
                                                    options["training"]["index_root"],
                                                    options["training"]["padding"],
                                                    True)##TODO:3

        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True)
        

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 10)
        return self.learningrate * pow(0.5, decay)

    def __call__(self, model, epoch):
        #set up the loss function.
        model.train()
        if self.use_3d:
            criterion=model.loss()#TODO:2
        else:
            #criterion=nn.
            criterion =Cir_loss()

        if self.use_lstm:
            criterion=NLLSequenceLoss()
        if(self.usecudnn):
            net = nn.DataParallel(model).cuda()
            criterion = criterion.cuda()
               
        optimizer = optim.Adam(
                        model.parameters(),
                        lr = self.learningRate(epoch),amsgrad=True)
        
        #transfer the model to the GPU.       
            
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            length = Variable(sample_batched['length'])
           # break
            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs,seg_input = net(input)
            input_r=input*seg_input
            input_inv=input*(1-seg_input)
            outputs_r, _ = net(input_r)
            outputs_inv, _ = net(input_inv)
            if self.use_3d or self.use_lstm:
                loss = criterion(outputs, length,labels.squeeze(1))
            else:
                loss = criterion(outputs, labels.squeeze(1),outputs_r,outputs_inv,seg_input,input_r)
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize

            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime, len(self.trainingdataset))
                Trainer.writer.add_scalar('Train Loss', loss, Trainer.tot_iter)
            Trainer.tot_iter += 1

        print("Epoch "+str(epoch)+"completed, saving state...")
        print(self.use_3d)
        torch.save(model.state_dict(), "{}_{:0>8}.pt".format(self.save_prefix, epoch))       
