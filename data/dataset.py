import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset

from PIL import Image
from radiomics import featureextractor
#import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
#import torchvision
import torch
import time
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import SimpleITK as sitk
import os
import numpy as np
import glob,six
import pandas as pd
import random
import cv2 as cv
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms

class NCPDataset(Dataset):
    def __init__(self, index_root, padding, augment=False,z_length=5):
        self.padding = padding
        self.data = []
        self.padding = padding
        self.augment = augment
        self.z_length=z_length
        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()
            self.mask=[item.split(',')[-1][:-1] for item in  self.data]
            self.data = [item.split(',')[-0] for item in self.data]
        cls = []
        for data_path in self.data:
            if 'healthy' in data_path:
                cls.append(0)
            elif 'cap' in data_path:
                cls.append(1)
            else:
                cls.append(2)  # covid
        self.labels=cls
        print('num of data:', len(self.data))

    def __len__(self):
        return len(self.data)
    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append((n_samples / float(cnt_dict[label])))

        return weights
    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if 'healthy' in data_path:
            cls = 0
        elif 'cap' in data_path:
            cls = 1
        else:
            cls = 2
        seg_path = self.mask[idx]
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)

        Mask = sitk.ReadImage(seg_path)
        M = sitk.GetArrayFromImage(Mask)

        valid=M.sum(1).sum(1)>500
        M=M[valid,:,:]
        data=data[valid,:,:]
        try:
            xx, yy, zz = np.where(M > 0)
            data = data[min(xx):max(xx), min(yy):max(yy), min(zz):max(zz)]
            M = M[min(xx):max(xx), min(yy):max(yy), min(zz):max(zz)]
        except:
            print(data_path)

        #data=np.stack([data,data,data],0)
        data[data > 500] = 500
        data[data < -1200] = -1200
        data = data * 255.0 / 1700
        data=(data+1200).astype(np.uint8)
        if self.augment:
            data,M=self.do_augmentation(data,M)
        #cv.imwrite('temp.jpg', data[:,64,:,:])
        temporalvolume,length = self.bbc(data, self.padding, self.z_length)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([length]),
            'features': torch.LongTensor([length])
            }
    def do_augmentation(self, array, mask):

        #array = array[None, ...]
        patch_size = np.asarray(array.shape)
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))
        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None,None, ...]

        mask = mask[None, None, ...]
        r_range = (0, (15 / 360.) * 2 * np.pi)
        r_range2 = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.
        augmented, mask = spatial_transforms.augment_spatial(
            augmented, seg=mask, patch_size=patch_size,
            do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range2, angle_y=r_range2, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=1,
            p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        mask = mask[0]
        augmented= (augmented[0,0, :, :, :]).astype(np.uint8)
        return augmented, mask
    def bbc(self,V, padding,z_length=3):

        temporalvolume = torch.zeros((z_length, padding, 224, 224))
        for cnt,i in enumerate(range(0,V.shape[0]-z_length,z_length)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            result=[]
            for j in range(z_length):
                result.append(transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0, 0, 0], [1, 1, 1]),
                ])(V[i+j:i+j+1,:,:])[0,:,:])
            temporalvolume[:, cnt] = torch.stack(result,0)

        #print(cnt)
        return temporalvolume,cnt

class NCP2DDataset(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False):
        self.padding = padding
        self.data = []
        self.data_root = data_root
        self.padding = padding
        self.augment = augment

        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()

        #for item in list:
         #   self.data.append(item)

        #print('index file:', index_root)
        print('num of data:', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        cls=1-int(data_path.split('/')[-1][0]=='c')
        data=np.load(os.path.join(self.data_root, data_path))

        #data[data>400]=400
        #data[data<-1700]=-1700
        #data=data+1700
        #data=(data/data.max()*255).astype(np.uint8)
        #cv.imwrite('temp.jpg', data[:,64,:,:])
        temporalvolume,length = self.bbc(data, self.padding, self.augment)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([length])
            }

    def bbc(self,V, padding, augmentation=True):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        croptransform = transforms.CenterCrop((224, 224))
        if (augmentation):
            crop = StatefulRandomCrop((224, 224), (224, 224))
            flip = StatefulRandomHorizontalFlip(0.5)

            croptransform = transforms.Compose([
                crop,
                flip
            ])

        for cnt,i in enumerate(range(V.shape[0])):
            if cnt>=padding:
                break
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop((256, 256)),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]),
            ])(V[i,:,:,:])

            temporalvolume[:, cnt] = result

        if cnt==0:
            print(cnt)
        return temporalvolume,cnt

class NCPJPGDataset(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=2):
        self.padding = padding
        self.data = []
        self.data_root = open(data_root,'r').readlines()
        self.text_book=[item.split('\t') for item in self.data_root]
        self.padding = padding
        self.augment = augment
        self.cls_num=cls_num
        self.train_augmentation = transforms.Compose([transforms.Resize(288),##just for abnormal detector
                                                     transforms.RandomCrop(224),
                                                     #transforms.RandomRotation(45),
                                                     transforms.RandomHorizontalFlip(0.2),
                                                     transforms.RandomVerticalFlip(0.2),
                                                     transforms.RandomAffine(45, translate=(0,0.2),fillcolor=0),

                                                     transforms.ToTensor(),
                                                     transforms.RandomErasing(p=0.1),
                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                     ])
        self.test_augmentation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                 ])
        with open(index_root, 'r') as f:
        #list=os.listdir(data_root)
            self.data=f.readlines()

        #for item in list:
         #   self.data.append(item)

        #print('index file:', index_root)
        print('num of data:', len(self.data))
        pa_id=list(set([st.split('/')[-1].split('_')[0] for st in self.data]))
        #pa_id_0=[id[0]=='c' or id[1]=='.' for id in pa_id]
        #print(np.sum(pa_id_0),len(pa_id)-np.sum(pa_id_0))
        if self.cls_num==2:
            cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1]=='.' or
                           data_path.split('/')[-2]=='masked_ild') for data_path in self.data]
        elif self.cls_num==4:
            cls=[]
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'CAP' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                else:
                    cls.append(3)#covid
        elif self.cls_num==5:
            cls=[]
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'lidc' in data_path:
                    cls.append(1)
                elif 'ild' in data_path:
                    cls.append(2)
                elif 'CAP' in data_path:
                    cls.append(3)#covid
                else:
                    cls.append(4)
        nums=[np.sum(np.array(cls)==i) for i in range(max(cls)+1)]
        print(nums)
        self.nums=nums
    def get_w(self):
        S=np.sum(self.nums)
        nums=S/(self.nums)
        w=nums/np.sum(nums)
        return w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if self.cls_num==2:
            cls=1-int(data_path.split('/')[-1][0]=='c' or data_path.split('/')[-1][1]=='.' or
                      data_path.split('/')[-2]=='masked_ild')
        elif self.cls_num==3:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif data_path.split('/')[-1][1] == '.' or data_path.split('/')[-2] == 'masked_ild':
                cls = 1
            else:
                cls = 2  # covid
        elif self.cls_num==4:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'CAP' in data_path:
                cls = 1
            elif  'ILD' in data_path:
                cls = 2  # covid
            else:
                cls=3
        elif self.cls_num==5:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'lidc'in data_path:
                cls = 1
            elif 'ild'in data_path:
                cls = 2
            elif 'CAP'in data_path:
                cls=3
            else:
                cls=4 # covid
        data=Image.open(data_path)
        age = -1
        gender = -1
        if  'lidc'in data_path or data_path.split('/')[-3] == 'reader_ex':
            age = -1
            gender = -1
        elif 'ILD' in data_path:
            temp = 'ILD/' + data_path.split('/')[-1].split('_')[0]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        elif 'CAP' in data_path :
            temp = 'CAP/' + data_path.split('/')[-1].split('_')[0]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        else:
            if data_path.split('/')[-3]=='slice_test_seg':
                if len(data_path.split('/')[-1].split('_')[1])>2:
                    a=data_path.split('/')[-1].split('c--')[-1]
                    temp='test1/'+a.split('_')[0]+'_'+a.split('_')[1]
                else:
                    a = data_path.split('/')[-1].split('c--')[-1]
                    temp='train1/'+a.split('_')[0]+'_'+a.split('_')[1]
            else:
                temp = data_path.split('/')[-2].split('_')[-1] + '/' + data_path.split('/')[-1].split('_')[0] + '_' + \
                       data_path.split('/')[-1].split('_')[1]
            for line in self.text_book:
                if line[0].split('.nii')[0] == temp:
                    age = int(line[1])
                    gender = int(line[2][:-1] == 'M')  # m 1, f 0
                    break
        if self.augment:
            data=self.train_augmentation(data)
        else:
            data=self.test_augmentation(data)
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([1]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age])
            }

class NCPJPGtestDataset(Dataset):
    def __init__(self, data_root, pre_lung_root,padding,lists=None,exlude_lists=True,age_list=None,cls_num=2):
        self.padding = padding
        self.cls_num=cls_num
        self.data = []
        self.text_book=None
        if isinstance(age_list,str):
            self.data_root = open(age_list, 'r').readlines()
            self.text_book = [item.split('\t') for item in self.data_root]
        self.mask=[]
        if isinstance(lists,list):
            if  not exlude_lists:
                self.data=lists
                self.mask=[item.split('_data')[0]+'_seg'+item.split('_data')[1][:-1] for item in self.data]
                self.data = [item[:-1] for item in self.data]
            else:
                if isinstance(data_root, list):
                    for r1, r2 in zip(data_root, pre_lung_root):
                        D= glob.glob(r1 + '/*.n*')
                        D=[t for t in D if not (t+'\n') in lists]
                        M= [item.split('_data')[0]+'_seg'+item.split('_data')[1] for item in D]
                        self.data+=D
                        self.mask+=M
                else:
                    D = glob.glob(data_root + '/*.n*')
                    D = [t for t in D if not (t+'\n') in lists]
                    M = [item.split('_data')[0] + '_seg' + item.split('_data')[1] for item in D]
                    self.data += D
                    self.mask += M
        else:
            if isinstance (data_root,list):
                for r1,r2 in zip(data_root,pre_lung_root):
                    self.data+=glob.glob(r1+'/*.n*')
                    self.mask+=glob.glob(r2+'/*.n*')
            else:
                self.data = glob.glob(data_root)
                self.mask=glob.glob(pre_lung_root)
        self.pre_root=pre_lung_root
        self.data_root = data_root
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                        transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))

        if self.cls_num == 2:
            cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.index('LIDC')>-1 or
                           data_path.index('ILD')>-1) for data_path in self.data]
        elif self.cls_num == 4:
            cls = []
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'CAP' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                else:
                    cls.append(3)  # covid
        #cls=0
        elif self.cls_num==5:
            cls = []
            for data_path in self.data:
                if data_path.split('/')[-1][0] == 'c':
                    cls.append(0)
                elif 'LIDC' in data_path:
                    cls.append(1)
                elif 'ILD' in data_path:
                    cls.append(2)
                elif 'CAP' in data_path:
                    cls.append(3)
                else:
                    cls.append(4)  # covid
        nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
        print(nums)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #print(data_path,mask_path)
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        if self.cls_num == 2:
            cls = 1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1] == '.' or
                          data_path.split('/')[-3] == 'ILD')
        elif self.cls_num == 3:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif data_path.split('/')[-1][1] == '.' or data_path.split('/')[-3] == 'ILD':
                cls = 1
            else:
                cls = 2  # covid
        elif self.cls_num==4:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'CAP' in data_path:
                cls = 1
            elif 'ILD' in data_path:
                cls = 2  # covid
            else:
                cls = 3
        elif self.cls_num==5:
            if data_path.split('/')[-1][0] == 'c':
                cls = 0
            elif 'LIDC' in data_path:
                cls = 1
            elif 'ILD' in data_path:
                cls = 2
            elif 'CAP' in data_path:
                cls = 3
            else:
                cls = 4# covid

        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        M=M[:data.shape[0],:,:]
        valid=np.where(M.sum(1).sum(1)>500)
        data = data[valid[0], :, :]
        M = M[valid[0], :data.shape[1], :data.shape[2]]
        data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,name = self.bbc(data, self.padding,M)
        age = -1
        gender = -1

        if isinstance(self.text_book,list):
            if 'LIDC' in data_path or\
                  data_path.split('/')[-3] == 'reader_ex':
                age = -1
                gender = -1
            elif 'ILD' in data_path:
                temp = 'ILD/' + data_path.split('/')[-1].split('.nii')[0]
                for line in self.text_book:
                    if line[0].split('.nii')[0] == temp:
                        age = int(line[1])
                        try:
                            gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        except:
                            gender=-1
                        break
            elif 'CAP' in data_path:
                temp = 'CAP/' + data_path.split('/')[-1].split('_')[1]
                for line in self.text_book:
                    if line[0].split('.nii')[0] == temp:
                        age = int(line[1])
                        gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        break
            else:
                temp = data_path.split('/')[-2].split('_')[-1] + '/' + data_path.split('/')[-1].split('_')[0] + '_' + \
                       data_path.split('/')[-1].split('_')[1]
                for line in self.text_book:
                    if line[0] == temp:
                        age = int(line[1])
                        gender = int(line[2][:-1] == 'M')  # m 1, f 0
                        break

        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,name],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age])

            }

    def bbc(self,V, padding,pre=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        #croptransform = transforms.CenterCrop((224, 224))
        cnt=0
        name=[]
        for cnt,i in enumerate(range(1,V.shape[0]-1,3)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i-1:i+1,:,:]
            data[data > 700] = 700
            data[data < -1200] = -1200
            data = data * 255.0 / 1900
            name.append(i)
            data = data - data.min()
            data = np.concatenate([pre[i-1:i, :, :] * 255,data], 0)  # mask one channel
            data = data.astype(np.uint8)
            data=Image.fromarray(data.transpose(1,2,0))
            #data.save('temp.jpg')
            result = self.transform(data)

            temporalvolume[:, cnt] = result

        if cnt==0:
            print(cnt)
        return temporalvolume,name

class NCPJPGtestDataset_MHA(Dataset):
    def __init__(self, data_root, pre_lung_root,padding,lists=None,exlude_lists=True):
        self.padding = padding
        self.data = []
        self.mask=[]
        if isinstance(lists,list):
            if  not exlude_lists:
                self.data=lists
                self.mask=[item.split('images')[0]+'lungsegs'+item.split('images')[1][:-1] for item in self.data]
                self.data = [item[:-1] for item in self.data]
            else:
                if isinstance(data_root, list):
                    for r1, r2 in zip(data_root, pre_lung_root):
                        D= glob.glob(r1 + '/*/*.mha')
                        D=[t for t in D if not (t+'\n') in lists]
                        M= [item.split('images')[0]+'lungsegs'+item.split('images')[1] for item in D]
                        self.data+=D
                        self.mask+=M
                else:
                    D = glob.glob(data_root + '/*/*.mha')
                    D = [t for t in D if not (t+'\n') in lists]
                    M = [item.split('images')[0] + 'lungsegs' + item.split('images')[1] for item in D]
                    self.data += D
                    self.mask += M
        else:
            if isinstance (data_root,list):
                for r1,r2 in zip(data_root,pre_lung_root):
                    self.data+=glob.glob(r1+'/*/*.mha')
                    self.mask+=glob.glob(r2+'/*/*.mha')
            else:
                self.data = glob.glob(data_root)
                self.mask=glob.glob(pre_lung_root)
        self.data=list(set(self.data))
        self.mask = list(set(self.mask))
        self.pre_root=pre_lung_root
        self.data_root = data_root
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                        transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))

        cls = [1 - int(data_path.split('/')[-1][0] == 'c' or data_path.split('/')[-1][1] == '.'
                       or data_path.split('/')[-3]=='ILD' or data_path.split('/')[-3]=='reader_ex') for
               data_path in self.data]
        #cls=0
        print(np.sum(np.array(cls) == 0), np.sum(np.array(cls) == 1))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]
        cls=1-int(data_path.split('/')[-1][0]=='c'or
                  data_path.split('/')[-3]=='ILD' or
                  data_path.split('/')[-3] == 'LIDC' or
                  data_path.split('/')[-3] == 'reader_ex')

        #cls=0
        #cls=0
        #volume = sitk.ReadImage(os.path.join(input_path, name))
        mask = sitk.ReadImage(mask_path)
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        #data = data[-300:-40, :, :]
        #print(M.shape)
        M = M[:, :data.shape[1], :data.shape[2]]
        data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,name = self.bbc(data, self.padding,M)
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,name]
            }

    def bbc(self,V, padding,pre=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        #croptransform = transforms.CenterCrop((224, 224))
        cnt=0
        name=[]
        for cnt,i in enumerate(range(V.shape[0]-1)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            #if cnt>=padding:
            #    break
            data=V[i:i+1,:,:]
            data[data > 700] = 700
            data[data < -1200] = -1200
            data = data * 255.0 / 1900
            name.append(i)
            data = data - data.min()
            data = np.concatenate([pre[i:i + 1, :, :] * 255,data,data], 0)  # mask one channel
            data = data.astype(np.uint8)
            data=Image.fromarray(data.transpose(1,2,0))
            #data.save('temp.jpg')
            result = self.transform(data)

            temporalvolume[:, cnt] = result

        #if cnt==0:
        print(cnt)
        temporalvolume=temporalvolume[:,:cnt+1]
        return temporalvolume,name



class NCPJPGDataset_new(Dataset):
    def __init__(self, data_root,index_root, padding, augment=False,cls_num=2,mod='ab',options=None):
        self.mod=mod
        self.padding = padding
        self.data = []
        self.options=options
        if 'radiomics_path' in options['general'].keys():
            self.radiomics_path = options['general']['radiomics_path']
            os.makedirs(self.radiomics_path,exist_ok=True)
        else:
            self.radiomics_path=[]
        if data_root[-3:]=='csv':
            self.r=pd.read_csv(data_root)
        self.data_root=data_root
        self.padding = padding
        self.augment = augment
        self.cls_num=cls_num
        self.extractor = featureextractor.RadiomicsFeatureExtractor('radiomics/RadiomicsParams.yaml')
        self.train_augmentation = transforms.Compose([transforms.Resize((224,224)),##just for abnormal detector
                                                     #transforms.RandomRotation(45),
                                                     transforms.RandomAffine(30, translate=(0,0.1),fillcolor=0),
                                                     #transforms.RandomCrop(224),

                                                     transforms.ToTensor(),
                                                     transforms.RandomErasing(p=0.1),
                                                     transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                     ])
        self.test_augmentation = transforms.Compose([transforms.Resize((224,224)),
                                                 #transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0, 0, 0], [1, 1, 1])
                                                 ])
        with open(index_root, 'r') as f:
            self.data=f.readlines()
        print('num of data:', len(self.data))

        if self.cls_num==2:
            if self.mod=='ab':#abnormal detection
                cls = [int('pos' in data_path) for data_path in self.data]
            elif self.mod=='co':
                cls = [1-int('cap' in data_path) for data_path in self.data]
            else:
                cls = [1 - int('healthy' in data_path ) for data_path in self.data]
        elif self.cls_num==3:
            cls=[]
            for data_path in self.data:
                if 'healthy' in data_path:
                    cls.append(0)
                elif 'cap' in data_path:
                    cls.append(1)
                else:
                    cls.append(2)#covid

        nums=[np.sum(np.array(cls)==i) for i in range(self.cls_num)]
        print(nums)
        self.labels=cls
        self.nums=nums
    def get_w(self):
        S=np.sum(self.nums)
        nums=(S/(self.nums))
        w=nums/np.sum(nums)
        return w
    def make_weights_for_balanced_classes(self):
        """Making sampling weights for the data samples
        :returns: sampling weigghts for dealing with class imbalance problem

        """
        n_samples = len(self.labels)
        unique, cnts = np.unique(self.labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in self.labels:
            weights.append((n_samples / float(cnt_dict[label])))

        return weights
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        if data_path[-1]=='\n':
            data_path=data_path[:-1]

        feature=0
        if self.data_root[-3:] == 'csv' and not isinstance(self.radiomics_path,str):
            this_name=data_path.split('/')[-2].split('masked_')[-1].split('2nd')[0]+'_' +\
                      data_path.split('/')[-1][:data_path.split('/')[-1].index('_',-7)]+'_label.nrrd'
            try:
                feature=self.r[self.r['filename'].isin([this_name])].values[0,1:]
                feature=np.delete(feature,2)
            except:
                print(this_name)

        if self.cls_num==2:
            if self.mod=='ab':#abnormal detection
                cls = int('pos' in data_path)
            elif self.mod=='co':
                cls = 1-int('ild' in data_path or 'cap' in data_path)
            else:
                cls = 1 - int('healthy' in data_path )
        elif self.cls_num==3:
            if 'healthy' in data_path:
                cls = 0
            elif 'cap' in data_path:
                cls = 1
            else:
                cls=2
        data=Image.open(data_path)
        age = int(data_path.split('_')[-3])
        gender = int(data_path.split('_')[-2]=='M')
        pos=int(data_path.split('_')[-1].split('.')[0])
        time1=time.time()
        if isinstance(self.radiomics_path,str):
            aim=os.path.join(self.radiomics_path,data_path.split('/')[-2].split('masked_')[-1].split('2nd')[0]+'_'+
                                                               data_path.split('/')[-1].split('.jpg')[0]+'.npy')
            if os.path.exists(aim):
                feature=np.load(aim)
            else:
                feature = []
                masks_a=(np.array(data)[:,:,0]>20).astype(np.int)
                if np.sum(masks_a>0)<2:
                    feature=np.zeros(479)
                else:
                    try:
                        result = self.extractor.execute(sitk.GetImageFromArray(np.array(data)[:,:,2]),
                                                        sitk.GetImageFromArray(masks_a))
                        for idx, (key, val) in enumerate(six.iteritems(result)):
                            if idx < 11:
                                continue
                            if not isinstance(val, (float, int, np.ndarray)):
                                continue
                            if np.isnan(val):
                                val = 0
                            feature.append(val)
                    except:
                        feature = np.zeros(479)

                feature=np.array(feature)
                np.save(aim,feature)
       # time2 = time.time()-time1
        #print(time2)
        if self.augment:
            data=self.train_augmentation(data)
        else:
            data=self.test_augmentation(data)
        return {'temporalvolume': data,
            'label': torch.LongTensor([cls]),
            'length':torch.LongTensor([1]),
            'gender':torch.LongTensor([gender]),
            'age':torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos/100]),
            'features':torch.FloatTensor([feature])
            }

class NCPJPGtestDataset_new(Dataset):
    def __init__(self, data_root,padding,lists,age_list=None,cls_num=2,mod='ab',options=None):
        #self.padding = padding
        self.data_root=data_root
        if data_root[-3:]=='csv':
            self.r=pd.read_csv(data_root)
        self.options = options
        if 'radiomics_path' in options['general'].keys():
            self.radiomics_path = options['general']['radiomics_path']
            os.makedirs(self.radiomics_path, exist_ok=True)
        else:
            self.radiomics_path = []
        self.extractor = featureextractor.RadiomicsFeatureExtractor('radiomics/RadiomicsParams.yaml')
        self.cls_num=cls_num
        self.data = []
        self.text_book=None
        self.mod=mod
        self.data=open(lists,'r').readlines()
        self.mask=[item.split(',')[1] for item in self.data]
        self.data = [item.split(',')[0] for item in self.data]
        self.padding = padding

        self.transform=  transforms.Compose([#transforms.ToPILImage(),
                                         transforms.Resize((224,224)),
                                         #transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0, 0, 0], [1, 1, 1])
                                         ])
        print('num of data:', len(self.data))
        person=[da.split('/')[-2]+'_'+da.split('/')[-1].split('_')[0]+'_'+da.split('/')[-1].split('_')[1] for da in self.data]
        person=list(set(person))


        if self.cls_num==2:
            if self.mod=='ab':#abnormal detection
                cls = [int('pos' in data_path) for data_path in self.data]
                cls_stage = [int('pos' in data_path) for data_path in self.data]
            elif self.mod=='co':
                cls = [1-int('cap' in data_path) for data_path in self.data]
                cls_stage = [1 - int('cap' in data_path) for data_path in self.data]
            else:
                cls = [1 - int('healthy' in data_path ) for data_path in self.data]
                cls_stage = [1 - int('healthy' in data_path) for data_path in self.data]

        elif self.cls_num == 3:
            cls = []
            for data_path in person:
                if 'healthy' in data_path:
                    cls.append(0)
                elif 'cap' in data_path:
                    cls.append(1)
                else:
                    cls.append(2)  # covid
            cls_stage=[]
            for data_path in self.data:
                if 'healthy' in data_path:
                    cls_stage.append(0)
                elif 'cap' in data_path:
                    cls_stage.append(1)
                else:
                    cls_stage.append(2)  # covid

        nums = [np.sum(np.array(cls) == i) for i in range(np.max(cls) + 1)]
        print('patient',nums)
        nums = [np.sum(np.array(cls_stage) == i) for i in range(np.max(cls_stage) + 1)]
        print('stages', nums)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #load video into a tensor
        data_path = self.data[idx]
        mask_path=self.mask[idx]
        #print(data_path,mask_path)
        if data_path[-1]=='\n':
            data_path=data_path[:-1]

        if self.data_root[-3:] == 'csv' and not isinstance(self.radiomics_path, str):
            this_name = data_path.split('/')[-2].split('masked_')[-1].split('2nd')[0] + '_' + \
                        data_path.split('/')[-1].split('.nii')[0] + '_label.nrrd'
            try:
                feature = self.r[self.r['filename'].isin([this_name])].values[0, 1:]
                feature = np.delete(feature, 2)
            except:
                print(this_name)
                a = 1
        if self.cls_num == 2:
            if self.mod=='ab':#abnormal detection
                cls = int('pos' in data_path)
            elif self.mod=='co':
                cls = 1-int('cap' in data_path)
            else:
                cls = 1 - int('healthy' in data_path)
        elif self.cls_num==3:
            if 'healthy' in data_path:
                cls = 0
            elif 'cap' in data_path:
                cls = 1
            else:
                cls = 2
        try:
            mask = sitk.ReadImage(mask_path[:-1])
        except:
            mask_path=mask_path.split('2020')[0]+mask_path.split('2020')[1][:-1]
            mask = sitk.ReadImage(mask_path)
        lesion_name = (mask_path.split('.nii')[0] + '_label.nrrd').replace('lungs','lesion')
        L = sitk.ReadImage(lesion_name)
        L = sitk.GetArrayFromImage(L)
        L[L > 0] = 1
        M = sitk.GetArrayFromImage(mask)
        volume=sitk.ReadImage(data_path)
        data=sitk.GetArrayFromImage(volume)
        M=M[:data.shape[0],:,:]
        M[M>1]=1
        valid=np.where(M.sum(1).sum(1)>500)
        area = np.where(M>0)
        data = data[valid[0], area[1].min():area[1].max(), area[2].min():area[2].max()]
        L=L[valid[0],area[1].min():area[1].max(),area[2].min():area[2].max()]
        M = M[valid[0], area[1].min():area[1].max(), area[2].min():area[2].max()]
        #data=data[:M.shape[0],:M.shape[1],:M.shape[2]]
        temporalvolume,pos,feature = self.bbc(data, self.padding,data_path,M,L)
        age = int(data_path.split('_')[-2])
        gender = int(data_path.split('_')[-1].split('.nii')[0]=='M')
        return {'temporalvolume': temporalvolume,
            'label': torch.LongTensor([cls]),
            'length':[data_path,pos],
            'gender': torch.LongTensor([gender]),
            'age': torch.LongTensor([age]),
            'pos':torch.FloatTensor([pos]),
            'features':torch.FloatTensor(feature)
            }

    def bbc(self,V, padding,data_path,pre=None,L=None):
        temporalvolume = torch.zeros((3, padding, 224, 224))
        F=np.zeros((padding,479))
        #croptransform = transforms.CenterCrop((224, 224))
        stride=max(V.shape[0]//padding,5)
        cnt=0
        name=[]
        if data_path=='/home/cwx/extra/covid_project_data/cap-qqhr/1_85_20200118_50_M.nii':
            a=1
        for cnt,i in enumerate(range(0,V.shape[0],stride)):
        #for cnt, i in enumerate(range(V.shape[0]-5,5, -3)):
            if cnt>=padding:
                break
            data=V[i,:,:]
            data[data > 500] = 500
            data[data < -1200] = -1200
            data = data * 255.0 / 1700
            name.append(float(i/V.shape[0]))
            data = data - data.min()
            data = np.stack([L[i,:,:]*data,pre[i, :, :] * data,data], -1)  # mask one channel
            data = data.astype(np.uint8)
            data=Image.fromarray(data)
            #data.save('temp.jpg')
            result = self.transform(data)
            if isinstance(self.radiomics_path,str):

                aim=os.path.join(self.radiomics_path,data_path.split('/')[-2].split('masked_')[-1].split('2nd')[0]+'_'+\
                                                                   data_path.split('/')[-1].split('.nii')[0]+'_'+\
                                                                    str(int(100*i/V.shape[0]))+'.npy')
                if os.path.exists(aim):
                    feature=np.load(aim,allow_pickle=True)
                else:
                    warnings.filterwarnings("ignore")
                    feature = []

                    try:
                        r = self.extractor.execute(sitk.GetImageFromArray(np.array(data)[:,:,2]),
                                                        sitk.GetImageFromArray(L[i,:,:]))
                        for idx, (key, val) in enumerate(six.iteritems(r)):
                            if idx < 11:
                                continue
                            if not isinstance(val, (float, int, np.ndarray)):
                                continue
                            if np.isnan(val):
                                val = 0
                            feature.append(val)
                    except:
                        feature = np.zeros(479)
                        #print('zeros',data_path)
                    feature=np.array(feature)
                    np.save(aim,feature,allow_pickle=True)
                F[cnt] = feature
            temporalvolume[:, cnt] = result

        #temporalvolume=temporalvolume
        return temporalvolume[:,:cnt,:,:],name,F[:cnt]