import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from segmentation.unet import UNet
import SimpleITK as sitk
def get_model(model_path,cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.set_grad_enabled(False)
    model = UNet(n_classes=3)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
    
def predict(img, model, batch_size=8, cuda=True):
    model=get_model(model)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    img = sitk.GetArrayFromImage(img)
    img[img<-1024]=-1024
    img = img/255.
    # print(model)

    data = torch.from_numpy(img[:, np.newaxis, :, :])
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    _, H, W = img.shape
    res = np.zeros((1, H, W),dtype=np.int8)
    for i, data in enumerate(loader):
        images = data[0].to(device)
        images = images.float()
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        labels = labels.cpu().numpy()
        res = np.concatenate((res, labels), axis=0)

    return res[1:,:,:]

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    img_path = '/home/cwx/extra/covid_project_data/HxNx'
    pred_path = '/home/cwx/extra/covid_project_segs/lungs/HxNx'
    model_path = './lung_checkpoint.pth'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    print('# images:  ', len(os.listdir(img_path)))
    for filename in tqdm(os.listdir(img_path), dynamic_ncols=True):
        img=sitk.ReadImage(os.path.join(img_path, filename))
        result = predict(img, model_path)
        result[result>1]=1
        result=np.array(result,np.uint8)
        result=sitk.GetImageFromArray(result)
        sitk.WriteImage(result,os.path.join(pred_path, pred_path.split('/')[-1]+'_'+filename))
        a=1
        #np.save(os.path.join(pred_path, 'pred_' + filename), result)


