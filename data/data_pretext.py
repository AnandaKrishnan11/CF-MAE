import torch
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader, Dataset

import numpy as np
import os

from skimage.io import imread



def listing(pre_datafolder,post_datafolder):

    pre_pathlist = [os.path.join(pre_datafolder,img) for img in os.listdir(pre_datafolder)]

    post_pathlist= [os.path.join(post_datafolder,img) for img in os.listdir(post_datafolder)]

    return pre_pathlist,post_pathlist


# Dataset and Dataloader
class CustomDataset(Dataset):

    def __init__(self,pre_datafolder,post_datafolder,pre_meanstd,post_meanstd):
        super().__init__()

        self.pre_meanstd = pre_meanstd
        self.post_meanstd = post_meanstd


        self.pre,self.post = listing(pre_datafolder,post_datafolder)
        


    def __getitem__(self, index):
        pre = self.pre[index]
        post = self.post[index]

        pre = imread(pre)
        post = imread(post)

        pre, post = self.pre_transform(pre), self.post_transform(post)
        return pre, post


    def pre_transform(self,x):

        x = torch.from_numpy(x).permute(2,0,1).float()
        
        x = normalize(x,self.pre_meanstd['mean'],self.pre_meanstd['std'],inplace=True)

        return x

    def post_transform(self,x):

        x  = torch.from_numpy(x).permute(2,0,1).float()
        
        x = normalize(x,self.post_meanstd['mean'],self.post_meanstd['std'],inplace=True)

        return x
    
    
    def __len__(self):
        return len(self.pre)
    

# dataloader function
def dataloader(pre_datafolder,post_datafolder,pre_meanstd,post_meanstd):
    
    dataset = CustomDataset(pre_datafolder,post_datafolder,pre_meanstd,post_meanstd)

    return DataLoader(dataset,shuffle=True,batch_size=200)