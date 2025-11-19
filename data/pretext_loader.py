import torch
import torch.distributed as dist
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import os
import numpy as np
from skimage.io import imread


# Dataset and Dataloader
class CF_MAE_Dataset(Dataset):

    def __init__(self,config,logger):
        super().__init__()

        self.data_config = config    
        self.pre,self.post = self.get_data()
        self.logger = logger
        self.logger.info(f'[CF_MAE_Dataset INIT]---------- with {len(self.pre)} PRE and {len(self.post)} Images')
        
    def __getitem__(self, index):

        pre = self.pre[index]
        post = self.post[index]
        return pre, post
    
    def get_data(self):

        pre_list = []
        post_list = []

        for folds in os.listdir(self.data_config['path']):
            if folds == 'SEN4FLOOD_A' or folds == 'SEN4FLOOD_B':
                pre = os.listdir(os.path.join(self.data_config['path'],folds,'pre'))
                post = os.listdir(os.path.join(self.data_config['path'],folds,'post'))

                for pre_img,post_img in zip(pre,post):
                    pre_im = imread(pre_img)
                    post_im = imread(post_img)
                    pre_list.extend(pre), post_list.extend(post) = self.pre_transform(pre_im,type='sar'), self.post_transform(post_im, type='sar')
            else: 
                pre = os.listdir(os.path.join(self.data_config['path'],folds,'pre'))
                post = os.listdir(os.path.join(self.data_config['path'],folds,'post'))

                for pre_img,post_img in zip(pre,post):
                    pre_im = imread(pre_img)
                    post_im = imread(post_img)
                    pre_list.extend(pre), post_list.extend(post) = self.pre_transform(pre_im,type='optical'), self.post_transform(post_im, type='optical')
        return pre_list,post_list

    def pre_transform(self,x,type):
        
        if type == 'sar':
            x = torch.from_numpy(x).permute(2,0,1).float()
            x = normalize(x,self.data_config['modalities']['PRE']['SAR']['mean'],self.data_config['modalities']['PRE']['SAR']['std'],inplace=True)
        else:
            x = torch.from_numpy(x).permute(2,0,1).float()
            x = normalize(x,self.data_config['modalities']['PRE']['OPTICAL']['mean'],self.data_config['modalities']['PRE']['OPTICAL']['std'],inplace=True)
        return x

    def post_transform(self,x,type):

        if type == 'sar':
            x = torch.from_numpy(x).permute(2,0,1).float() 
            x = normalize(x,self.data_config['modalities']['POST']['SAR']['mean'],self.data_config['modalities']['POST']['SAR']['std'],inplace=True)
        else:
            x = torch.from_numpy(x).permute(2,0,1).float()          
            x = normalize(x,self.data_config['modalities']['POST']['OPTICAL']['mean'],self.data_config['modalities']['POST']['OPTICAL']['std'],inplace=True)
        return x
    
    def __len__(self):
        return len(self.pre)

# dataloader function
def build_loader(config,logger,distributed = False):

    config = config(is_pretrain=True,type='data')          
    dataset = CF_MAE_Dataset(config,logger)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader =  DataLoader(dataset,sampler=sampler,batch_size=config['batch_size'],num_workers=config['num_workers'],pin_memory=config['pin_memory'], drop_last=True)   
    else: 
        dataloader =  DataLoader(dataset,shuffle=config['shuffle'],batch_size=config['batch_size'],num_workers=config['num_workers'],pin_memory=True, drop_last=True)  
    return dataloader