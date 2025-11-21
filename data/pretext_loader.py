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

        for folds in os.listdir(self.data_config['PATH']):
            if folds == 'SEN4FLOOD_A' or folds == 'SEN4FLOOD_B':
                pre = os.listdir(os.path.join(self.data_config['PATH'],folds,'pre'))
                post = os.listdir(os.path.join(self.data_config['PATH'],folds,'post'))

                for pre_img,post_img in zip(pre,post):
                    pre_im = imread(pre_img)
                    post_im = imread(post_img)
                    pre_list.extend(pre), post_list.extend(post) = self.pre_transform(pre_im,type='sar'), self.post_transform(post_im, type='sar')
            else: 
                pre = os.listdir(os.path.join(self.data_config['PATH'],folds,'pre'))
                post = os.listdir(os.path.join(self.data_config['PATH'],folds,'post'))

                for pre_img,post_img in zip(pre,post):
                    pre_im = imread(pre_img)
                    post_im = imread(post_img)
                    pre_list.extend(pre), post_list.extend(post) = self.pre_transform(pre_im,type='optical'), self.post_transform(post_im, type='optical')
        return pre_list,post_list

    def pre_transform(self,x,type):
        
        if type == 'sar':
            x = torch.from_numpy(x).permute(2,0,1).float()
            x = normalize(x,self.data_config['MODALITIES']['PRE']['SAR']['MEAN'],self.data_config['MODALITIES']['PRE']['SAR']['STD'],inplace=True)
        else:
            x = torch.from_numpy(x).permute(2,0,1).float()
            x = normalize(x,self.data_config['MODALITIES']['PRE']['OPTICAL']['MEAN'],self.data_config['MODALITIES']['PRE']['OPTICAL']['STD'],inplace=True)
        return x

    def post_transform(self,x,type):

        if type == 'sar':
            x = torch.from_numpy(x).permute(2,0,1).float() 
            x = normalize(x,self.data_config['MODALITIES']['POST']['SAR']['MEAN'],self.data_config['MODALITIES']['POST']['SAR']['STD'],inplace=True)
        else:
            x = torch.from_numpy(x).permute(2,0,1).float()          
            x = normalize(x,self.data_config['MODALITIES']['POST']['OPTICAL']['MEAN'],self.data_config['MODALITIES']['POST']['OPTICAL']['STD'],inplace=True)
        return x
    
    def __len__(self):
        return len(self.pre)

# dataloader function
def build_loader(config,logger,distributed = False):

    config = config(is_pretrain=True,type='data')          
    dataset = CF_MAE_Dataset(config,logger)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader =  DataLoader(dataset,sampler=sampler,batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)   
    else: 
        dataloader =  DataLoader(dataset,shuffle=config['SHUFFLE'],batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=True, drop_last=True)  
    return dataloader