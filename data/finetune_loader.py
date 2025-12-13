import torch
import torch.distributed as dist
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import os
import glob
import numpy as np
from skimage.io import imread


# Dataset and Dataloader
class Downstream_Dataset(Dataset):

    def __init__(self,config,logger,train):
        super().__init__()

        self.data_config = config    
        self.logger = logger
        self.pre_list, self.post_list,self.lbl_list = self.loadfile_list(train)
        self.logger.info(f'[Downstream_Dataset INIT]---------- with {len(self.pre_list)} PRE , {len(self.post_list)} POST and {len(self.lbl_list)} Label Images')
        
    def __getitem__(self, index):

        pre_im = imread(self.pre_list[index])
        post_im = imread(self.post_list[index])
        lbl_im = imread(self.lbl_list[index])
        lbl_tensor = torch.from_numpy(lbl_im.astype(np.float32)).unsqueeze(0)
        
        pre_tensor = self.pre_transform(pre_im)
        post_tensor = self.post_transform(post_im)
    
        return pre_tensor, post_tensor,lbl_tensor
    
    def loadfile_list(self,train=True):

        pre_list = []
        post_list = []
        lbl_list = []

        if train:
            pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'pre','train','*.tif')))
            post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'post','train','*.tif')))
            lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'lbl','train','*.tif')))
        else:
            pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'pre','test','*.tif')))
            post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'post','test','*.tif')))
            lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'lbl','test','*.tif')))

        if len(pre) != len(post) or len(pre)  != len(lbl):
            raise ValueError(f"Mismatch in file count in PRE, POST or Label")
            
        else:
            for pre_files,post_files,lbl_files in zip(pre, post,lbl):
                
                pre_list.append(pre_files)
                post_list.append(post_files)   
                lbl_list.append(lbl_files)   

        return pre_list,post_list,lbl_list

    def pre_transform(self,x):
        
        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        if self.data_config['MODALITIES']['MIN_MAX']:
            min_vals = x.amin(dim=(1, 2), keepdim=True)
            max_vals = x.amax(dim=(1, 2), keepdim=True)
            x = (x - min_vals) / (max_vals - min_vals + 1e-6)
        else:
            raise NotImplementedError
        return x

    def post_transform(self,x):

        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        if self.data_config['MODALITIES']['MIN_MAX']:
            min_vals = x.amin(dim=(1, 2), keepdim=True)
            max_vals = x.amax(dim=(1, 2), keepdim=True)
            x = (x - min_vals) / (max_vals - min_vals + 1e-6)
        else:
            raise NotImplementedError
        return x
    
    def __len__(self): 
        return len(self.pre_list)

# dataloader function
def build_loader(config,logger,train,distributed = False):

    dataset = Downstream_Dataset(config,logger,train)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader =  DataLoader(dataset,sampler=sampler,batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)   
    else: 
        dataloader =  DataLoader(dataset,shuffle=config['SHUFFLE'],batch_size=config['BATCH_SIZE'],num_workers=4,pin_memory=config['PIN_MEMORY'], drop_last=True)  
    return dataloader