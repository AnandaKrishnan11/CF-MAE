import torch
import torch.distributed as dist
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import os
import glob
import numpy as np
from skimage.io import imread


# Dataset and Dataloader
class CF_MAE_Dataset(Dataset):

    def __init__(self,config,logger):
        super().__init__()

        self.data_config = config    
        self.logger = logger
        self.pre_list, self.post_list = self.loadfile_list()
        self.logger.info(f'[CF_MAE_Dataset INIT]---------- with {len(self.pre_list)} PRE and {len(self.post_list)} Images')
        
    def __getitem__(self, index):

        pre_im = imread(self.pre_list[index])
        post_im = imread(self.post_list[index])

        if np.any(~np.isfinite(pre_im)) or np.any(~np.isfinite(post_im)):
            pre_im = np.nan_to_num(pre_im)
            post_im = np.nan_to_num(post_im)
        
        pre_tensor = self.pre_transform(pre_im)
        post_tensor = self.post_transform(post_im)
    
        return pre_tensor, post_tensor
    
    def loadfile_list(self):

        pre_list = []
        post_list = []

        for folds in os.listdir(self.data_config['PATH']):
        
            pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],folds,'pre','*.tif'))+ glob.glob(os.path.join(self.data_config['PATH'], folds, 'pre', '*.png')))
            post = sorted(glob.glob(os.path.join(self.data_config['PATH'],folds,'post','*.tif')) + glob.glob(os.path.join(self.data_config['PATH'], folds, 'post', '*.png')))
            if len(pre) != len(post):
                raise ValueError(f"Mismatch in file count in {folds} for PRE and POST")
                continue
            pre_list.extend(pre)
            post_list.extend(post)   
        return pre_list,post_list

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
def build_loader(config,logger,distributed = False):

    dataset = CF_MAE_Dataset(config,logger)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader =  DataLoader(dataset,sampler=sampler,batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)   
    else: 
        dataloader =  DataLoader(dataset,shuffle=config['SHUFFLE'],batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)  
    return dataloader