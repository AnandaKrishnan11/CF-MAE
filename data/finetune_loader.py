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

    def __init__(self,config,logger,run,data):
        super().__init__()

        self.data_config = config    
        self.logger = logger
        self.pre_list, self.post_list,self.lbl_list = self.loadfile_list(run,data)
        self.logger.info(f'[Downstream_Dataset INIT]---------- with {len(self.pre_list)} PRE , {len(self.post_list)} POST and {len(self.lbl_list)} Label Images')
        
    def __getitem__(self, index):

        pre_im = imread(self.pre_list[index])
        #pre_im = pre_im[:,:,:3]
        post_im = imread(self.post_list[index])
        lbl_im = imread(self.lbl_list[index])
        #lbl_im = (lbl_im > 1).astype(np.int8)

        lbl_tensor = torch.from_numpy(lbl_im.astype(np.float32)).unsqueeze(0)
        
        pre_tensor = self.pre_transform(pre_im)
        post_tensor = self.post_transform(post_im)
    
        return pre_tensor, post_tensor,lbl_tensor
    
    def loadfile_list(self,run,data='DSIFN'):

        pre_list = []
        post_list = []
        lbl_list = []

        if data == 'S2WCD' or data == 'MAUI':

            if run == 'train':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'train','pre','*.tif')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'train','post','*.tif')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'train','label','*.tif')))
            elif run == 'test':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'test','pre','*.tif')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'test','post','*.tif')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'test','label','*.tif')))
            elif run == 'val':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'val','pre','*.tif')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'val','post','*.tif')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'val','label','*.tif')))
            
        elif data == 'DSIFN':

            if run == 'train':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TRAIN','pre','*.png')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TRAIN','post','*.png')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TRAIN','label','*.png')))
            elif run == 'test':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TEST','pre','*.png')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TEST','post','*.png')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'TEST','label','*.png')))
            elif run == 'val':
                pre = sorted(glob.glob(os.path.join(self.data_config['PATH'],'VAL','pre','*.png')))
                post = sorted(glob.glob(os.path.join(self.data_config['PATH'],'VAL','post','*.png')))
                lbl = sorted(glob.glob(os.path.join(self.data_config['PATH'],'VAL','label','*.png')))

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
        min_vals = x.amin(dim=(1, 2), keepdim=True)
        max_vals = x.amax(dim=(1, 2), keepdim=True)
        x = (x - min_vals) / (max_vals - min_vals + 1e-6)
      
        return x

    def post_transform(self,x):

        x = torch.from_numpy(x.astype(np.float32)).permute(2, 0, 1)
        min_vals = x.amin(dim=(1, 2), keepdim=True)
        max_vals = x.amax(dim=(1, 2), keepdim=True)
        x = (x - min_vals) / (max_vals - min_vals + 1e-6)

        return x
    
    def __len__(self): 
        return len(self.pre_list)

# dataloader function
def build_loader(config,logger,run,data,distributed = False):

    dataset = Downstream_Dataset(config,logger,run,data)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader =  DataLoader(dataset,sampler=sampler,batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)   
    else: 
        dataloader =  DataLoader(dataset,shuffle=config['SHUFFLE'],batch_size=config['BATCH_SIZE'],num_workers=config['NUM_WORKERS'],pin_memory=config['PIN_MEMORY'], drop_last=True)  
    return dataloader