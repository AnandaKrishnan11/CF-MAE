import os
import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from skimage.io import imread

class CF_MAE_Dataset(Dataset):

    def __init__(self, config, logger):
        super().__init__()
        self.data_config = config
        self.logger = logger
        self.pre_list, self.post_list = self.loadfile_list()
        logger.info(f"[CF_MAE_Dataset] {len(self.pre_list)} pre/post pairs")

    def loadfile_list(self):
        pre_list, post_list = [], []
        root = self.data_config["PATH"]

        for fold in sorted(os.listdir(root)):
            pre = self.list_files(os.path.join(root, fold, "pre"))
            post = self.list_files(os.path.join(root, fold, "post"))
            if len(pre) != len(post):
                raise ValueError(f"Mismatch in file count in {fold} for PRE and POST")
            pre_list += pre
            post_list += post

        if not pre_list:
            raise RuntimeError(f"no images found under {root}")
        return pre_list, post_list

    @staticmethod
    def list_files(directory):
        files = []
        for extension in ("*.tif", "*.png"):
            files += glob.glob(os.path.join(directory, extension))
        return sorted(files)

    def transform(self, image):
        x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        x = torch.nan_to_num(x)

        min_vals = x.amin(dim=(1, 2), keepdim=True)
        max_vals = x.amax(dim=(1, 2), keepdim=True)
        return (x - min_vals) / (max_vals - min_vals + 1e-6)

    def __getitem__(self, index):
        pre = self.transform(imread(self.pre_list[index]))
        post = self.transform(imread(self.post_list[index]))
        return pre, post

    def __len__(self):
        return len(self.pre_list)


def build_loader(config, logger):
    dataset = CF_MAE_Dataset(config, logger)

    return DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=config["SHUFFLE"],
        num_workers=config["NUM_WORKERS"],
        pin_memory=config["PIN_MEMORY"],
        drop_last=True)