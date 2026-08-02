import os
import glob

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset

EXTENSIONS = ("*.tif", "*.tiff", "*.png")

class Downstream_Dataset(Dataset):
    """Paired pre/post images with a binary change label.

    Expected layout:  PATH/<run>/{pre,post,label}/<name>.{tif,png}

    """

    def __init__(self, config, logger, run):
        super().__init__()
        self.data_config = config
        self.logger = logger
        self.pre_list, self.post_list, self.lbl_list = self.loadfile_list(run)
        logger.info(f"[Downstream_Dataset] {run}: {len(self.pre_list)} triplets")

    def loadfile_list(self, run):
        root = os.path.join(self.data_config["PATH"], run.lower())
        pre = self.list_files(os.path.join(root, "pre"))
        post = self.list_files(os.path.join(root, "post"))
        lbl = self.list_files(os.path.join(root, "label"))

        names = sorted(set(pre) & set(post) & set(lbl))
        if not names:
            raise RuntimeError(f"no complete pre/post/label triplets under {root}")

        incomplete = (set(pre) | set(post) | set(lbl)) - set(names)
        if incomplete:
            raise ValueError(
                f"{len(incomplete)} incomplete triplets in {root}, "
                f"e.g. {sorted(incomplete)[:5]}"
            )

        return [pre[n] for n in names], [post[n] for n in names], [lbl[n] for n in names]

    @staticmethod
    def list_files(directory):
        """Map the shared index -> full path (pre_0.tif / post_0.tif / label_0.tif)."""
        files = {}
        for extension in EXTENSIONS:
            for path in glob.glob(os.path.join(directory, extension)):
                stem = os.path.splitext(os.path.basename(path))[0]
                key = stem.split("_", 1)[1] if "_" in stem else stem
                files[key] = path
        return files

    def transform(self, image):
        if image.ndim == 2:
            image = image[..., None]
        x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        x = torch.nan_to_num(x)

        min_vals = x.amin(dim=(1, 2), keepdim=True)
        max_vals = x.amax(dim=(1, 2), keepdim=True)
        return (x - min_vals) / (max_vals - min_vals + 1e-6)

    def label_transform(self, label):
        """Binarise to {0, 1} and add the channel dimension."""
        if label.ndim == 3:
            label = label[:, :, 0] 
        label = (label > 0).astype(np.float32)
        return torch.from_numpy(label).unsqueeze(0)

    def __getitem__(self, index):
        pre = self.transform(imread(self.pre_list[index]))
        post = self.transform(imread(self.post_list[index]))
        lbl = self.label_transform(imread(self.lbl_list[index]))
        return pre, post, lbl

    def __len__(self):
        return len(self.pre_list)


def build_loader(config, logger, run):
    dataset = Downstream_Dataset(config, logger, run)
    is_train = run == "train"

    return DataLoader(
        dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=config["SHUFFLE"] and is_train,
        num_workers=config["NUM_WORKERS"],
        pin_memory=config["PIN_MEMORY"],
        drop_last=is_train)