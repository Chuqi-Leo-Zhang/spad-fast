import os
import random
import torch
import imageio
import numpy as np
from torch.utils.data import Dataset
import pickle

from spad.geometry import get_batch_from_spherical


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


class ObjaverseMultiViewDataset(Dataset):

    def __init__(self, 
                 root, 
                 num_views=2, 
                 image_size=256,
                 training=True):
        
        self.root = root
        self.obj_ids = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.num_views = num_views
        self.image_size = image_size
        self.training = training

    def __len__(self):
        return len(self.obj_ids)

    def load_image(self, path):
        img = imageio.imread(path)   # H W C or H W 4C
        img = img.astype(np.float32) / 255.0

        # drop alpha if present
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        return img

    def __getitem__(self, idx):
        obj_id = self.obj_ids[idx]
        path = os.path.join(self.root, obj_id)

        meta = read_pickle(os.path.join(path, "meta.pkl"))
        elevations = np.array(meta["elevations"])   # [T]
        azimuths   = np.array(meta["azimuths"])     # [T]
        caption    = str(meta["caption"])

        # add [tdv] token
        if "[tdv]" not in caption:
            caption = "[tdv] " + caption

        total_views = len(elevations)
        assert total_views >= self.num_views, f"{obj_id}: only {total_views} views, need {self.num_views}"

        # randomly sample views
        ids = np.random.choice(total_views, self.num_views, replace=False)

        sampled_imgs = []
        for vid in ids:
            img_path = os.path.join(path, f"{vid:03d}.png")
            img = self.load_image(img_path)  # H W C
            sampled_imgs.append(img)

        imgs = torch.from_numpy(np.stack(sampled_imgs, axis=0))  # [V, H, W, C]
        elev_sel = elevations[ids]            # [V]
        azim_sel = azimuths[ids]             # [V]
        
        geometry = get_batch_from_spherical(
            elevations=elev_sel,
            azimuths=azim_sel,
        )

        sample = {
            "img": imgs,                 # [V, H, W, C], np.float32 in [0,1]
            "txt": caption,              # single string
            "elevations": elev_sel,      # [V]
            "azimuths": azim_sel,        # [V]
            "epi_constraint_masks": geometry["epi_constraint_masks"],  # [V, V, H, W]   
            "plucker_embeds": geometry["plucker_embeds"],              # [V, V, 6]
        }

        return sample
