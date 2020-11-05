#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from natsort import natsorted
from tifffile import imread

from dataset.preprocess import augment


class ScaredDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(ScaredDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        if self.split == 'train' or self.split == 'validation' or self.split == 'validation_all':
            self.prefix = 'train'
        else:
            self.prefix = 'test'
        self._read_data()
        self._augmentation()

    def _read_data(self):
        self.left_data = []

        left_fold = os.path.join(self.datadir, self.prefix, 'left')
        self.left_data = [os.path.join(left_fold, img) for img in os.listdir(left_fold)]

        self.left_data = natsorted(self.left_data)

    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        result = {}

        left_fname = self.left_data[idx]
        result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)

        right_fname = left_fname.replace('left', 'right')
        result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)

        disp_fname = left_fname.replace('left', 'disparity').replace('.png', '.tiff')
        result['disp'] = np.array(imread(disp_fname)).astype(np.float32).squeeze(0)

        occ_fname = left_fname.replace('left', 'occlusion')
        result['occ_mask'] = np.array(Image.open(occ_fname)).astype(np.uint8) == 128
        result['disp'][result['occ_mask']] = 0.0

        result = augment(result, self.transformation)

        return result
