import os
from os import path as osp

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class SingleImage(Dataset):  # modified from multi_person_tracker.data.ImageFolder
    def __init__(self, image: np.ndarray):
        self.image = image

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return to_tensor(img)
