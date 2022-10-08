import os
from os import path as osp

import cv2
import numpy as np
from torch.utils.data import Dataset

from vibe.data_utils.img_utils import get_single_image_crop_demo
from vibe.utils.smooth_bbox import get_all_bbox_params


class SingleImgInference(Dataset):
    """
    Extract Patch
    """
    def __init__(self, image: np.ndarray, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224):
        self.image = image
        # self.image_file_names = [
        #     osp.join(image_folder, x)
        #     for x in os.listdir(image_folder)
        #     if x.endswith('.png') or x.endswith('.jpg')
        # ]
        # self.image_file_names = sorted(self.image_file_names)
        # self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False

        self.norm_joints2d = np.zeros_like(self.joints2d)

        # if self.has_keypoints:
        #     bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        #     bboxes[:, 2:] = 150. / bboxes[:, 2:]
        #     self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T
        #
        #     self.image_file_names = self.image_file_names[time_pt1:time_pt2]
        #     self.joints2d = joints2d[time_pt1:time_pt2]
        #     self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[idx]

        j2d = self.joints2d[idx] if self.has_keypoints else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)
        if self.has_keypoints:
            return norm_img, kp_2d
        else:
            return norm_img