
import os

import torch
import numpy as np
import colorsys
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from vibe.rt.rt_mpt import MptLive

from vibe.models.vibe import VIBE_Demo
from vibe.utils.renderer import Renderer
from vibe.dataset.inference import Inference
from vibe.utils.smooth_pose import smooth_pose
from vibe.data_utils.kp_utils import convert_kps
from vibe.utils.pose_tracker import run_posetracker
from vibe.rt.single_img_inference import SingleImgInference
from vibe.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    # prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
from vibe.rt.utils import prepare_rendering_results_rt

os.environ['PYOPENGL_PLATFORM'] = 'egl'


class HiddenStates:
    """
    Manage GRU hidden states across multiple target.
    """
    def __init__(self):
        self._hidden = {}  # Dict: {person_id, gru_hidden_state}

    def update_person_id(self, tracking_results):
        """
        preserve person_ids which are still being tracked.
        Init newly tracked people with 'None' state.
        """
        new_hidden: dict = {}
        for person_id in tracking_results.keys():
            # Not a New person
            if person_id in self._hidden.keys():  # a person still being tracked
                new_hidden[person_id] = self._hidden[person_id]
            else:  # A new person
                new_hidden[person_id] = None  # use None and pytorch will take care of the initialization.
        self._hidden = new_hidden  # The lost trackers are deleted

    def get_hidden(self, person_id):
        return self._hidden[person_id]

    def set_hidden(self, person_id, hidden_state):
        self._hidden[person_id] = hidden_state


class RtVibe:
    # Color for different person. Do not use this directly, use _mesh_color() instead
    _mesh_color_20 = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in range(20)}

    @torch.no_grad()
    def __init__(self, render=True):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.render = render
        self.wireframe = False
        self.sideview = False

        self.mpt_live = MptLive(
            device=self.device,
            batch_size=1,
            display=False,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )
        self.vibe_model = VIBE_Demo(
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(self.device)

        # GRU hidden states for previous tracking person id. dict:{id, hidden}
        # self.prev_tracking_hidden: dict = {}
        self.hidden_states = HiddenStates()

        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        self.vibe_model.load_state_dict(ckpt, strict=False)
        self.vibe_model.eval()
        print(f'Loaded pretrained weights from \"{pretrained_file}\"')

        self.renderer = None  # lazy initialize with first image to get W,H

    @classmethod
    def _mesh_color(cls, k):
        k = k % 20
        return cls._mesh_color_20[k]

    def _track(self, image: np.ndarray):
        # Image to tracking result.
        tracking_results = self.mpt_live(image)  # BUG: This will always set frame_num to 1.
        # Keep hidden state for each individual.
        self.hidden_states.update_person_id(tracking_results)
        return tracking_results

    def _vibe_infer(self, image: np.ndarray, track_res: dict):
        # RT-VIBE
        orig_height, orig_width = image.shape[:2]
        vibe_results = {}
        for person_id in list(track_res.keys()):
            bboxes = track_res[person_id]['bbox']
            frames = track_res[person_id]['frames']
            joints2d = None
            bbox_scale = 1.1

            dataset = SingleImgInference(
                image=image,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )
            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            seqlen = 1
            batch_size = 1

            # Use corresponding hidden state!
            self.vibe_model.encoder.gru_final_hidden = self.hidden_states.get_hidden(person_id)
            batch = next(iter(dataloader))
            batch = batch.unsqueeze(0)
            batch = batch.to(self.device)

            # batch_size, seqlen = batch.shape[:2]
            output = self.vibe_model(batch)[-1]
            # Update hidden state for person_id
            self.hidden_states.set_hidden(person_id, self.vibe_model.encoder.gru_final_hidden)

            # Just to be consist with original format
            pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
            pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
            pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
            pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
            pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
            smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            joints2d_img_coord = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=224,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': None,  # This refs to pose tracker result.
                'joints2d_img_coord': joints2d_img_coord,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            vibe_results[person_id] = output_dict
        return vibe_results

    @torch.no_grad()
    def __call__(self, image: np.ndarray):

        # Tracking
        tracking_results = self._track(image)
        # Run VIBE inference
        vibe_results = self._vibe_infer(image, tracking_results)

        # Render
        if self.render:
            self._render_and_show(image, vibe_results)

        return vibe_results

    def _render_and_show(self, image: np.ndarray, vibe_results: dict):
        """Render and show render result"""
        orig_height, orig_width = image.shape[:2]

        if self.renderer is None:
            self.renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=self.wireframe)

        num_frames = 1
        # ========= Render results as a single video ========= #

        # prepare results for rendering
        frame_results = prepare_rendering_results_rt(vibe_results)

        img = image

        for person_id, person_data in frame_results[0].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mc = self._mesh_color(person_id)

            mesh_filename = None

            img = self.renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )

            if self.sideview:
                side_img = np.zeros_like(img)
                side_img = self.renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0, 1, 0],
                )

        if self.sideview:
            img = np.concatenate([img, side_img], axis=1)

        cv2.imshow('Video', img)
        cv2.waitKey(1)