#  Modify the multi-person-tracker to receive image instead of folder
import numpy as np
import torch
from torch.utils.data import DataLoader

from vibe.rt.single_image import SingleImage

from multi_person_tracker import MPT
from multi_person_tracker import Sort


class MptLive(MPT):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_tracker = None

    @torch.no_grad()
    def run_tracker(self, dataloader):
        '''
        Run tracker on an input video
        Change: Disable timing functions as this will be called at each frame instead of each batch.

        :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
        :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
        '''

        # Dirty frame-by-frame implementation
        if self.prev_tracker is None:
            self.tracker = Sort()
        else:
            self.tracker = self.prev_tracker  # initialize tracker

        trackers = []
        for batch in dataloader:
            batch = batch.to(self.device)

            predictions = self.detector(batch)

            for pred in predictions:
                bb = pred['boxes'].cpu().numpy()
                sc = pred['scores'].cpu().numpy()[..., None]
                dets = np.hstack([bb, sc])
                dets = dets[sc[:, 0] > self.detection_threshold]

                # if nothing detected do not update the tracker
                if dets.shape[0] > 0:
                    track_bbs_ids = self.tracker.update(dets)
                else:
                    track_bbs_ids = np.empty((0, 5))
                trackers.append(track_bbs_ids)
        self.prev_tracker = self.tracker
        return trackers

    @torch.no_grad()
    def __call__(self, image: np.ndarray, output_file=None):
        '''
        Execute MPT and return results as a dictionary of person instances
        Change: input an image instead of image_folder

        :param video (ndarray): input video tensor of shape NxHxWxC
        :return: a dictionary of person instances
        '''

        image_dataset = SingleImage(image)

        dataloader = DataLoader(image_dataset, batch_size=1, num_workers=0)  # self.batch_size -> 1

        trackers = self.run_tracker(dataloader)
        # if self.display:
        #     self.display_results(image_folder, trackers, output_file)  # TODO: fix display

        if self.output_format == 'dict':
            result = self.prepare_output_tracks(trackers)
        elif self.output_format == 'list':
            result = trackers

        return result


