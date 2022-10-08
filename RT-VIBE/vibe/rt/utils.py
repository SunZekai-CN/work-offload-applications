import numpy as np
from collections import OrderedDict


def prepare_rendering_results_rt(vibe_results):
    nframes = 1
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[0][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
            }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k, v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]: frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results