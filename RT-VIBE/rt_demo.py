import argparse

import cv2
from vibe.rt import RtVibe


def main(args):
    rt_vibe = RtVibe()

    if args.camera:
        video = 0
    elif args.vid_file:
        video = args.vid_file
    else:
        video = 'sample_video.mp4'

    rt_vibe.render = args.render
    rt_vibe.sideview = args.sideview

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result = rt_vibe(frame)
        else:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--render', action='store_false',
                        help='render output')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--camera', action='store_true',
                        help='use camera as input')

    args = parser.parse_args()

    main(args)



