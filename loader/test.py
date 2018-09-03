import argparse
from loader.vlog import VLOG
import time
import sys
import ipdb
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import os
from PIL import Image
import matplotlib

matplotlib.use('agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import utils.vis as vis_utils
from utils.other import *

dict_dataset = {'vlog': VLOG}


def get_coco_names():
    # read the txt file
    fn = './coco_names.txt'
    with open(fn) as f:
        list_coco_obj = f.readlines()
    list_coco_obj = [x.strip() for x in list_coco_obj]

    # Insert background for the first obj
    list_coco_obj.insert(0, 'background')

    return list_coco_obj


def main(options):
    # Dataset
    loader = VLOG if options['dataset'] == 'vlog' else None

    # Loader
    videodataset = loader(options, dataset='test', mask_size=100)
    print("Size of the dataset = {}".format(len(videodataset)))

    # Loop and show
    for i, input in enumerate(videodataset):
        # Get the data
        target = input['target']
        clip = input['clip']
        mask = input['mask']
        obj_id = input['obj_id']
        obj_bbox = input['obj_bbox']
        max_nb_obj = input['max_nb_obj']
        id = input["id"]

        # Video id
        bytes_id = id.cpu().numpy()  # it has been padded
        video_id = decode_videoId(bytes_id)

        # Shape
        C, T, H, W = clip.shape

        # Clip to numpy array
        clip_np = input['clip'].cpu().numpy().astype('uint8')

        # Loop over time
        for t in range(T):
            # Image
            img_np = clip_np[:, t]
            image = img_np.transpose([1, 2, 0])

            # Saved
            image_fn = '{}_maskRCNN_100X100_50.png'.format(t+1)
            output_dir = './img'
            vis_utils.vis_one_image(
                image,  # BGR -> RGB for visualization # (W,H,3) np.array
                image_fn,  # outfile filename
                output_dir,  # output dir
                obj_id[t].cpu().numpy(),
                obj_bbox[t].cpu().numpy(),
                mask[t].cpu().numpy(),
                None,
                dataset=get_coco_names(),
                box_alpha=0.4,
                show_class=True,
                thresh=0.5,
                kp_thresh=10,
                show=True,
                W=W,
                H=H
            )

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the loader')
    parser.add_argument('--dataset', metavar='D',
                        default='vlog',
                        help='Dataset')
    parser.add_argument('--root', metavar='DIR',
                        default='../data/vlog',
                        help='Path to the dataset')
    parser.add_argument('--t', default=2, type=int,
                        metavar='H', help='Number of timesteps to extract from a super_video')

    # Args
    args, _ = parser.parse_known_args()

    # Dict
    options = vars(args)

    main(options)
