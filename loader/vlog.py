from __future__ import print_function
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import random
from PIL import Image
import numpy as np
import ipdb
import pickle
from pycocotools import mask as maskUtils
import lintel
import time
from torch.utils.data.dataloader import default_collate
from random import shuffle
from loader.videodataset import VideoDataset


class VLOG(VideoDataset):
    """
    Loader for the VLOG dataset
    """

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)

        # Dict_video_label pickle
        self.video_label_pickle = os.path.join(self.video_dir_full, 'dict_video_label_{}.pickle'.format(self.dataset))

        # Videos paths
        self.list_video, self.dict_video_length, self.dict_video_label = self.get_videos()

    def get_videos(self):
        # Open the pickle file
        with open(self.dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)

        # Load dict_video_label
        dict_video_label = self.load_or_create_dict_video_label()

        # Intersect
        list_video_from_label = list(dict_video_label.keys())
        list_video_from_length = list(dict_video_length.keys())
        list_video_from_length = [v[1:].split('clip')[0] for v in list_video_from_length]
        list_video = list(set(list_video_from_length) & set(list_video_from_label))

        return list_video, dict_video_length, dict_video_label

    def load_or_create_dict_video_label(self):
        if os.path.isfile(self.video_label_pickle):
            with open(self.video_label_pickle, 'rb') as file:
                dict_video_label = pickle.load(file)
        else:
            # Load the label matrix
            label_npy_path = os.path.join(self.root, 'meta', 'hand_object', 'hand_object.npy')
            matrix_label = np.load(label_npy_path)

            # split number
            if self.dataset == 'test':
                split_id = [0]
            elif self.dataset == 'val':
                split_id = [3]
            elif self.dataset == 'train':
                split_id = [1, 2]
            elif self.dataset == 'train+val':
                split_id = [1, 2, 3]
            else:
                raise NameError

            # Get the corresponding files
            manifest_file = os.path.join(self.root, 'meta', 'manifest.txt')
            split_file = os.path.join(self.root, 'meta', 'splitId.txt')

            ## Read each file into a list
            # avi file
            with open(manifest_file) as f:
                list_video_path = f.readlines()
            list_video_path = [x.strip() for x in list_video_path]

            # split
            with open(split_file) as f:
                list_split = f.readlines()
            list_split = [x.strip() for x in list_split]
            dict_video_label = {}
            print("\n* Creating the dictionnary: video -> label")
            for i, video in enumerate(list_video_path):
                if i % 20000 == 0:
                    print("{}/{} ".format(i, len(list_video_path)))
                # Look if it is a good file for the split
                if int(list_split[i]) in split_id:
                    dict_video_label[video] = matrix_label[i]
            print("")

            # Store
            self.store_dict_video_into_pickle(dict_video_label, self.dataset)

        return dict_video_label

    def starting_point(self, id):
        return 0

    def get_mask_file(self, id):
        # Get the approriate masks
        mask_fn = os.path.join(self.mask_dir_full, id, 'clip.pkl')

        return mask_fn

    def get_video_fn(self, id):
        # Video location
        video_location = os.path.join(self.video_dir_full, id, 'clip' + self.video_suffix)
        return video_location

    def get_length(self, id):
        return self.dict_video_length['/{}clip'.format(id)]

    def get_target(self, id):
        label = self.dict_video_label[id]
        label = np.clip(label, 0, 1)
        return torch.FloatTensor(label)
