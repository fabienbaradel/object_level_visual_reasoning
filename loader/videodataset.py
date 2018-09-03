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
from  abc import abstractmethod, ABCMeta
import torch.nn.functional as F


class VideoDataset(data.Dataset):
    __metaclass__ = ABCMeta
    """
    Generic loader for videos dataset
    """

    def __init__(self, options, nb_classes=30,
                 dataset='train',
                 nb_crops=1,
                 #
                 usual_transform=False,
                 add_background=True,
                 #
                 video_dir='videos_256x256_30', mask_dir='masks/preds_100x100_50',
                 #
                 nb_obj_t_max=10, mask_confidence=0.5,
                 video_suffix='.mp4',
                 mask_size=28,
                 w=224, h=224):
        # Settings
        self.root = options['root']
        self.w = w
        self.h = h
        self.t = options['t']
        self.video_dir = video_dir
        self.nb_classes = nb_classes
        self.usual_transform = usual_transform
        self.nb_obj_max_t = nb_obj_t_max
        self.mask_confidence = mask_confidence
        self.video_dir_full = os.path.join(self.root, self.video_dir)
        self.video_suffix = video_suffix
        self.mask_dir = mask_dir
        self.mask_dir_full = os.path.join(self.root, self.mask_dir)
        self.nb_crops = nb_crops
        self.dataset = dataset
        self.w_mask, self.h_mask = mask_size, mask_size
        self.dict_video_length_fn = os.path.join(self.video_dir_full, 'dict_id_length.pickle')
        self.minus_len = 2
        self.add_background = add_background
        self.video_label_pickle = ''
        self.list_video = []

        # Retrieve the real shape of the super_video
        self.retrieve_w_and_h_from_dir()

        # Max length of a clip
        self.max_len_clip = 3 * self.real_fps  # sec by fps -> num of frames - 3 seconds

        # Video and length
        # self.list_video, self.dict_video_length = self.get_video_and_length()

    def store_dict_video_into_pickle(self, dict_video_label, dataset):
        with open(self.video_label_pickle, 'wb') as file:
            pickle.dump(dict_video_label, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Dict_video_label of {} saved! -> {}\n".format(dataset, self.video_label_pickle))

    def get_video_and_length(self):
        # Open the pickle file
        with open(self.dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)

        # Loop in each super_video dir to get th right super_video file
        list_video = []
        for video_id, length in dict_video_length.items():
            # Video id
            real_id = int(video_id.split('/')[1])
            list_video.append(real_id)

        return list_video, dict_video_length

    def retrieve_w_and_h_from_dir(self):
        _, w_h, fps = self.video_dir.split('_')
        w, h = w_h.split('x')
        self.real_w, self.real_h, self.real_fps = int(w), int(h), int(fps)
        self.ratio_real_crop_w, self.ratio_real_crop_h = self.real_w / self.w, self.real_h / self.h
        self.real_mask_w, self.real_mask_h = int(self.ratio_real_crop_w * self.w_mask), int(
            self.ratio_real_crop_h * self.h_mask)

    def time_sampling(self, video_len):
        # update the video_len on some dataset
        video_len = video_len - self.minus_len

        # Check that the super_video is not too long
        diff = self.max_len_clip - video_len

        # Change the start and adapt the length of the super_video
        if diff >= 0:
            start = 0
        else:
            start = random.sample(range(abs(diff)), 1)[0]

        video_len_up = video_len - start

        # Size of the sub-seq
        len_subseq = video_len_up / float(self.t)

        # Sample over each bin and add the start time
        if self.dataset != 'train' and self.nb_crops == 1:
            timesteps = [int((len_subseq / 2.0) + t * len_subseq + start) for t in range(self.t)]
        else:
            timesteps = [int(random.sample(range(int(len_subseq)), 1)[0] + t * len_subseq + start) for t in
                         range(self.t)]

        return timesteps

    def extract_frames(self, video_file, timesteps):

        with open(video_file, 'rb') as f:
            encoded_video = f.read()

            decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                                       frame_nums=timesteps,
                                                       width=self.real_w,
                                                       height=self.real_h)
            try:
                np_clip = np.frombuffer(decoded_frames, dtype=np.uint8)
                np_clip = np.reshape(np_clip,
                                     newshape=(len(timesteps), self.real_h, self.real_w, 3))
                np_clip = np_clip.transpose([3, 0, 1, 2])
                np_clip = np.float32(np_clip)
            except Exception as e:
                np_clip = decoded_frames
                print("cannot decode the stream...")
        return np_clip

    @staticmethod
    def load_masks(file):
        with open(file, 'rb') as f:
            masks = pickle.load(f, encoding='latin-1')
        return (masks['segms'], masks['boxes'])

    def retrieve_associated_masks(self, masks_file, video_len, timesteps, add_background_mask=True, start=0):
        T = len(timesteps)

        # update the timesteps dpending on the starting point
        timesteps = [t + start for t in timesteps]

        np_obj_id = np.zeros((T, self.nb_obj_max_t, 81)).astype(np.float32)
        np_bbox = np.zeros((T, self.nb_obj_max_t, 4)).astype(np.float32)
        np_masks = np.zeros((T, self.nb_obj_max_t, self.real_mask_h, self.real_mask_w)).astype(np.float32)
        np_max_nb_obj = np.asarray([self.nb_obj_max_t]).reshape((1,))

        try:
            # raise Exception
            segms, boxes = self.load_masks(masks_file)

            # Timestep factor
            factor = video_len / len(segms)
            timesteps = [int(t / factor) for t in timesteps]

            # Retrieve information
            list_nb_obj = []
            for t_for_clip, t in enumerate(timesteps):
                nb_obj_t = 0
                # Range of objects
                range_objects = list(range(2, 81))
                shuffle(range_objects)
                range_objects = [1] + range_objects
                for c in range_objects:
                    for i in range(len(boxes[t][c])):
                        if boxes[t][c][i] is not None and len(boxes[t][c]) > 0 and boxes[t][c][i][
                            -1] > self.mask_confidence:
                            # Obj id
                            np_obj_id[t_for_clip, nb_obj_t, c] = 1

                            # Bounding box
                            H, W = segms[t][c][i]['size']
                            x1, y1, x2, y2, _ = boxes[t][c][i]
                            x1, x2 = (x1 / W) * self.real_w, (x2 / W) * self.real_w
                            y1, y2 = (y1 / H) * self.real_h, (y2 / H) * self.real_h
                            np_bbox[t_for_clip, nb_obj_t] = [x1, y1, x2, y2]

                            # Masks
                            rle_obj = segms[t][c][i]
                            m = maskUtils.decode(rle_obj)  # Python COCO API
                            # My resize
                            # m = resize(m, (H, W), (self.real_mask_h, self.real_mask_w), thresold=0.1
                            # Resize
                            m_pil = Image.fromarray(m)
                            m_pil = m_pil.resize((self.real_mask_w, self.real_mask_h))
                            m = np.array(m_pil, copy=False)
                            np_masks[t_for_clip, nb_obj_t] = m

                            nb_obj_t += 1

                            # Break if too much objects
                            if nb_obj_t > (self.nb_obj_max_t - 1):
                                break
                    # Break if too much objects
                    if nb_obj_t > (self.nb_obj_max_t - 1):
                        break

                # Append
                list_nb_obj.append(nb_obj_t)

            # And now fill numpy array
            np_max_nb_obj[0] = max(list_nb_obj)


        except Exception as e:
            print("mask reading problem: ", e)
            ipdb.set_trace()
            np_max_nb_obj[0] = 1.

            # Add the background mask
        if add_background_mask:
            # Find the background pixels
            sum_masks = np.clip(np.sum(np_masks, 1), 0, 1)
            background_mask = 1 - sum_masks

            # Add meta data about background
            idx_bg_mask = int(np_max_nb_obj[0])
            idx_bg_mask -= 1 if self.nb_obj_max_t == idx_bg_mask else 0
            np_masks[:, idx_bg_mask] = background_mask
            np_obj_id[:, idx_bg_mask, 0] = 1
            np_bbox[:, idx_bg_mask] = [0, 0, 1, 1]

            # Update the number of mask
            np_max_nb_obj[0] = np_max_nb_obj[0] + 1 if np_max_nb_obj < self.nb_obj_max_t else np_max_nb_obj[0]

        return (np_obj_id, np_bbox, np_masks, np_max_nb_obj)

    def video_transform(self, np_clip, np_masks, np_bbox):

        # Random crop
        _, _, h, w = np_clip.shape
        w_min, h_min = random.sample(range(w - self.w), 1)[0], random.sample(range(h - self.h), 1)[0]
        # clip
        np_clip = np_clip[:, :, h_min:(self.h + h_min), w_min:(self.w + w_min)]
        # mask
        h_min_mask, w_min_mask = round((h_min / self.h) * self.h_mask), round((w_min / self.w) * self.w_mask)
        np_masks = np_masks[:, :, h_min_mask:(self.h_mask + h_min_mask), w_min_mask:(self.w_mask + w_min_mask)]
        # bbox
        np_bbox[:, :, [0, 2]] = np.clip(np_bbox[:, :, [0, 2]] - w_min, 0, self.w)
        np_bbox[:, :, [1, 3]] = np.clip(np_bbox[:, :, [1, 3]] - h_min, 0, self.h)
        # rescale to 0->1
        np_bbox[:, :, [0, 2]] /= self.w
        np_bbox[:, :, [1, 3]] /= self.h

        if self.usual_transform:
            # Div by 255
            np_clip /= 255.

            # Normalization
            np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)  # mean
            np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)  # std

        return np_clip, np_masks, np_bbox

    @abstractmethod
    def get_mask_file(self, id):
        return

    @abstractmethod
    def starting_point(self, id):
        return

    @abstractmethod
    def get_video_fn(self, id):
        return

    @abstractmethod
    def get_length(self, id):
        return

    @abstractmethod
    def get_target(self, index):
        return

    def extract_one_clip(self, id):

        # Length
        length = self.get_length(id)

        # Start of the video
        start = self.starting_point(id)  # 0 except for EPIC

        # Timesteps
        timesteps = self.time_sampling(length)

        return self.retrieve_clip_and_masks(id, timesteps, length, start)

    def retrieve_clip_and_masks(self, id, timesteps, length, start):
        # Clip
        np_clip = self.extract_frames(self.get_video_fn(id), timesteps)

        # Get the masks
        (np_obj_id, np_bbox, np_masks, np_max_nb_obj) = self.retrieve_associated_masks(self.get_mask_file(id),
                                                                                       length,
                                                                                       timesteps,
                                                                                       add_background_mask=self.add_background,
                                                                                       start=start)

        # Data processing on the super_video
        np_clip, np_masks, np_bbox = self.video_transform(np_clip, np_masks, np_bbox)

        return np_clip, np_masks, np_bbox, np_obj_id, np_max_nb_obj

    def extract_multiple_clips(self, id):
        # Length
        length = self.get_length(id)

        # Start of the video
        start = self.starting_point(id)  # 0 except for EPIC

        # NB_CROPS times
        list_timesteps = [self.time_sampling(length) for _ in range(self.nb_crops)]
        timesteps_union = list(set().union(*list_timesteps))
        timesteps_union.sort(key=int)  # sort numerically

        # Get the gathered data
        (np_clip_union, np_masks_union, np_bbox_union,
         np_obj_id_union, np_max_nb_obj_union) = self.retrieve_clip_and_masks(id,
                                                                              timesteps_union,
                                                                              length,
                                                                              start)

        # Loop and retreieve the data per clip
        list_np_clip, list_np_masks, list_np_bbox, list_np_obj_id = [], [], [], []
        for _, timesteps in enumerate(list_timesteps):
            # Get the idx from the timesteps union
            idx_union = [timesteps_union.index(time) for time in timesteps]

            # retrieve
            np_clip = np_clip_union[:, idx_union]
            np_masks = np_masks_union[idx_union]
            np_bbox = np_bbox_union[idx_union]
            np_obj_id = np_obj_id_union[idx_union]

            # append
            list_np_clip.append(np_clip)
            list_np_masks.append(np_masks)
            list_np_bbox.append(np_bbox)
            list_np_obj_id.append(np_obj_id)

        # stack
        np_clip = np.stack(list_np_clip)
        np_masks = np.stack(list_np_masks)
        np_obj_id = np.stack(list_np_obj_id)
        np_bbox = np.stack(list_np_bbox)

        return np_clip, np_masks, np_bbox, np_obj_id, np_max_nb_obj_union

    def __getitem__(self, index):
        """
          Args:
              index (int): Index
          Returns:
              dict: info about the video
        """

        try:
            # Get the super_video dir
            id = self.list_video[index]

            # Target
            torch_target = self.get_target(id)

            # If nb_crops = 1 it is easy
            if self.nb_crops == 1:
                np_clip, np_masks, np_bbox, np_obj_id, np_max_nb_obj = self.extract_one_clip(id)
            else:
                np_clip, np_masks, np_bbox, np_obj_id, np_max_nb_obj = self.extract_multiple_clips(id)

            # Video id
            np_uint8_id = np.fromstring(str(id), dtype=np.uint8)
            torch_id = torch.from_numpy(np_uint8_id)
            torch_id = F.pad(torch_id, (0, 300))[:300]

            # Torch world
            torch_clip = torch.from_numpy(np_clip)
            torch_masks = torch.from_numpy(np_masks)
            torch_obj_id = torch.from_numpy(np_obj_id)
            torch_obj_bboxs = torch.from_numpy(np_bbox)
            torch_max_nb_objs = torch.from_numpy(np_max_nb_obj)

            return {"target": torch_target,
                    "clip": torch_clip,
                    "mask": torch_masks,
                    "obj_id": torch_obj_id,
                    "obj_bbox": torch_obj_bboxs,
                    "max_nb_obj": torch_max_nb_objs,
                    "id": torch_id
                    }
        except Exception as e:
            return None

    def __len__(self):
        return len(self.list_video)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
