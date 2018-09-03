# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PIL import Image
import cv2
import numpy as np
import os
import pycocotools.mask as mask_util

from utils.colormap import colormap
# import utils.keypoints as keypoint_utils

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import ipdb

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def get_class_string(class_index, score, dataset):
    class_text = dataset[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text #+ ' {:0.2f}'.format(score).lstrip('0')


def vis_bbox(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img

def vis_one_image(
        im, im_name, output_dir, classes, boxes, masks=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf', show=False, W=224, H=224):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    color_list = colormap(rgb=True) / 255

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    boxes[:,0] *= W
    boxes[:, 2] *= W
    boxes[:,1] *= H
    boxes[:, 3] *= H
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    # torch to numpy
    # ipdb.set_trace()
    if masks is not None:
        # uint8
        masks = masks.astype('uint8')
        # rescale
        w_masks, h_masks, _ = masks.shape

    mask_color_id = 0
    # ipdb.set_trace()
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=1, alpha=box_alpha))

        if show_class:
            # (x, y) = (bbox[0], bbox[3] + 2) if classes[i] == 1 else (bbox[3], bbox[1] - 2)  # below for person or above for the rest
            x, y = (bbox[0], bbox[1] - 2)
            classes_i = np.argmax(classes[i])
            # print(get_class_string(classes_i, score, dataset), classes_i, score)
            ax.text(
                x, y,
                get_class_string(classes_i, score, dataset),
                fontsize=4,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if masks is not None:
            # ipdb.set_trace()
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e_down = masks[i, :, :]

            # Rescale mask
            e_pil = Image.fromarray(e_down)
            e_pil_up = e_pil.resize((H, W),Image.ANTIALIAS)
            e = np.array(e_pil_up)

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    print('result saved to {}'.format(os.path.join(output_dir, '{}'.format(output_name))))
    if show:
        plt.show()
    plt.close('all')
