#!/usr/bin/env python

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

"""Script for visualizing results saved in a detections.pkl file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys

import pickle
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_root',
        dest='image_root',
        help='image_root',
        default=None,
        type=str
    )
    parser.add_argument(
        '--detection_pickle_file',
        dest='detection_pickle_file',
        help='detections pickle file',
        default='',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.9,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='./tmp/vis-output',
        type=str
    )
    parser.add_argument(
        '--num_processes',
        dest='num_processes',
        help='num_processes',
        default=4,
        type=int
    )
    parser.add_argument(
        '--first',
        dest='first',
        help='only visualize the first k images',
        default=0,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def vis(image_root, detection_json_file, thresh, output_dir, limit=1, num_processes=4):
    print("start")
    image_hash_map = pickle.load(open(detection_json_file, "rb"))
    print("loaded", detection_json_file)
    image_hash_list = list(image_hash_map.keys())
    if limit > 0:
        image_hash_list = image_hash_list[:limit]

    def _do(image_hash):
        print(image_hash)
        image_info = image_hash_map[image_hash]

        item_id = image_info['image_file'].split("_")[0]
        item_id_idx = image_info['image_file'].split("_")[1]
        file_path = os.path.join(image_root, item_id[:3], item_id[3:7], item_id[7:], image_info['image_file'])

        im = cv2.imread(file_path)

        bboxes = image_info["bboxes"]
        if len(bboxes) < 1:
            return

        os.makedirs(os.path.join(output_dir, item_id[:3], item_id[3:7], item_id[7:]), exist_ok=True)

        for i, bbox in enumerate(bboxes):
            try:
                crop_im = im[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
                score = bbox["score"]
                output_file = os.path.join(output_dir, item_id[:3], item_id[3:7], item_id[7:],
                                           "{}_{}_{}_{}.jpg".format(item_id, item_id_idx, i, score))
                cv2.imwrite(output_file, crop_im)
            except:
                continue

    pool = Pool(num_processes)
    pool.map(_do, image_hash_list)


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.image_root,
        opts.detection_pickle_file,
        opts.thresh,
        opts.output_dir,
        limit=opts.first,
        num_processes=opts.num_processes
    )
