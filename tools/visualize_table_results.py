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

import detectron.utils.vis as vis_utils
import pickle
import numpy as np
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import json

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument("--use_img_ar_filter", type=str2bool, nargs='?', default=True,
                        help='use image aspect ratio filter')
    parser.add_argument("--min_img_ar", type=float, default=0.37, help='min image aspect ratio')
    parser.add_argument("--max_img_ar", type=float, default=4.0, help='max image aspect ratio')
    parser.add_argument("--use_size_filter", type=str2bool, nargs='?', default=True)
    parser.add_argument("--min_height", type=int, default=75)
    parser.add_argument("--min_width", type=int, default=75)
    parser.add_argument("--use_bbox_ar_filter", type=str2bool, nargs='?', default=True,
                        help='use bbox aspect ratio filter')
    parser.add_argument("--min_bbox_ar", type=float, default=0.333, help='min bbox aspect ratio')
    parser.add_argument("--max_bbox_ar", type=float, default=6.0, help='max bbox aspect ratio')
    parser.add_argument("--use_text_count_filter", type=str2bool, nargs='?', default=True)
    parser.add_argument("--text_count_threshold", type=int, default=10)
    parser.add_argument("--use_color_count_filter", type=str2bool, nargs='?', default=True)
    parser.add_argument("--color_bin_step", type=int, default=5)
    parser.add_argument("--color_count_threshold", type=int, default=7)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def vis(image_root, detection_json_file, thresh, output_dir, limit=1, num_processes=4, opts=None):
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

        if opts.use_img_ar_filter:
            if image_info['aspect_ratio'] < opts.min_img_ar or image_info['aspect_ratio'] > opts.max_img_ar:
                return [{"skip": True, "skip_type": "img_ar_filter", "type": "image",
                         "image_file": image_info['image_file'],
                         "item_id": item_id, "value": image_info['aspect_ratio']}]

        if opts.use_size_filter:
            if image_info['height'] < opts.min_height or image_info['width'] < opts.min_width:
                return [{"skip": True, "skip_type": "img_size_filter", "type": "image",
                         "image_file": image_info['image_file'],
                         "item_id": item_id, "value": "%s,%s" % (image_info['height'], image_info['width'])}]

        file_path = os.path.join(image_root, item_id[:3], item_id[3:7], item_id[7:], image_info['image_file'])

        im = cv2.imread(file_path)

        bboxes = image_info["bboxes"]
        if len(bboxes) < 1:
            return [
                {"skip": True, "skip_type": "none_bbox", "type": "bbox", "image_file": image_info['image_file'],
                 "item_id": item_id}]

        encoded_bboxes = []
        bbox_skip_list = []
        for bbox in bboxes:
            bbox_width = bbox["x2"] - bbox["x1"]
            bbox_height = bbox["y2"] - bbox["y1"]

            if opts.use_bbox_ar_filter:
                bbox_ar = float(bbox_width) / float(bbox_height)
                if bbox_ar < opts.min_bbox_ar or bbox_ar > opts.max_bbox_ar:
                    bbox["aspect_ratio"] = bbox_ar
                    bbox_skip_list.append({"skip": True, "skip_type": "bbox_ar_filter", "type": "bbox",
                                           "image_file": image_info['image_file'],
                                           "item_id": item_id, "value": json.dumps(bbox)})
                    continue


            if opts.use_text_count_filter:
                if bbox_ar < opts.min_bbox_ar or bbox_ar > opts.max_bbox_ar:
                    bbox["aspect_ratio"] = bbox_ar
                    bbox_skip_list.append({"skip": True, "skip_type": "bbox_ar_filter", "type": "bbox",
                                           "image_file": image_info['image_file'],
                                           "item_id": item_id, "value": json.dumps(bbox)})
                    continue
            encoded_bboxes.append(np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], bbox["score"]]))

        try:
            vis_utils.vis_one_image_custom(
                im[:, :, ::-1],
                image_info['image_file'],
                output_dir,
                item_id,
                np.array(encoded_bboxes),
                thresh=thresh,
                box_alpha=0.8
            )
            if len(bbox_skip_list) < 1:
                return [{"skip": False, "type": "bbox"}]
            return bbox_skip_list
        except:
            return [
                {"skip": True, "skip_type": "vis_except", "type": "image", "image_file": image_info['image_file'],
                 "item_id": item_id}]

    pool = Pool(num_processes)
    results = pool.map(_do, image_hash_list)
    print("multiprocessing completed")
    skip_file = open(os.path.join(output_dir, "skip.json"), "w+")
    for result in results:
        if result["skip"]:
            del result["skip"]
            skip_file.write(",".join(result) + "\n")
    print("write the skip file is completed")


if __name__ == '__main__':
    opts = parse_args()
    vis(
        opts.image_root,
        opts.detection_pickle_file,
        opts.thresh,
        opts.output_dir,
        limit=opts.first,
        num_processes=opts.num_processes,
        opts=opts
    )
