from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import traceback


import json
from multiprocessing import Pool




def chunker_list(seq, size):
    return list(seq[i::size] for i in range(size))


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-file',
        dest='output_file',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='./table_detect_dict.pkl',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for storing bbox',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--image_hash_dict_file',
        dest='image_hash_dict_file',
        help='image_hash_dict_file',
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
        'image_root', help='image_root', default=None, type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_hash_dict = json.load(open(args.image_hash_dict_file))


    def main(im_hash_list):
        if len(im_hash_list) < 1:
            return {}

        result = {}

        from caffe2.python import workspace

        from detectron.core.config import assert_and_infer_cfg
        from detectron.core.config import cfg
        from detectron.core.config import merge_cfg_from_file
        from detectron.utils.io import cache_url
        from detectron.utils.logging import setup_logging
        from detectron.utils.timer import Timer
        import detectron.core.test_engine as infer_engine
        import detectron.utils.c2 as c2_utils
        import detectron.utils.vis as vis_utils

        c2_utils.import_detectron_ops()

        # OpenCL may be enabled by default in OpenCV3; disable it because it's not
        # thread safe and causes unwanted GPU memory allocations.
        cv2.ocl.setUseOpenCL(False)

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        setup_logging(__name__)
        logger = logging.getLogger(__name__)

        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'
        model = infer_engine.initialize_model_from_cfg(args.weights)

        for i, im_hash in enumerate(im_hash_list):
            if image_hash_dict[im_hash] < 1:
                continue

            bbox_list = []
            im_name = image_hash_dict[im_hash][0]
            im_fn = os.path.basename(im_name)
            print(i, len(im_hash_list), len(image_hash_dict))
            item_id = im_fn.split("_")[0]

            im_name = os.path.join(args.image_root, item_id[:3], item_id[3:7], item_id[7:], im_fn)
            try:
                im = cv2.imread(im_name)
                timers = defaultdict(Timer)
                # t = time.time()
                with c2_utils.NamedCudaScope(0):
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, im, None, timers=timers
                    )
                # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                # for k, v in timers.items():
                #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

                boxes, cls_segms, cls_keyps, classes = vis_utils.convert_from_cls_format(
                    cls_boxes, cls_segms, cls_keyps)

                if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < args.thresh:
                    print("no detected")
                else:
                    for i in range(len(boxes)):
                        bbox = boxes[i, :4]
                        score = boxes[i, -1]
                        if score < args.thresh:
                            continue

                        bbox_list.append(
                            {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3]),
                             "score": float(score)})
                height = len(im)
                width = len(im[0])
                result[im_hash] = {"bboxes": bbox_list, "nums_bbox": len(bbox_list), "image_file": im_fn,
                                   "item_id": im_fn.split("_")[0], "width": int(width),
                                   "height": int(height), "aspect_ratio": float(width) / float(height)}
            except:
                traceback.print_exc()
                continue
        return result


    hash_list = list(image_hash_dict.keys())
    hash_chunks = chunker_list(hash_list, args.num_processes)

    # output_dir = os.path.dirname(args.image_hash_dict_file)
    # for i, hash_chunk in enumerate(hash_chunks):
    #     json.dump(hash_chunk, open(os.path.join(output_dir), "image_hash_chunk_%d.json" % i))
    # sys.exit()

    pool = Pool(args.num_processes)
    result_dict_list = pool.map(main, hash_chunks)
    merged_dict = {}
    for result_dict in result_dict_list:
        merged_dict.update(result_dict)

    json.dump(merged_dict, open(args.output_file, "w+"))
