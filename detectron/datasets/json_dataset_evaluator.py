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

"""Functions for evaluating results computed for a json dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import six
import uuid

from pycocotools.cocoeval import COCOeval

from detectron.core.config import cfg
from detectron.utils.io import save_object
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)

class CustomCOCOeval(COCOeval):
    def set_param(self, params, cocoGt=None):
        self.params = params

        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((29,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.85, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, iouThr=.86, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=.87, maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.88, maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=.89, maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=.90, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(1, iouThr=.94, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, iouThr=.98, maxDets=self.params.maxDets[2])
            stats[11] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[13] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[14] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
            stats[15] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[16] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])
            stats[17] = _summarize(1, iouThr=.75, areaRng='small', maxDets=self.params.maxDets[2])
            stats[18] = _summarize(1, iouThr=.75, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[19] = _summarize(1, iouThr=.75, areaRng='large', maxDets=self.params.maxDets[2])
            stats[20] = _summarize(1, iouThr=.95, areaRng='small', maxDets=self.params.maxDets[2])
            stats[21] = _summarize(1, iouThr=.95, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[22] = _summarize(1, iouThr=.95, areaRng='large', maxDets=self.params.maxDets[2])
            stats[23] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[24] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[25] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[26] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[27] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[28] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        # self.iouThrs = np.linspace(.90, .95, int(np.round((.95 - .90) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        # self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 500. ** 2], [500. ** 2, 600. ** 2], [600. ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'segmentations_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_segms_results_file(
        json_dataset, all_boxes, all_segms, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_segmentation_eval(json_dataset, res_file, output_dir)
    else:
        logger.warning(
            '{} eval ignored as annotations are undisclosed on test: {} ignored'
            .format("Segmentation", json_dataset.name)
        )
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_segms_results_file(
    json_dataset, all_boxes, all_segms, res_file
):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "segmentation": [...],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_segms_results_one_category(
            json_dataset, all_boxes[cls_ind], all_segms[cls_ind], cat_id))
    logger.info(
        'Writing segmentation results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which /always produces strings/ cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does.
        if six.PY3:
            for r in results:
                rle = r['segmentation']
                if 'counts' in rle:
                    rle['counts'] = rle['counts'].decode("utf8")

        json.dump(results, fid)


def _coco_segms_results_one_category(json_dataset, boxes, segms, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    assert len(segms) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        rles = segms[i]

        if isinstance(dets, list) and len(dets) == 0:
            continue

        dets = dets.astype(np.float)
        scores = dets[:, -1]

        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'segmentation': rles[k],
              'score': scores[k]}
              for k in range(dets.shape[0])])

    return results


def _do_segmentation_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'segmentation_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def evaluate_boxes(
    json_dataset, all_boxes, output_dir, use_salt=True, cleanup=False
):
    res_file = os.path.join(
        output_dir, 'bbox_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_bbox_results_file(json_dataset, all_boxes, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_detection_eval(json_dataset, res_file, output_dir)
    else:
        logger.warning(
            '{} eval ignored as annotations are undisclosed on test: {} ignored'
            .format("Bbox", json_dataset.name)
        )
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_bbox_results_file(json_dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_bbox_results_one_category(
            json_dataset, all_boxes[cls_ind], cat_id))
    logger.info(
        'Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_bbox_results_one_category(json_dataset, boxes, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        if isinstance(dets, list) and len(dets) == 0:
            continue
        dets = dets.astype(np.float)
        scores = dets[:, -1]
        xywh_dets = box_utils.xyxy_to_xywh(dets[:, 0:4])
        xs = xywh_dets[:, 0]
        ys = xywh_dets[:, 1]
        ws = xywh_dets[:, 2]
        hs = xywh_dets[:, 3]
        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results


def _do_detection_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    # coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval = CustomCOCOeval(json_dataset.COCO, coco_dt, 'bbox')
    params = Params(iouType='bbox')
    coco_eval.set_param(params, json_dataset.COCO)
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'detection_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def _log_detection_eval_metrics(json_dataset, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    # IoU_lo_thresh = 0.9
    # IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    logger.info(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    logger.info('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        logger.info('{:.1f}'.format(100 * ap))
    logger.info('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


def evaluate_box_proposals(
    json_dataset, roidb, thresholds=None, area='all', limit=None, class_specific=False
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    gt_classes = np.zeros(0)
    num_pos = 0
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_areas = entry['seg_areas'][gt_inds]
        valid_gt_inds = np.where(
            (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        _gt_classes = entry["gt_classes"][valid_gt_inds]
        assert gt_boxes.shape[0] == _gt_classes.shape[0]
        gt_classes = np.hstack((gt_classes, _gt_classes))
        num_pos += len(valid_gt_inds)
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        boxes = entry['boxes'][non_gt_inds, :]
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]
        overlaps = box_utils.bbox_overlaps(
            boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False))
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # and get the iou amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)

    if not class_specific:
        gt_overlaps = np.sort(gt_overlaps)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps, 'num_pos': num_pos}
    else:
        gt_classes_unique = np.unique(gt_classes)
        recalls = np.zeros((gt_classes_unique.shape[0], thresholds.shape[0]))
        # compute recall for each category and each iou threshold
        for i, category_id in enumerate(gt_classes_unique):
            inds = (gt_classes == category_id)
            num_pos_per_category = float(inds.sum())
            for j, thresh in enumerate(thresholds):
                recalls[i][j] = (
                    gt_overlaps[inds] >= thresh
                ).sum() / num_pos_per_category
        ar = recalls.mean(axis=1).mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps, 'num_pos': num_pos}

def evaluate_keypoints(
    json_dataset,
    all_boxes,
    all_keypoints,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'keypoints_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_keypoint_results_file(
        json_dataset, all_boxes, all_keypoints, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_keypoint_eval(json_dataset, res_file, output_dir)
    else:
        logger.warning(
            '{} eval ignored as annotations are undisclosed on test: {} ignored'
            .format("Keypoints", json_dataset.name)
        )
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_keypoint_results_file(
    json_dataset, all_boxes, all_keypoints, res_file
):
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_keypoints):
            break
        logger.info(
            'Collecting {} results ({:d}/{:d})'.format(
                cls, cls_ind, len(all_keypoints) - 1))
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_kp_results_one_category(
            json_dataset, all_boxes[cls_ind], all_keypoints[cls_ind], cat_id))
    logger.info(
        'Writing keypoint results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_kp_results_one_category(json_dataset, boxes, kps, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(kps) == len(image_ids)
    assert len(boxes) == len(image_ids)
    use_box_score = False
    if cfg.KRCNN.KEYPOINT_CONFIDENCE == 'logit':
        # This is ugly; see utils.keypoints.heatmap_to_keypoints for the magic
        # indexes
        score_index = 2
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'prob':
        score_index = 3
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'bbox':
        use_box_score = True
    else:
        raise ValueError(
            'KRCNN.KEYPOINT_CONFIDENCE must be "logit", "prob", or "bbox"')
    for i, image_id in enumerate(image_ids):
        if len(boxes[i]) == 0:
            continue
        kps_dets = kps[i]
        scores = boxes[i][:, -1].astype(np.float)
        if len(kps_dets) == 0:
            continue
        for j in range(len(kps_dets)):
            xy = []

            kps_score = 0
            for k in range(kps_dets[j].shape[1]):
                xy.append(float(kps_dets[j][0, k]))
                xy.append(float(kps_dets[j][1, k]))
                xy.append(1)
                if not use_box_score:
                    kps_score += kps_dets[j][score_index, k]

            if use_box_score:
                kps_score = scores[j]
            else:
                kps_score /= kps_dets[j].shape[1]

            results.extend([{'image_id': image_id,
                             'category_id': cat_id,
                             'keypoints': xy,
                             'score': kps_score}])
    return results


def _do_keypoint_eval(json_dataset, res_file, output_dir):
    ann_type = 'keypoints'
    imgIds = json_dataset.COCO.getImgIds()
    imgIds.sort()
    coco_dt = json_dataset.COCO.loadRes(res_file)
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, ann_type)
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    eval_file = os.path.join(output_dir, 'keypoint_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    coco_eval.summarize()
    return coco_eval
