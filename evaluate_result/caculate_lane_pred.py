import os
import argparse
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm
from p_tqdm import t_map, p_map
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.ops as so

def continuous_cross_iou(xs, ys, width=30, img_shape=(1080, 1920, 3)):
    h, w, _ = img_shape
    image = Polygon([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])
    xs = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in xs
    ]
    ys = [
        LineString(lane).buffer(distance=width / 2., cap_style=1,
                                join_style=2).intersection(image)
        for lane in ys
    ]


    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if(x.union(y).area == 0):
                ious[i,j] = 0
            else:
                ious[i, j] = x.intersection(y).area / x.union(y).area
    return ious


def interp(points, n=50):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    # print(x, y)
    try:
        tck, u = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

        u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    except ValueError:
        del_index = []
        for i in range(len(x)-1):
            if abs(x[i+1] - x[i]) < 1:
                del_index.append(i+1)
        for i in reversed(del_index):
            del x[i]
            del y[i]
        tck, u = splprep([x, y], s=0, t=n, k=min(3, len(x) - 1))

        u = np.linspace(0., 1., num=(len(u) - 1) * n + 1)

    return np.array(splev(u, tck)).T


def culane_metric_v2(pred,
                  anno,
                  width=30,
                  iou_thresholds=[0.5],
                  official=True,
                  img_shape=(1080, 1920, 3)):
    _metric = {}
    for thr in iou_thresholds:
        tp = 0
        fp = 0 if len(anno) != 0 else len(pred)
        fn = 0 if len(pred) != 0 else len(anno)
        _metric[thr] = [tp, fp, fn]
    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred],
                           dtype=object)  # (4, 50, 2)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno],
                           dtype=object)  # (4, 50, 2)
    if official:
        ious = discrete_cross_iou(interp_pred,
                                  interp_anno,
                                  width=width,
                                  img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred,
                                    interp_anno,
                                    width=width,
                                    img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    _metric = {}
    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        _metric[thr] = [tp, fp, fn]
    return _metric


def load_culane_img_data(path):
    try:
        json_data = open(path.replace('png', 'json'))
        data = json.load(json_data)
        img_data = []
        for j in data['shapes']:
            img_data.append(j['points'])
    except:
        img_data = []
        #print(f"Error loading JSON for {path}")  # Add this
    return img_data


def load_culane_data(data_dir, file_list_path):

    data = []
    with open(file_list_path, 'r') as file_list:
        for line in file_list.readlines():
            line = line.strip()
            # print('line', line)
            if line != '':
            # for path in filepaths:
                img_data = load_culane_img_data(os.path.join(data_dir,line))
                data.append(img_data)
    return data

def eval_predictions(pred_dir,
                     anno_dir,
                     list_path,
                     iou_thresholds=[0.2, 0.4],
                     width=30,
                     official=True,
                     sequential=True):
    import logging
    logger = logging.getLogger(__name__)
    logger.info('Calculating metric for List: {}'.format(list_path))
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    print(f"Predictions: {predictions}")  # Add this
    print(f"Annotations: {annotations}")  # Add this
    # input()
    img_shape = (1920, 1080, 3) #(800, 1280, 3)(1080, 1920, 3)
    if sequential:
        results = map(
            partial(culane_metric_v2,
                    width=width,
                    official=official,
                    iou_thresholds=iou_thresholds,
                    img_shape=img_shape), predictions, annotations)
    else:
        from multiprocessing import Pool, cpu_count
        from itertools import repeat
        with Pool(cpu_count()) as p:
            results = p.starmap(culane_metric_v2, zip(predictions, annotations,
                        repeat(width),
                        repeat(iou_thresholds),
                        repeat(official),
                        repeat(img_shape)))

    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        logger.info('iou thr: {:.2f}, tp: {}, fp: {}, fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(
            thr, tp, fp, fn, precision, recall, f1))
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3)
        }

    if len(iou_thresholds) > 2:
        logger.info('mean result, total_tp: {}, total_fp: {}, total_fn: {},'
                'precision: {}, recall: {}, f1: {}'.format(total_tp, total_fp,
            total_fn, mean_prec, mean_recall, mean_f1))
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret


def main():
    args = parse_args()
    results = eval_predictions(args.pred_dir,
                                args.anno_dir,
                                args.list,
                                width=args.width,
                                official=args.official,
                                sequential=args.sequential)

    header = '=' * 20 + ' Results ({})'.format(
        os.path.basename(args.list)) + '=' * 20
    print(header)
    for metric, value in results.items():
        if isinstance(value, float):
            print('{}: {:.4f}'.format(metric, value))
        else:
            print('{}: {}'.format(metric, value))
    print('=' * len(header))


def parse_args():
    parser = argparse.ArgumentParser(description="Measure CULane's metric")
    parser.add_argument(
        "-p","--pred_dir",
        #default = './lane_label_dataset/201116145712/pred_modify_2',  # pred_modify  original_pred
        default = './ground_truth/lane_detection/220531153403/pred_modify_2',  
        help="Path to directory containing the predicted lanes")
    parser.add_argument(
        "-t","--anno_dir",
        default = './ground_truth/lane_detection/220531153403/gt',
        help="Path to directory containing the annotated lanes")
    parser.add_argument("--width",
                        type=int,
                        default=30,
                        help="Width of the lane")
    parser.add_argument("--list",
                        # nargs='+',
                        default = './ground_truth/lane_detection/220531153403/img.txt',
                        help="Path to txt file containing the list of files")
    parser.add_argument("--sequential",
                        action='store_true',
                        help="Run sequentially instead of in parallel")
    parser.add_argument("--official",
                        action='store_true',
                        help="Use official way to calculate the metric")

    return parser.parse_args()


if __name__ == '__main__':
    main()
