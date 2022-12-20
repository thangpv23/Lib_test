import asyncio
from argparse import ArgumentParser
from mmdet.models import build_detector
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmdet.core.mask.structures import bitmap_to_polygon
import os
import tqdm
import gdown
import loguru
from mmcv import Config
import mmcv
import numpy as np
import mediafire_dl
import torch
import torch.distributed as dist
import cv2
from mmdet.datasets import  build_dataset


# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument('--path', default="", help='Image(s) path/dir')
#     parser.add_argument('--config', default="../configs/yolact/yolact_r50_1x8_coco.py", help='Config file')
#     parser.add_argument('--workdir', default="../workdir/", help='work dir')
#     parser.add_argument('--checkpoint', default="../workdir/weight.pth", help='Checkpoint file')
#     parser.add_argument('--out-path', default="../output/", help='Path to output file')
#     parser.add_argument('--out-file', default=False, help='option to visualize result')
#     parser.add_argument('--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--palette',
#         default='voc',
#         choices=['coco', 'voc', 'citys', 'random'],
#         help='Color palette used for visualization')
#     parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
#     args = parser.parse_args()
#     return args






def get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)

def draw_masks( img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha
    print(polygons)
    return img, polygons

def draw_bboxes( bboxes):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))

    print(polygons)
    return polygons

def export_file(model, img, result, out_file):
    show_result_pyplot(
        model,
        img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=out_file)



def main(args):
    # build the model from a config file and a checkpoint file

    workdir = args.workdir
    if (os.path.exists(workdir)):
        pass
    else:
        os.mkdir(workdir)

    cfg = Config.fromfile(args.config)
    path_to_img = args.path 
    out_file = args.out_file
    if os.path.exists(args.checkpoint):
        path_to_checkpoint = args.checkpoint
    else:
        url = 'https://www.mediafire.com/file/h0m6mm51xb1wc98/best_weight_phase3.pth'
        mediafire_dl.download(url, '../workdir/weight.pth', quiet=False)
        path_to_checkpoint = args.checkpoint
    
    # dataset = build_dataset(cfg.data.test) 
    model = init_detector(args.config, path_to_checkpoint, device=args.device)
    print(model.CLASSES[0])
    if os.path.isdir(path_to_img):

        for filename in tqdm.tqdm(os.listdir(args.path_to_img)):
            img = os.path.join(path_to_img, filename)
            result = inference_detector(model, img)
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            # draw segmentation masks
            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy()
                else:
                    segms = np.stack(segms, axis=0)
            # if out_file specified, do not show image in window
            # if out_file is not None:
            #     show = False

            if out_file:
                out_file = os.path.join(args.out_path, filename)
                export_file(model, img, result, out_file)
    else:
        filename = os.path.basename(path_to_img)
        img = path_to_img
        result = inference_detector(model, path_to_img)
    

        if out_file:
            out_file = os.path.join(args.out_path, filename)
            export_file(model, img, result, out_file)


if __name__ == '__main__':

    try:
        args = parse_args()
        main(args)
    except Exception as e:
        loguru.logger.exception(e)