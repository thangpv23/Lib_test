import math
import cv2
import sys
import torch
import numpy as np
import os
from pathlib import Path


CWD = Path(__file__).resolve().parent
sys.path.append(CWD.as_posix())
sys.path.append(CWD.parent.as_posix())
from mmcv import Config
from mmcv.ops import RoIPool

from mmdet.models import build_detector
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.utils import setup_multi_processes

import tqdm
from mmcv.runner import load_checkpoint, wrap_fp16_model
import mediafire_dl



class CardInference:
    def __init__(self, config=None, checkpoint=None, device='cuda:0'):
        if config is None:
            current_file_location = Path(__file__).resolve().parent.parent
            config = os.path.join(current_file_location, 'configs/yolact/yolact_r50_1x8_coco.py')
        
        if checkpoint is None:
            url = 'https://www.mediafire.com/file/h0m6mm51xb1wc98/best_weight_phase3.pth'
            if not  os.path.exists('../workdir/weight.pth'):
                
                os.mkdir('../workdir')
                mediafire_dl.download(url, '../workdir/weight.pth', quiet=False)
                checkpoint = '../workdir/weight.pth'
            else:
                checkpoint = '../workdir/weight.pth'

        cfg = Config.fromfile(config)
        # if args.cfg_options is not None:
        #     cfg.merge_from_dict(args.cfg_options)

        self.__input_size = (550, 550)
        self.__theta_threshold = 15
        self.__angle_threshold = 30

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None

        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        self.__model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.__model)
        checkpoint = load_checkpoint(self.__model, checkpoint)
        # if args.fuse_conv_bn:
        #     self.__model = fuse_conv_bn(self.__model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            self.__model.CLASSES = checkpoint['meta']['CLASSES']

        cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
        # build the data pipeline
        self.test_pipeline = Compose(cfg.data.test.pipeline)
        self.__model.to(device)
        self.__model.eval()

    def inference_on_image(self, path_to_img, visualize=False, save_path=None, corner=False, debug=False):
        result_dict = {}
        if os.path.isdir(path_to_img):
            
            for filename in tqdm.tqdm(os.listdir(path_to_img)):
                
                img_path = os.path.join(path_to_img, filename)
                if isinstance(img_path, str):
                    save_name = img_path.split("/")[-1].split(".")[0]
                else:
                    save_name = 'output'
                result_dict[filename+""] = []
                if save_path is not None and not os.path.exists(save_path):
                    os.makedirs(save_path)

                ret = []
                data, image = self.__preprocess(img_path, visualize)
                with torch.no_grad():
                    results = self.__model(return_loss=False, rescale=True, **data)[0]
                    bboxes_raw, segms_raw = results

        
                for idx, bbox in enumerate (bboxes_raw):
                    m, n = np.shape(bbox)

                    if m and n:
                            bbox = np.squeeze(bboxes_raw[idx])[:4]
                            result_dict[filename+""].append({
                                "class": self.__model.CLASSES[idx],
                                "bbox": bbox

                            })

                if save_path is not None:

                    self.__model.show_result(image, results, out_file=os.path.join(save_path, filename + '.jpg'))
            
            # for result in result_dict:
            #     print(result)

            #     print('\n')

            # print(result_dict)
            # return result_dict

        else:
            
            if isinstance(path_to_img, str):
                save_name = path_to_img.split("\\")[-1].split(".")[0]
            else:
                save_name = 'output'
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
            result_dict[save_name+""] = []
            data, image = self.__preprocess(path_to_img, visualize)

            with torch.no_grad():
                results = self.__model(return_loss=False, rescale=True, **data)[0]
                bboxes_raw, segms_raw = results
  
            
            # result_to_export, message_code_list = self.__post_process(image, results, corner=corner, debug=debug)

            for idx, bbox in enumerate (bboxes_raw):
            
                m, n = np.shape(bbox)

                if m and n:
                
                    bbox = np.squeeze(bboxes_raw[idx])[:4]
                    result_dict[save_name+""].append({
                        "class": self.__model.CLASSES[idx],
                        "bbox": bbox

                    })


        
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.__model.show_result(image, results, out_file=os.path.join(save_path, save_name + '.jpg'))
        
        # print(result_dict)
        
        return result_dict



    def __preprocess(self, image, visualize=False):
        """
        :param image: input image
        :type image: cv2.Mat (H,W,3)
        :return: data (as input of inference model), square_image (padding image to square)
        :rtype: dict, cv2.Mat (size, size, 3)
        """
        # To Square images
        if isinstance(image, str):
            image = cv2.imread(image)
        # if visualize:
        #     square_image = image
        # else:
        #     h, w, c = image.shape
        #     image_size = h if h >= w else w
        #     self.start_h = 0 if h > w else int((w - h) / 2)
        #     self.start_w = 0 if w > h else int((h - w) / 2)
        #     square_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        #     square_image[self.start_h:self.start_h + h, self.start_w:self.start_w + w] = image
        #     image = square_image

        square_image = image
        # image = cv2.resize(image, self.__input_size)

        device = next(self.__model.parameters()).device
        data = dict(img=image)

        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        if next(self.__model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            for m in self.__model.modules():
                assert not isinstance(m, RoIPool), 'CPU inference with RoIPool is not supported currently.'

            data["img_metas"] = data["img_metas"][0].data

        return data, square_image



   

    # def get_connected_group(self, node, already_seen, graph):
    #     result = []
    #     nodes = set([node])
    #     while nodes:
    #         node = nodes.pop()
    #         already_seen.add(node)
    #         nodes = nodes | graph[node] - already_seen
    #         result.append(node)
    #     return result, already_seen

    # def check_same_line(self, line1_param, line2_param):
    #     theta1, ro1, a1, b1 = line1_param
    #     theta2, ro2, a2, b2 = line2_param
    #     if (-7 <= theta1 - theta2 <= 7 or 173 <= abs(theta1 - theta2) <= 187) and abs(ro1 - ro2) < 70:
    #         # Verify
    #         if a1 is None:
    #             point1 = (b1, 0)
    #         else:
    #             point1 = (-a1 * b1 / (a1 * a1 + 1), -b1 / (a1 * a1 + 1))

    #         if a2 is None:
    #             point2 = (b2, 0)
    #         else:
    #             point2 = (-a2 * b2 / (a2 * a2 + 1), -b2 / (a2 * a2 + 1))

    #         cosine = (point1[0] * point2[0] + (point1[1] * point2[1])) / (
    #                 math.sqrt(point1[0] ** 2 + point1[1] ** 2) * math.sqrt(point2[0] ** 2 + point2[1] ** 2))
    #         if cosine > 1:
    #             cosine = 1
    #         elif cosine < -1:
    #             cosine = -1
    #         angle = math.acos(cosine)
    #         if angle * 180 / math.pi < 30:
    #             # print(point1, point2, angle)
    #             return True

    #     return False

    # def group_line(self, lines):
    #     line_params = [self.compute_line(line) for line in lines]
    #     connection_dict = {}
    #     for i in range(len(lines)):
    #         connection_dict[i] = set()

    #     for i in range(len(lines) - 1):
    #         for j in range(i + 1, len(lines)):
    #             if self.check_same_line(line_params[i], line_params[j]):
    #                 connection_dict[i].add(j)
    #                 connection_dict[j].add(i)
    #     connected_compponents = self.get_all_connected_groups(connection_dict)
    #     return connected_compponents

    # def dist(self, p1, p2):

    #     x0 = p1[0] - p2[0]
    #     y0 = p1[1] - p2[1]
    #     return x0 * x0 + y0 * y0

    # # Function to find the maximum
    # # distance between any two points
    # def maxDist(self, p):

    #     n = len(p)
    #     maxm = 0

    #     # Iterate over all possible pairs
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             # Update maxm
    #             maxm = max(maxm, self.dist(p[i], p[j]))

    #     # Return actual distance
    #     return math.sqrt(maxm)

  
    # def compute_group_line(self, lines, width, h):
    #     # print("lines", lines)
    #     if len(lines) == 1:
    #         return lines[0]

    #     points = [point for line in lines for point in line]
    #     points = np.array(points)
    #     X = points[:, 0]
    #     # X = X.reshape(-1, 1)

    #     Y = points[:, 1]

    #     a, b = np.polyfit(X, Y, 1)
    #     test_x = np.array([0, 1000])
    #     new_y = test_x * a + b
    #     pts1 = [test_x[0], new_y[0]]
    #     pts2 = [test_x[1], new_y[1]]

    #     x_mean = np.mean(X)
    #     distance = [abs(y - a * x - b) / math.sqrt(a ** 2 + 1) for x, y in zip(list(X), list(Y))]
    #     distance = sum(distance)
    #     horizontal_distance = sum([abs(x - x_mean) for x in X])
    #     if distance < horizontal_distance:
    #         return pts1, pts2
    #     else:
    #         return (x_mean, 0), (x_mean, h)

    # def line_intersection(self, line1, line2, w, h):
    #     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    #     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    #     def det(a, b):
    #         return a[0] * b[1] - a[1] * b[0]

    #     div = det(xdiff, ydiff)
    #     if div == 0:
    #         return None
    #         # raise Exception('lines do not intersect')

    #     d = (det(*line1), det(*line2))
    #     x = det(d, xdiff) / div
    #     y = det(d, ydiff) / div
    #     if x < -20 or x > w + 20 or y < -20 or y > h + 20:
    #         return None
    #     if x < 0: x = 0
    #     if x > w: x = w
    #     if y < 0: y = 0
    #     if y > h: y = h
    #     return int(x), int(y)

    
    # def order_points(self, pts):
    #     # sort the points based on their x-coordinates
    #     xSorted = pts[np.argsort(pts[:, 0]), :]

    #     # grab the left-most and right-most points from the sorted
    #     # x-roodinate points
    #     leftMost = xSorted[:2, :]
    #     rightMost = xSorted[2:, :]

    #     # now, sort the left-most coordinates according to their
    #     # y-coordinates so we can grab the top-left and bottom-left
    #     # points, respectively
    #     leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    #     (tl, bl) = leftMost

    #     # if use Euclidean distance, it will run in error when the object
    #     # is trapezoid. So we should use the same simple y-coordinates order method.

    #     # now, sort the right-most coordinates according to their
    #     # y-coordinates so we can grab the top-right and bottom-right
    #     # points, respectively
    #     rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    #     (tr, br) = rightMost

    #     # return the coordinates in top-left, top-right,
    #     # bottom-right, and bottom-left order
    #     return np.array([tl, tr, br, bl], dtype="float32")

    
  
