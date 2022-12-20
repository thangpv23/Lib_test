import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import os
import tqdm

from numba import jit, cuda
def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('--img', default="D:\AI\CV\mmdetection\data\sub_sample\\test\\blx\\", help='Image file')
    parser.add_argument('--img', default="D:\AI\CV\mmdetection\data\sub_sample\\test\\all\\", help='Image file')
    parser.add_argument('--config', default="D:\AI\CV\mmdetection\mmdetection\configs\yolact\yolact_r50_1x8_coco.py", help='Config file')
    parser.add_argument('--checkpoint', default="D:\AI\CV\mmdetection\mmdetection\epoch_55.pth", help='Checkpoint file')
    # parser.add_argument('--out-file', default="D:\AI\CV\mmdetection\data\sub_sample\output\\blx\\", help='Path to output file')
    parser.add_argument('--out-file', default="D:\AI\CV\mmdetection\data\sub_sample\output\\all_2nd_test\\", help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='voc',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args



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
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for filename in tqdm.tqdm(os.listdir(args.img)):
        img = os.path.join(args.img, filename)
        result = inference_detector(model, img)
        out_file = os.path.join(args.out_file, filename)

        export_file(model, img, result, out_file)



if __name__ == '__main__':
    args = parse_args()
    main(args)