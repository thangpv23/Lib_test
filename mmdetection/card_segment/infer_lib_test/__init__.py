from argparse import ArgumentParser
import sys
from pathlib import Path
CWD = Path(__file__).resolve().parent
sys.path.append(CWD.as_posix())
sys.path.append(CWD.parent.as_posix())
from CardInference import *


    

# def main():
#     model = CardInference()
#     # img = 'D:\AI\CV\FTech\\test_lib\mmdetection\\test_lib_img\\1c1891b0-d25a-425d-b711-4850a65dc44920170812103840_121038156.jpg'
#     # img = 'D:\AI\CV\FTech\\test_lib\mmdetection\\test_lib_img\\0f37fbb4-dd86-4762-a23b-50a83696bb60.jpg'
#     img = 'D:\AI\CV\FTech\\test_lib\mmdetection\\test_lib_img'
#     model.inference_on_image(img, save_path='D:\AI\CV\\FTech\\test_lib\\mmdetection\output') 




# if __name__ == '__main__':

    # try:

    #     main()
    # except Exception as e:
    #     loguru.logger.exception(e)