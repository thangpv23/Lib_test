import io

from loguru import logger
from PIL import Image
import numpy as np  

def prepare_data(data):
    try:
        pil_img = Image.open(io.BytesIO(data))
        open_cv_image = np.array(pil_img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  
        # print("pil_img: ", pil_img.size)
        return open_cv_image
    except Exception as e:
        logger.exception("Cannot prepare data:", e)
        exit()
