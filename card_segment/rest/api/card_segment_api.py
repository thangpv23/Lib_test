import fastapi
import sys

sys.path.append(".")

import time

from loguru import logger
from typing import List

from fastapi import UploadFile, File

from processing_data.prepare_data import prepare_data
from processing_data.validation_error import ValidationError

'''
from app.main_api import fashion_processing_unit as fpu

'''
from infer_lib_test import CardInference


router = fastapi.APIRouter()


@router.post("/Card_segment/upload", response_model=list)
async def upload_files(files: List[UploadFile] = File(...)):
    logger.info("---Upload and Detection processing---")

    final_result=[]
    uploaded_imgs = []
    uploaded_imgs_name = []
    non_uploaded_files = []
    for file in files:
        if file.content_type.startswith("image/") is False:
            non_uploaded_files.append(file.filename)
            pass
        else:
            # img_name = file.filename
            data = await file.read()
            # data = cv2.imread(file)
            
            # data =  file.read()
            img = prepare_data(data)
            # img = cv2.imread(img)
            uploaded_imgs.append (img)
            uploaded_imgs_name.append(file.filename)
            print(file.filename)



    logger.info("upload Done")

    try:
        model_predict = CardInference()
        for i, img in enumerate( uploaded_imgs):

            final_result.append ( model_predict.inference_on_image(img, filename=uploaded_imgs_name[i]))

        if final_result:

            logger.info("Card Segment result: {}".format((final_result)))

        # Show files isn't image that don't upload
        for i in non_uploaded_files:
            logger.info("File isn't image: {}".format(i))

        return final_result
        """
        detection = [
            {
                img_0:{
                    "area_0":{class:, score:, xmin:, ymin:, xmax:, ymax:},
                    "area_1":{class, score, xmin, ymin, xmax, ymax}
                }
            }, 
            {
                img_1:{
                    "area_2":{class, score, xmin, ymin, xmax, ymax},
                    "area_3":{class, score, xmin, ymin, xmax, ymax}
                }
            },
            ...
        ]
        """

    except ValidationError as ve:
        return fastapi.Response(content=ve.error_msg, status_code=ve.status_code)
        pass
    except Exception as e:

        logger.exception(e)
        return fastapi.Response(content=str(e), status_code=500)
        pass
