from card_segment.infer_lib_test import CardInference
import loguru

try:    
    model = CardInference()

    img = 'D:\AI\CV\\FTech\Lib_test\\test_lib_img\\1c1891b0-d25a-425d-b711-4850a65dc44920170812103840_121038156.jpg'

    results = model.inference_on_image( path_to_img=img)
    # print(results)

    model.inference_on_image(path_to_img= img, save_path='output')
except Exception as e:
    loguru.logger.exception(e)