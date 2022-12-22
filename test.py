from card_segment.infer_lib_test import CardInference



model = CardInference()

#inference on single image
img = 'D:\AI\CV\\FTech\Lib_test\\test_lib_img\\1c1891b0-d25a-425d-b711-4850a65dc44920170812103840_121038156.jpg'
results = model.inference_on_image( path_to_img=img)
model.inference_on_image(path_to_img= img, save_path='output')

#inference on folder images
# imgs_folder =  'D:\AI\CV\\FTech\Lib_test\\test_lib_img' 
# results = model.inference_on_image( path_to_img=imgs_folder)


print(results)


