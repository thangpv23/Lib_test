

Requirements
```
pip install mmcv-full==1.4.7
```
Build
```
python mmdetection/setup.py bdist_wheel
```

Installation
```
pip install dist/CardSegment-2.25.1-py3-none-any.whl --force-reinstall
```

# Usage
Import and use
```
from mmdetection.card_segment.infer_lib_test import CardInference

### load card segmentation model

model = CardInference() Auto download checkpoint
### or load card model with your own config file:
model = CardInference(config=absolute_path_to_config, checkpoint=absolute_path_to_checkpoint)
model = CardInference(checkpoint=absolute_path_to_checkpoint) if there is local checkpoint else checkpoint will be downloaded

# define path to img
img = path_to_img
# or img as numpy array
img = path_to_folder_img
```
Use cases
```

# extract and transform id card from image
results = model.inference_on_image(path_to_img(s)) ## return dict result on img or imgs


# extract and transform id card from image, save to save path
model.inference_on_image(img, save_path=path_to_save_folder)


```
Example
```
from mmdetection.card_segment.infer_lib_test import CardInference

model = CardInference(checkpoint='/home/trangtnt/projects/yolact_weights/r50_1x8_coco/latest.pth')

img = '/home/trangtnt/projects/ekyc_data/ekyc_segmentation_v1/test/data_01/test_augment/0_back_1000019.jpg'

results = model.inference_on_image(img)

model.inference_on_image(img, save_path='test_card_package')

```
