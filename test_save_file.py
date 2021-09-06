import glob
import tensorflow as tf
import numpy as np 
import os
import matplotlib.image as mpimg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_ids = []
THRESHOLD = 0.6
for dirname, _, filenames in os.walk('data/test/DICOM'):
    for filename in filenames:
        test_ids.append(os.path.join(dirname, filename))
print(len(test_ids))
test_ids = sorted(test_ids)

import pydicom

def transform_to_hu(medical_image, image):
    hu_image = image * medical_image.RescaleSlope + medical_image.RescaleIntercept
    hu_image[hu_image < -1024] = -1024
    return hu_image

def window_image(image, window_center, window_width):
    window_image = image.copy()
    image_min = window_center - (window_width / 2)
    image_max = window_center + (window_width / 2)
    window_image[window_image < image_min] = image_min
    window_image[window_image > image_max] = image_max
    return window_image

def resize_normalize(image):
    image = np.array(image, dtype=np.float64)
    image -= np.min(image)
    image /= np.max(image)
    return image

def read_dicom(path, window_widht, window_level):
    image_medical = pydicom.dcmread(path)
    image_data = image_medical.pixel_array

    image_hu = transform_to_hu(image_medical, image_data)
    image_window = window_image(image_hu.copy(), window_level, window_widht)
    image_window_norm = resize_normalize(image_window)

    image_window_norm = np.expand_dims(image_window_norm, axis=2)   # (512, 512, 1)
    image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)   # (512, 512, 3)
    return image_ths

from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = np.expand_dims(data, axis=-1).astype(np.float32)
    out_data=np.concatenate([data, data, data], axis=-1)
    return out_data

def my_encode(mask1):
    # x = np.zeros((mask1.shape[0], mask1.shape[1], 1))
    # x[mask1[:,:,1] > THRESHOLD] = 1
    # x[mask1[:,:,2] > THRESHOLD] = 2
    return np.argmax(mask1, axis=-1).astype(np.uint8)

from segmentation_models import Unet, Linknet
BACKBONE = 'efficientnetb7'
model = Unet(BACKBONE, encoder_weights=None, classes=3, input_shape=(512, 512, 3), activation='softmax')  

mod_path = '/home/ubuntu/Tu/workspace/demoo/kfold_weights_0905/'

from tqdm import tqdm
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, ]
    elif axis == 2:
        return img[:, ::-1, ]
    elif axis == 3:
        return img[::-1, ::-1, ]
    else:
        return img

def predict(image, TTA=False):
    TTAS = [0,1,2]
    pred = None
    model_list = glob.glob(mod_path + 'unet-efficientnetb7-albu-bce_loss-heavy_augfold-*.h5')
    count = 0
    for fold_model_path in model_list:
        # if 'fold-1' in fold_model_path or 'fold-3' in fold_model_path or 'fold-4' in fold_model_path:
        if 'fold-1' in fold_model_path or 'fold-0' in fold_model_path:
            model.load_weights(fold_model_path)
            count += 1
            if TTA:
                for tta_mode in TTAS:
                    image_aug = flip(image, axis=tta_mode)
                    image_aug = np.expand_dims(image_aug, axis=0)
                    if pred is None:
                        pred = flip(np.squeeze(model.predict(image_aug)), axis=tta_mode)
                    else:
                        pred += flip(np.squeeze(model.predict(image_aug)), axis=tta_mode)
            else:
                if pred is None:
                    pred = np.squeeze(model.predict(np.expand_dims(image, axis=0)))
                else:
                    pred += np.squeeze(model.predict(np.expand_dims(image, axis=0)))
        else: 
            continue
    if TTA:
        pred = pred/(count * len(TTAS))
    else:
        pred = pred/count
    # pred = (pred > THRESHOLD).astype(np.uint8)
    return pred
import cv2
def load_test(lists):
    for i in lists:
        # img = read_dicom(i, 100, 50)
        img = read_xray(i)
        pred = predict(img)
        out_x = my_encode(pred).astype(np.uint8)
        # out_x = np.expand_dims(out_x, axis=-1)
        filename = os.path.join('test_results_softmax', i.split('/')[-2], i.split('/')[-1]).replace('dcm', 'png')

        if not os.path.exists(os.path.join('test_results_softmax', i.split('/')[-2])):
            os.makedirs(os.path.join('test_results_softmax', i.split('/')[-2]))
        cv2.imwrite(filename, out_x)


preds_string=[]
for i in tqdm(range(0, len(test_ids[:]), 64)):
    sample = test_ids[i:i+64].copy()
    # if i>-4480:
    preds = load_test(sample)
    # for label_code in [1,2]:
    #     tmp = np.array([], dtype='uint8')
    #     for s in preds:
    #         x = np.equal(s[:,:,0].flatten(), label_code)*255
    #         tmp = np.hstack([tmp, x])
    #     enc = rle_to_string(rle_encode(np.array(tmp)))

    #     preds_string.append(enc)

# import pandas as pd
# sample_submission = pd.read_csv('data/sample_submission.csv')
# sample_submission['EncodedPixels'] = preds_string
# sample_submission.to_csv('submission_groupkfold-unet-effb7-fold_0.35-5fs.csv', index=False)