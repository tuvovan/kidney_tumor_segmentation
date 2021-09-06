from albumentations.core.composition import OneOf
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import matplotlib.image as mpimg

import os

import segmentation_models
import albumentations as A

# Instantiate augments
# we can apply as many augments we want and adjust the values accordingly
# here I have chosen the augments and their arguments at random
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

transforms = A.Compose([
            A.Rotate(limit=40),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(p=0.5),              
            A.RandomRotate90(p=0.5),
        ])

import albumentations as A

transforms_ = A.Compose([

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=512, width=512, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                # A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        )])

def aug_fn(image, mask, img_size=512):
    data = {"image":image, "mask":mask}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_msk = aug_data["mask"]
    aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    aug_msk = tf.image.resize(aug_msk, size=[img_size, img_size])
    return aug_img, aug_msk

def process_data(image, label):
    aug_img, label = tf.numpy_function(func=aug_fn, inp=[image, label], Tout=tf.float32)
    return aug_img, label


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
#     image_window_norm = image_window

    image_window_norm = np.expand_dims(image_window_norm, axis=2)   # (512, 512, 1)
    image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)   # (512, 512, 3)
    return image_ths

def get_mask_innest(mask):
    mask_1 = mask == 0.0
    mask_3_int = mask_1.astype(float)
    return mask_3_int

def get_mask_middle(mask):
    mask_1 = mask == 0.00392157
    mask_3_int = mask_1.astype(float)
    return mask_3_int

def get_mask_outest(mask):
    mask_1 = mask == 0.00784314
    mask_3_int = mask_1.astype(float)
    return mask_3_int

import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator

MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]


def my_encode(mask1):
    x = np.zeros((mask1.shape[0], mask1.shape[1], 1))
    x[mask1[:,:,0] > 0.5] = 0
    x[mask1[:,:,1] > 0.5] = 1
    x[mask1[:,:,2] > 0.5] = 2
    return x

import os
import sys
import random
import warnings

import numpy as np
# import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import tensorflow as tf
from functools import partial


# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
BATCH_SIZE = 4
TRAIN_PATH_IMAGE = 'data/train/DICOM/'
TRAIN_PATH_MASKS = 'data/train/Label/'
TEST_PATH = 'data/test/DICOM/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Get train and test IDs
total_train_inp_files = []
total_train_tar_files = []
test_ids = []
for dirname, _, filenames in os.walk('data/train/DICOM'):
    for filename in filenames:
        total_train_inp_files.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('data/train/Label'):
    for filename in filenames:
        total_train_tar_files.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('data/test/DICOM'):
    for filename in filenames:
        test_ids.append(os.path.join(dirname, filename))

train_inp_files = sorted(total_train_inp_files)[:int(0.8*len(total_train_inp_files))]
train_tar_files = sorted(total_train_tar_files)[:int(0.8*len(total_train_inp_files))]

valid_inp_files = sorted(total_train_inp_files)[int(0.8*len(total_train_inp_files)):]
valid_tar_files = sorted(total_train_inp_files)[int(0.8*len(total_train_inp_files)):]


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

# @tf.function
def train_map_func(inp_path, tar_path, augment=False):
    inp_path = tf.compat.as_str_any(inp_path)
    img = read_xray(inp_path).astype(np.float32)
    mask = mpimg.imread(tar_path)
    mask = np.around(mask, 8)

    m_in = get_mask_innest(mask)
    m_md = get_mask_middle(mask)
    m_ot = get_mask_outest(mask)
    
    m_in = np.expand_dims(m_in, axis=-1)
    m_md = np.expand_dims(m_md, axis=-1)
    m_ot = np.expand_dims(m_ot, axis=-1)
    
    m_fn = np.concatenate([m_in, m_md, m_ot], axis=-1).astype(np.float32)
    m_fn = my_encode(m_fn)
    m_fn = tf.keras.utils.to_categorical(m_fn, num_classes=3)
    # m_fn = my_encode(m_fn)
    # m_fn = np.concatenate([m_fn, m_fn, m_fn], axis=-1).astype(np.float32)
    if augment:
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            m_fn = tf.image.flip_left_right(m_fn)

        if tf.random.uniform(()) > 0.4:
            img = tf.image.flip_up_down(img)
            m_fn = tf.image.flip_up_down(m_fn)

        if tf.random.uniform(()) > 0.5:
            img = tf.image.rot90(img, k=1)
            m_fn = tf.image.rot90(m_fn, k=1)

        if tf.random.uniform(()) > 0.45:
            img = tf.image.random_saturation(img, 0.7, 1.3)

        if tf.random.uniform(()) > 0.45:
            img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.cast(img, tf.float32), tf.cast(m_fn, tf.float32)

def valid_map_func(inp_path, tar_path, augment=False):
    inp_path = tf.compat.as_str_any(inp_path)
    img = read_xray(inp_path).astype(np.float32)
    mask = mpimg.imread(tar_path)
    mask = np.around(mask, 8)

    m_in = get_mask_innest(mask)
    m_md = get_mask_middle(mask)
    m_ot = get_mask_outest(mask)
    
    m_in = np.expand_dims(m_in, axis=-1)
    m_md = np.expand_dims(m_md, axis=-1)
    m_ot = np.expand_dims(m_ot, axis=-1)
    
    m_fn = np.concatenate([m_in, m_md, m_ot], axis=-1).astype(np.float32)
    m_fn = my_encode(m_fn)
    m_fn = tf.keras.utils.to_categorical(m_fn, num_classes=3)
    # m_fn = my_encode(m_fn)
    # m_fn = np.concatenate([m_fn, m_fn, m_fn], axis=-1).astype(np.float32)
    if augment:
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            m_fn = tf.image.flip_left_right(m_fn)

        if tf.random.uniform(()) > 0.4:
            img = tf.image.flip_up_down(img)
            m_fn = tf.image.flip_up_down(m_fn)

        if tf.random.uniform(()) > 0.5:
            img = tf.image.rot90(img, k=1)
            m_fn = tf.image.rot90(m_fn, k=1)

        if tf.random.uniform(()) > 0.45:
            img = tf.image.random_saturation(img, 0.7, 1.3)

        if tf.random.uniform(()) > 0.45:
            img = tf.image.random_contrast(img, 0.8, 1.2)
    return tf.cast(img, tf.float32), tf.cast(m_fn, tf.float32)

def test_map_func(inp_path):
    inp_path = tf.compat.as_str_any(inp_path)
    img = read_dicom(inp_path, 100, 50).astype(np.float32)
    
    return img

def _fixup_shape(images, mask):
    images.set_shape([BATCH_SIZE, 512, 512, 3])
    mask.set_shape([BATCH_SIZE, 512, 512, 3])
    # weights.set_shape([None])
    return images, mask

def get_train_dataset(train_inp_files, train_tar_files):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inp_files, train_tar_files))
    train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(train_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(get_augmented)
    # train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(get_augmented, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(process_data, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=8)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(_fixup_shape)
    return train_dataset

def get_valid_dataset(valid_inp_files, valid_tar_files):
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_inp_files, valid_tar_files))
    valid_dataset = valid_dataset.map(lambda item1, item2: tf.numpy_function(valid_map_func, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(get_augmented)
    # train_dataset = train_dataset.map(lambda item1, item2: tf.numpy_function(get_augmented, [item1, item2], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.map(_fixup_shape)
    return valid_dataset


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# %env SM_FRAMEWORK=tf.keras
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import binary_focal_dice_loss
from segmentation_models.metrics import iou_score, f1_score
from keras_unet_collection import models
from keras_unet_collection.losses import focal_tversky


from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def bce_log_focal(y_true, y_pred):
    return 0.4*bce_logdice_loss(y_true, y_pred) + 0.6*binary_focal_dice_loss(y_true, y_pred)

# https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html#dice_coe
def dice_coe(output, target, axis = None, smooth=1e-10):
    output = output[:,:,:,1:]
    target = target[:,:,:,1:]
    output = tf.dtypes.cast( tf.math.greater(output, 0.5), tf. float32 )
    target = tf.dtypes.cast( tf.math.greater(target, 0.5), tf. float32 )
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

# https://www.kaggle.com/kool777/training-hubmap-eda-tf-keras-tpu
def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
    # thanks @mfernezir for catching a bug in an earlier version of this implementation!

def calculate_stradified_kfold(files):
    labels = []
    for file in files:
        mask = mpimg.imread(file)
        mask_unique = np.unique(mask)
        labels.append(len(list(mask_unique))-1)
    return labels

label_list = calculate_stradified_kfold(total_train_tar_files)

group_by_folder = [i.split('/')[3] for i in total_train_inp_files]

def add_sample_weights(image, mask):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tf.constant([0.026555240154266357, 0.9738444089889526, 0.999600350856781])
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(mask, tf.int32))

  return image, mask, sample_weights


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, GroupKFold
from segmentation_models.losses import bce_dice_loss

# fold = KFold(n_splits=5, shuffle=True, random_state=2021)
gkf = GroupKFold(n_splits = 5)

for fold,(tr_idx, val_idx) in enumerate(gkf.split(total_train_inp_files, label_list, groups=group_by_folder)):
    
    print('#'*35); print('############ FOLD ',fold+1,' #############'); print('#'*35)
    print(f'Image Size: {512}, Batch Size: {BATCH_SIZE}')
    
    # CREATE TRAIN AND VALIDATION SUBSETS
    TRAINING_FILENAMES = [total_train_inp_files[fi] for fi in tr_idx]
    TRAINING_MASKNAMES = [total_train_tar_files[fi] for fi in tr_idx]
    # if P['OVERLAPP']:åc
    #     TRAINING_FILENAMES += [ALL_TRAINING_FILENAMES2[fi] for fi in tr_idx]
    
    VALIDATION_FILENAMES = [total_train_inp_files[fi] for fi in val_idx]
    VALIDATION_MASKNAMES = [total_train_tar_files[fi] for fi in val_idx]
    STEPS_PER_EPOCH = 1 * len(TRAINING_FILENAMES) // BATCH_SIZE
    
    # BUILD MODEL
    K.clear_session()
    # with strategy.scope(): 
    BACK_BONE = 'efficientnetb7'
    model = segmentation_models.Unet(BACK_BONE, encoder_weights='imagenet', classes=3, input_shape=(512, 512, 3), activation='softmax')  
    dice_loss = segmentation_models.losses.DiceLoss(class_weights=np.array([0.5, 2])) 
    # binary_loss = segmentation_models.losses.BinaryCELoss(class_weights=np.array([0.5,2]))
    from adamp import AdamP

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 5e-4),
    # model.compile(optimizer = AdamP(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, weight_decay=1e-2),
                    # loss = focal_tversky,
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    # loss = bce_logdice_loss,
                    # loss = segmentation_models.losses.BinaryFocalLoss(),
                    metrics=[dice_coe, 'accuracy'])
        
    # CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint('kfold_weights_0905/unet-%s-albu-bce_loss-heavy_augfold-%d.h5'%(BACK_BONE, fold),
                                 verbose=1,monitor='val_dice_coe',
                                 mode='max',save_best_only=True)
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coe',mode = 'max', patience=10, restore_best_weights=True)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode = 'min', factor=0.1, patience=5, min_lr=0.00001)
        
    print(f'Training Model Fold {fold+1}...')
    history = model.fit(
        get_train_dataset(TRAINING_FILENAMES, TRAINING_MASKNAMES),
        epochs = 30,
        steps_per_epoch = STEPS_PER_EPOCH,
        callbacks = [checkpoint, reduce,early_stop],
        validation_data = get_valid_dataset(VALIDATION_FILENAMES, VALIDATION_MASKNAMES),
        verbose=1
    )   
    
    #with strategy.scope():
    #    model = tf.keras.models.load_model('/kaggle/working/model-fold-%i.h5'%fold, custom_objects = {"dice_coe": dice_coe})
    
    # SAVE METRICS
    M = {}
    metrics = ['loss','dice_coe']
    for fm in metrics:
        M['val_'+fm] = []
    m = model.evaluate(get_valid_dataset(VALIDATION_FILENAMES, VALIDATION_MASKNAMES),return_dict=True)
    for fm in metrics:
        M['val_'+fm].append(m[fm])
    
    # PLOT TRAINING
    # https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords
    # if P['DISPLAY_PLOT']:        
    plt.figure(figsize=(15,5))
    n_e = np.arange(len(history.history['dice_coe']))
    plt.plot(n_e,history.history['dice_coe'],'-o',label='Train dice_coe',color='#ff7f0e')
    plt.plot(n_e,history.history['val_dice_coe'],'-o',label='Val dice_coe',color='#1f77b4')
    x = np.argmax( history.history['val_dice_coe'] ); y = np.max( history.history['val_dice_coe'] )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max dice_coe\n%.2f'%y,size=14)
    plt.ylabel('dice_coe',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    plt2 = plt.gca().twinx()
    plt2.plot(n_e,history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(n_e,history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.legend(loc=3)
    plt.savefig('kfold_weights_0905/unet-%s-albu-bce_loss-fold-heavy_aug-%d.png'%(BACK_BONE, fold), dpi=600)

    del model
