from keras import backend as K
import numpy as np
import os
import cv2
from PIL import Image
import random
from sklearn.preprocessing import normalize as s_norm
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

def normalize(arr, axis=1):
    x = np.transpose(arr, (0, 2, 1))            # (N, W, H)
    x2d = x.reshape(-1, x.shape[-1])            # (N*W, H)
    s_colwise = s_norm(x2d, axis=1)              # normalize over H
    s_colwise = s_colwise.reshape(x.shape)      # (N, W, H)
    return np.transpose(s_colwise, (0, 2, 1))   # back to (N, H, W)


def read_images_and_masks(image_directory, mask_directory, SIZE=256):
    image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
    mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    images = os.listdir(image_directory)
    print(len(images))
    for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
        if (image_name.split('.')[-1] == 'tif'):
            #             print(image_directory+image_name)
            image = cv2.imread(image_directory + image_name, 0)
            if image is None:
                raise FileNotFoundError(f"Failed to read: {image_directory}. Check the path/extension and permissions.")
            image = Image.fromarray(image)
            image = np.array(image.resize((SIZE, SIZE)))
            #         image = (image - image.min()) / image.max()
            image_dataset.append(image)

    # Iterate through all images in Uninfected folder, resize to 64 x 64
    # Then save into the same numpy array 'dataset' but with label 1

    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if (image_name.split('.')[-1] == 'tif'):
            image = cv2.imread(mask_directory + image_name, 0)
            if image is None:
                raise FileNotFoundError(f"Failed to read: {mask_directory}. Check the path/extension and permissions.")
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            #         Normalize masks to be binary and save to the dataset
            image = np.array(image)
            mask_dataset.append(np.where(image > 0, 255, image))

    # Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    #     image_dataset = np.expand_dims(np.array(image_dataset),3)
    # Do not normalize masks, just rescale to 0 to 1.
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

    return image_dataset, mask_dataset

def show_datapoint(image_dataset, mask_dataset, idx):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(image_dataset[idx], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(mask_dataset[idx], (256, 256)), cmap='gray', vmin=0, vmax=1)

    plt.show()

def show_random_datapoint(image_dataset, mask_dataset):
    image_number = random.randint(0, len(image_dataset))
    print(image_number)
    show_datapoint(image_dataset, mask_dataset, image_number)