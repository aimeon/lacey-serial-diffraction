# from unet_model import simple_unet_model
import cv2
import numpy as np
from keras.utils import normalize
from matplotlib import pyplot as plt
from keras import models
from PIL import Image
from crystal_finder_from_instamatic import find_crystals
import os
from scipy import stats



# def get_model():
#     return simple_unet_model(256, 256, 1)


def prediction(model, image, patch_size):
    segm_img = np.zeros(image.shape[:2])  # Array with zeros to be filled with segmented values
    patch_num = 1
    for i in range(0, image.shape[0], 256):  # Steps of 256
        for j in range(0, image.shape[1], 256):  # Steps of 256
            print("Started processing patch number ", patch_num, " at position ", i, j)
            # print(i, j)
            single_patch = image[i:i + patch_size, j:j + patch_size]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
            single_patch_shape = single_patch_norm.shape[:2]
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            pred = model.predict(single_patch_input)

            # Force NumPy
            if hasattr(pred, "numpy"):
                pred = pred.numpy()

            pred = np.squeeze(pred)
            single_patch_prediction = (pred > 0.5).astype(np.uint8)

            segm_img[
                i:i + single_patch_prediction.shape[0],
                j:j + single_patch_prediction.shape[1]
            ] += single_patch_prediction

            print("Finished processing patch number ", patch_num, " at position ", i, j)
            patch_num += 1
    return segm_img


from skimage import filters


def reduce_noise(image, method='gaussian', sigma=1):
    """
    Function to reduce noise in the image using a Gaussian or Median filter.
    Parameters:
    - image: numpy array representing the image.
    - method: filtering method to be used, either 'gaussian' or 'median'.
    - sigma: Standard deviation for Gaussian kernel. The higher sigma, the more blurring/smoothing (default is 1).
    Returns:
    - Smoothed image with reduced noise.
    """
    if method.lower() == 'gaussian':
        return filters.gaussian(image, sigma=sigma)
    elif method.lower() == 'median':
        return filters.median(image)
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'median'.")



#### I was not satisfied with the result ###
def inpaint_grid(model, img, patch_size):
    img_processed = Image.fromarray(img)
    img_processed =np.array(img_processed.resize((512, 512)))
    mask = prediction(model, img_processed, patch_size)
    mask = np.uint8(mask)

    return cv2.inpaint(cv2.resize(img, (512, 512)), mask, 1, cv2.INPAINT_TELEA)


def remove_grid(image, model, patch_size=256, plot=True):
    """"
    Function to remove a grid pattern from an input image using a given model.

    Parameters:
    image: numpy array representing the image.
    model: The ML model loaded with keras used for prediction.
    patch_size: int - Size of patches to be used for prediction (default is 256).*
    plot: bool - If True, display the intermediate steps and the final result (default is True).

    Returns:
    image_without_grid: numpy array image with the grid removed.

    *Since the model was trained on patches 256*256, we patchify our image again,
     and make predictions on each patch individually.
    """
    segmented_image = prediction(model, image, patch_size)


    if plot:
        plt.hist(segmented_image.flatten())  # Threshold everything above 0

        # plt.imsave('data/results/segm.jpg', segmented_image, cmap='gray')

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('Large Image')
        plt.imshow(image, cmap='gray')
        plt.subplot(122)
        plt.title('Prediction of large Image')
        plt.imshow(segmented_image, cmap='gray')
        plt.show()

    inverted_grid_mask = np.logical_not(segmented_image)

    image_without_grid = image * inverted_grid_mask  # + min_grid_img
    median_intensity = np.median(image_without_grid[(image_without_grid >= 1) & (image_without_grid <= 255)])
    # median_intensity = stats.mode(image_without_grid[(image_without_grid >= 1) & (image_without_grid <= 255)].flatten())
    #

    # Replace the grid regions with the calculated median intensity
    min_grid_img = segmented_image * (median_intensity + 20)
    image_without_grid = image_without_grid + min_grid_img

    image_without_grid = reduce_noise(image_without_grid, method="median")

    return image_without_grid

##########
# Load model and predict
# model = get_model()
# model.load_weights('22epochs_5_all')


def find_crystals_unet(image_directory, model):
    # large_image = cv2.imread('images/20230523-162432.550991.png', 0)
    # large_image = cv2.imread("test_images/TiO2am__0435.png", 0)
    large_image = cv2.imread(image_directory, 0)
    large_image = Image.fromarray(large_image)

    large_image = np.array(large_image.resize((512, 512)))  # TODO: hadle this properly

    img_without_grid = remove_grid(large_image, model, plot=False)

    # img_without_grid = inpaint_grid(model, large_image, 256)
    # img_without_grid = reduce_noise(img_without_grid, "median")


    crystals, result_img = find_crystals(img_without_grid, 25000, plot=True, img_return=True)
    return crystals, result_img


if __name__ == "__main__":
    model = models.load_model('22epochs_5_all')
    # model = models.load_model('standard_aug_20_50epochs.keras')

    test_images_dir = "test_images"
    processed_images_dir = "processed"

    # Create the processed folder if it doesn't exist
    if not os.path.exists(processed_images_dir):
        os.makedirs(processed_images_dir)


    image_files = os.listdir(test_images_dir)

    for img in image_files:
        image_path = os.path.join(test_images_dir, img)
        print(image_path)
        result_image_path = f"{os.path.splitext(img)[0]}_unet"
        i = 1
        while os.path.exists(os.path.join(processed_images_dir, f"{result_image_path}_standard_{i}.png")):
            i += 1
        result_image_path = os.path.join(processed_images_dir, f"{result_image_path}_standard_{i}.png")

        crystals, processed_image_data = find_crystals_unet(image_path, model)

        # Save the processed image

        processed_image = Image.fromarray(np.uint8(processed_image_data))
        processed_image.save(result_image_path)


# image_without_grid = large_image * inverted_grid_mask
# crystals = find_crystals(image_without_grid, 200, plot=True)


# idea to remove the grid -- take a look at histograms

# print(large_image.shape)
# img_without_grid = remove_grid(model, large_image, patch_size)
# crystals = find_crystals(img_without_grid, 25000, plot=True)