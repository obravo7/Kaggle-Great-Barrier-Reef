from barrier_reef_data import FramePoints

from PIL import Image
import cv2
import numpy as np
import random


class Augment:
    @classmethod
    def total_blur(cls, image, filter_size=11):
        mb = Augment.med_blur(image, filter_size)
        total = Augment.bilateral_blur(mb)
        return total

    @classmethod
    def bilateral_blur(cls, image):
        blur = cv2.bilateralFilter(image, 9, 75, 75)
        return blur

    @classmethod
    def med_blur(cls, image, filter_size=5):
        blur = cv2.medianBlur(image, filter_size)
        return blur

    @classmethod
    def gauss_blur(cls, image, filter_size=15):
        """
         gaussian blur
        """
        blur = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
        return blur

    @classmethod
    def speckle_noise(cls, img):
        """
        add multiplicative speckle noise
        used for radar images
        """
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = img * (gauss / (len(gauss) - 0.50 * len(gauss)))

        return noisy

    @classmethod
    def salt_pepper_noise(cls, img, prob):
        """
        salt and pepper noise
        prob: probability of noise
        """
        output = np.zeros(img.shape, np.uint8)
        thresh = 1 - prob

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thresh:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output


def run_augment(func_list: list, img):

    for func in func_list:
        img = getattr(Augment, func)(img)
