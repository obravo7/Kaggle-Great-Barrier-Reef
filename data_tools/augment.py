from PIL import Image
import cv2
import numpy as np
import random

from typing import List


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
    def to_hsv(cls, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv_image

    @classmethod
    def increase_brightness(cls, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

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
    def salt_pepper_noise(cls, img, prob=0.50):
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


def run_augment(img, func_list: list = None) -> List[Image.Image]:
    if not func_list:
        func_list = [
            # 'salt_pepper_noise',
            # 'speckle_noise',
            'gauss_blur',
            'med_blur',
            'bilateral_blur',
            'total_blur',
            'to_hsv',
            'increase_brightness'
        ]
    img_list = []
    for func in func_list:
        img_a = getattr(Augment, func)(img.copy())
        img_a = Image.fromarray(img_a)
        img_list.append(img_a)
    return img_list
