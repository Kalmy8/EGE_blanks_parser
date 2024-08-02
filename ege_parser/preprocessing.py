import numpy as np
from cv2 import cv2


class Crop:
    def __init__(self, x_crop: tuple[int, int], y_crop: tuple[int, int]):
        self.x_crop = x_crop
        self.y_crop = y_crop

    def __call__(self, np_image: np.array) -> np.array:
        return np_image[self.x_crop[0] : self.x_crop[1], self.y_crop[0] : self.y_crop[1], :]


class Resize:
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.resize(
            np_image,
            (np_image.shape[1] * self.scale_factor, np_image.shape[0] * self.scale_factor),
            interpolation=cv2.INTER_CUBIC,
        )


class Gaussian_smooth:
    def __init__(self, kernel_size: tuple[int, int]):
        self.kernel_size = kernel_size

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.GaussianBlur(np_image, self.kernel_size, 0)


class ConvertToGrayscale:
    @staticmethod
    def __call__(np_image: np.array) -> np.array:
        return cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)


class BinaryThreshold:
    def __init__(self, n_neighbours: int, constant: int):
        self.n_neighbours = n_neighbours
        self.constant = constant

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.adaptiveThreshold(
            np_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.n_neighbours,
            self.constant,
        )
