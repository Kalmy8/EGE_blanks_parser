from __future__ import annotations

from paddleocr import PaddleOCR, draw_ocr
from paddleocr import PPStructure, save_structure_res, draw_structure_result
import cv2 as cv2
import numpy as np
from src.utils import DataLoader
from src.utils import Preprocessing
import yaml
from PIL import Image
from pathlib import Path
import os
# Upload config file constants
with open('./configs/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

SCAN_PATH = Path(config['SCAN_PATH'])
SAVE_FOLDER = config['SAVE_FOLDER']
DET_MODEL_DIR = config['DET_MODEL_DIR']
REC_MODEL_DIR = config['REC_MODEL_DIR']
TABLE_MODEL_DIR = config['TABLE_MODEL_DIR']
REC_CHAR_DICT_PATH = config['REC_CHAR_DICT_PATH']
TABLE_CHAR_DICT_PATH = config['TABLE_CHAR_DICT_PATH']
FONT_PATH = config['FONT_PATH']

class EGE_processor(Preprocessing):
    def __call__(self, np_images : list | np.array) -> list | np.array:
        if isinstance(np_images, np.ndarray):
            return self._process_single_image(np_images)
        elif isinstance(np_images, list):
            return [self._process_single_image(np_image) for np_image in np_images]
        else:
            raise TypeError("Input must be a numpy array or a list of numpy arrays")

    def _process_single_image(self, np_image):
        np_image = Preprocessing.convert_to_grayscale(np_image)
        np_image = Preprocessing.binary_threshold(np_image, n_neigbours=11, constant=2)

        # image_boundaries = Preprocessing.extract_image_grid(np_image,
        #                                                     horiz_kernel_divider=10,
        #                                                     vertic_kernel_divider=30,
        #                                                     horiz_closing_iterations=3,
        #                                                     vertical_closing_iterations=1)

        image_boundaries = cv2.imread('./ege_grid.jpg')

        # Convert the image to grayscale
        image_boundaries = cv2.cvtColor(image_boundaries, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        ret, image_boundaries = cv2.threshold(image_boundaries, 127, 255, cv2.THRESH_BINARY)

        np_image = Preprocessing.extract_grid_entries(np_image, image_boundaries, verbose = True)

        return np_image


def main():

    # Load data
    data_loader = DataLoader()
    data = data_loader(SCAN_PATH)

    # Preprocess images
    engine = EGE_processor()
    data = dict((filename, engine(image)) for filename, image in data.items())

    # Initialize OCR engine
    table_engine = PPStructure(det_model_dir=DET_MODEL_DIR,
                               rec_model_dir=REC_MODEL_DIR,
                               table_model_dir=TABLE_MODEL_DIR,
                               rec_char_dict_path=REC_CHAR_DICT_PATH,
                               table_char_dict_path=TABLE_CHAR_DICT_PATH,
                               image_orientation=True,
                               layout=True, show_log=True)

    for img in imgs:
        result = table_engine(img)
        save_structure_res(result, SCAN_PATH.joinpath(SAVE_FOLDER), file_path.name)

        image = Image.fromarray(img).convert('RGB')
        im_show = draw_structure_result(image, result, font_path=FONT_PATH)
        im_show = Image.fromarray(im_show)
        im_show.show()

if __name__ == '__main__':
    main()
