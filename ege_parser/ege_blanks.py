from __future__ import annotations

import os
import shutil

import numpy as np
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from PIL import Image

from ege_parser.preprocessing import BinaryThreshold, ConvertToGrayscale
from ege_parser.utils import (
    Config,
    DataLoader,
    ExtractGridEntries,
    ExtractImageGrid,
    VisualizeBoundingBoxes,
)


class Preprocessor:
    """
    Handle EGE-blanks related preprocessing tasks
    """

    def __init__(self):
        self.to_gray = ConvertToGrayscale()
        self.binarize = BinaryThreshold(n_neighbours=21, constant=40)

    def __call__(self, np_image: np.array) -> np.array:
        np_image = self.to_gray(np_image)
        np_image = self.binarize(np_image)

        return np_image


GridExtractor = ExtractImageGrid(
    horiz_kernel_divider=10,
    vertic_kernel_divider=30,
    horiz_closing_iterations=3,
    vertical_closing_iterations=1,
)


class MyPipline:
    def __init__(self, config: Config):
        self.config = config
        self.load_data = DataLoader(config)
        self.preprocessing = Preprocessor()
        self.grid_entries = ExtractGridEntries("inseparable", verbose=False)
        self.ocr_engine = PaddleOCR(lang="en")
        # self.reconstruction = Reconstruct

    def process(self):
        x = self.load_data()

        x = {
            filename: [self.preprocessing(image) for image in listed_images]
            for filename, listed_images in x.items()
        }

        image_boundaries = np.array(Image.open("./data/external/ege_grid.jpg"))
        image_boundaries = self.preprocessing(image_boundaries)

        x = {
            filename: [self.grid_entries(image, image_boundaries) for image in listed_images]
            for filename, listed_images in x.items()
        }

        # x = self.reconstruction(x)

        # Create the reports' directory, If it exists, delete it and recreate
        SAVE_FOLDER = self.config.SCAN_PATH / self.config.SAVE_FOLDER
        if SAVE_FOLDER.exists():
            shutil.rmtree(SAVE_FOLDER)

        SAVE_FOLDER.mkdir(parents=True, exist_ok=False)

        for filename, images_list in x.items():
            # Assume every filename can have multiple images (like a multi-page PDF)
            for page, image in enumerate(images_list):
                result = self.ocr_engine.ocr(image)
                image_boxes = VisualizeBoundingBoxes()(image, result)

                dir_path = SAVE_FOLDER / filename / f"page_{page}"
                dir_path.mkdir(parents=True, exist_ok=True)

                image_boxes = Image.fromarray(image_boxes)
                save_path = os.path.join(dir_path, "ocr_result.jpg")
                image_boxes.save(save_path)


def main():
    # Loads required environmental variables
    load_dotenv()
    myconfig = Config()
    myconfig.validate()

    # Initialize constructed Pipline
    mypipline = MyPipline(myconfig)

    # Invoke initialized Pipline
    mypipline.process()


if __name__ == "__main__":
    main()
