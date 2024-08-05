from __future__ import annotations

from copy import deepcopy

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
    OcrResult,
    ReconstructAndSupress,
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
        self.reconstruction = ReconstructAndSupress(verbose=True, iou_threshold=0.1)

    def process(self):
        data = self.load_data()

        # Preprocess
        preprocessed = deepcopy(data)
        for filename, page_list in data.items():
            for page in page_list:
                preprocessed[filename][page] = self.preprocessing(data[filename][page])

        image_boundaries = np.array(Image.open("./data/external/ege_grid.jpg"))
        image_boundaries = self.preprocessing(image_boundaries)

        # Extract grid entries
        extracted = deepcopy(preprocessed)
        for filename, page_list in preprocessed.items():
            for page in page_list:
                extracted[filename][page] = self.grid_entries(
                    preprocessed[filename][page], image_boundaries
                )

        # Perform an OCR recognition task
        ocr = deepcopy(extracted)
        for filename, page_list in extracted.items():
            for page in page_list:
                ocr[filename][page] = OcrResult(self.ocr_engine.ocr(extracted[filename][page]))

        # Expand boxes
        for filename, page_list in ocr.items():
            for page in page_list:
                self.reconstruction(
                    extracted[filename][page], ocr[filename][page]
                )  # mypy : ignore

        # Create the reports' directory, If it exists, delete it and recreate
        # SAVE_FOLDER = self.config.SCAN_PATH / self.config.SAVE_FOLDER
        # if SAVE_FOLDER.exists():
        #    shutil.rmtree(SAVE_FOLDER)


#
# SAVE_FOLDER.mkdir(parents=True, exist_ok=False)
#
# for filename, images_list in x.items():
#    # Assume every filename can have multiple images (like a multi-page PDF)
#    for page, image in enumerate(images_list):
#        result = self.ocr_engine.ocr(image)
#        image_boxes = VisualizeBoundingBoxes()(image, result)
#
#        dir_path = SAVE_FOLDER / filename / f"page_{page}"
#        dir_path.mkdir(parents=True, exist_ok=True)
#
#        image_boxes = Image.fromarray(image_boxes)
#        save_path = os.path.join(dir_path, "ocr_result.jpg")
#        image_boxes.save(save_path)


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
