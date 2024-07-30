from __future__ import annotations

from copy import deepcopy

import numpy as np
from dotenv import load_dotenv
from paddleocr import PaddleOCR

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

        # Perform an OCR recognition task
        ocr = deepcopy(preprocessed)
        for filename, page_list in preprocessed.items():
            for page in page_list:
                ocr[filename][page] = OcrResult(self.ocr_engine.ocr(preprocessed[filename][page]))

        # Expand boxes
        output_arrays = deepcopy(data)
        for filename, page_list in ocr.items():
            for page in page_list:
                output_arrays[filename][page] = self.reconstruction(
                    preprocessed[filename][page], ocr[filename][page]
                )  # mypy : ignore

        return output_arrays


def main():
    # Loads required environmental variables
    load_dotenv()
    myconfig = Config()
    myconfig.validate()

    # Initialize constructed Pipline
    mypipline = MyPipline(myconfig)

    # Invoke initialized Pipline
    output_arrays = mypipline.process()

    print(output_arrays)


if __name__ == "__main__":
    main()
