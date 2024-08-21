from __future__ import annotations

from copy import deepcopy

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from ege_parser.ocr_model import get_ocr_model
from ege_parser.preprocessor import get_preprocessor
from ege_parser.utils import (
    Config,
    DataLoader,
    ExtractGridEntries,
    ExtractImageGrid,
    OcrResult,
    ReconstructAndSupress,
)

GridExtractor = ExtractImageGrid(
    horiz_kernel_divider=10,
    vertic_kernel_divider=30,
    horiz_closing_iterations=3,
    vertical_closing_iterations=1,
)


class MyPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.load_data = DataLoader(config.SCAN_PATH)
        self.preprocessing = get_preprocessor()  # Need a preprocessor here

        self.grid_entries = ExtractGridEntries("inseparable", verbose=False)
        self.ocr_engine = get_ocr_model()
        self.reconstruction = ReconstructAndSupress(verbose=True, iou_threshold=0.05)

    def process(self):
        data = self.load_data.load_data()

        # Preprocess
        preprocessed = deepcopy(data)
        for filename, page_list in data.items():
            for page in page_list:
                preprocessed[filename][page] = self.preprocessing(data[filename][page])

        image_grid = np.array(Image.open("./data/external/ege_grid.jpg"))
        image_grid = self.preprocessing(image_grid)

        # Extract grid entries
        extracted = deepcopy(preprocessed)
        for filename, page_list in preprocessed.items():
            for page in page_list:
                extracted[filename][page] = self.grid_entries(
                    preprocessed[filename][page], image_grid
                )

        # Perform an OCR recognition task
        ocr = deepcopy(preprocessed)
        for filename, page_list in preprocessed.items():
            for page in page_list:
                page_image = preprocessed[filename][page]
                ocr[filename][page] = OcrResult(page_image, self.ocr_engine.ocr(page_image))

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
    mypipline = MyPipeline(myconfig)

    # Invoke initialized Pipline
    output_arrays = mypipline.process()

    print(output_arrays)


if __name__ == "__main__":
    main()
