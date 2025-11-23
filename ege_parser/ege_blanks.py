from __future__ import annotations

from abc import ABC

import numpy as np
from PIL import Image
from prepCV import Preprocessor

from ege_parser.ocr_model.ocr_model import OcrEngine, retrieve_best_ocr_engine
from ege_parser.preprocessing.run_preprocessing import retrieve_best_cached_preprocessor
from ege_parser.utils.config import Config
from ege_parser.utils.dataloader import DataLoader
from ege_parser.utils.ocr_utils import (
    ExtractGridEntries,
    ExtractImageGrid,
    OcrResult,
    ReconstructAndSupress,
)


class OcrTableSkeleton(ABC):
    def __init__(
        self,
        preprocessor: Preprocessor,
        extractor: ExtractGridEntries,
        ocr_engine: OcrEngine,
        reconstructor: ReconstructAndSupress,
    ):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.ocr_engine = ocr_engine
        self.reconstructor = reconstructor

    def preprocess(self, np_image: np.ndarray) -> np.ndarray:
        return self.preprocessor.process(np_image)

    # Optional "hook"
    def extract_grid_entries(self, np_image):
        pass

    def apply_ocr(self, np_image: np.ndarray):
        return self.ocr_engine.process(np_image)

    def reconstruct_table(self, np_image: np.ndarray, ocr_result: OcrResult) -> np.ndarray:
        return self.reconstructor.process(np_image, ocr_result)

    def process(self, image: np.ndarray):
        # Preprocess data
        image = self.preprocess(image)

        # Extract grid entries:
        image = self.extract_grid_entries(image)

        # Perform OCR
        ocr_result = self.ocr_engine.process(image)
        ocr_result = OcrResult(image, ocr_result)

        # Reconstruct and supress
        return self.reconstruct_table(image, ocr_result)


class MyPipeline(OcrTableSkeleton):
    def extract_grid_entries(self, np_image):
        image_grid = np.array(Image.open("./utils/ege_grid.jpg"))
        return self.extractor.process(np_image, image_grid)


def main():
    myconfig = Config()
    myconfig.validate()
    GridExtractor = ExtractImageGrid(
        horiz_kernel_divider=10,
        vertic_kernel_divider=30,
        horiz_closing_iterations=3,
        vertical_closing_iterations=1,
    )
    reconstructor = ReconstructAndSupress(verbose=True, iou_threshold=0.05)
    extractor = ExtractGridEntries("inseparable", verbose=False)
    dataloader = DataLoader(myconfig.SCAN_PATH)
    preprocessor = retrieve_best_cached_preprocessor()
    ocr_engine = retrieve_best_ocr_engine()

    # Initialize constructed Pipline
    mypipline = MyPipeline(preprocessor, extractor, ocr_engine, reconstructor)
    data = dataloader.load_data()

    for filename, page_list in data.items():
        for page in page_list:
            mypipline.process(data[filename][page])


if __name__ == "__main__":
    main()
