from abc import ABC, abstractmethod

import numpy as np
from paddleocr import PaddleOCR

from ege_parser.utils.ocr_utils import OcrResult


class OcrEngine(ABC):
    @abstractmethod
    def process(self, np_image: np.ndarray) -> np.ndarray:
        pass


class PaddleOcrEngine(OcrEngine):
    def __init__(self):
        self.model = PaddleOCR(lang="en")

    def process(self, np_image: np.ndarray) -> np.ndarray:
        return self.model.ocr(np_image)


class PaddleOcrDrawBoxesOnImages(PaddleOcrEngine):
    def process(self, np_image: np.ndarray) -> np.ndarray:
        ocr_result = OcrResult(np_image, self.model.ocr(np_image))
        return ocr_result.processed_image


def retrieve_best_ocr_engine() -> OcrEngine:
    return PaddleOcrEngine()


# Some testing routines
if __name__ == "__main__":
    pass
