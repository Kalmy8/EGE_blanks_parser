from __future__ import annotations

import os
import pickle
from typing import cast

from ege_parser.preprocessing_classes import PreprocessingConductor, Preprocessor

from .utils import Config

myconfig = Config()
PREPROCESSING_MODEL_PATH = myconfig.PREPROCESSING_MODEL_PATH


def load_cached_preprocessor() -> Preprocessor | None:
    if os.path.exists(PREPROCESSING_MODEL_PATH):
        with open(PREPROCESSING_MODEL_PATH, "rb") as f:
            preprocessor = cast(Preprocessor, pickle.load(f))
            return preprocessor
    return None


def save_preprocessor_to_cache(preprocessor: Preprocessor):
    with open(PREPROCESSING_MODEL_PATH, "wb") as f:
        pickle.dump(preprocessor, f)


def get_preprocessor() -> Preprocessor:
    # Try to load the cached preprocessor
    preprocessor = load_cached_preprocessor()
    if preprocessor is None:
        # If not found, run the preprocessing-picking routine
        try:
            from ege_parser.preprocessing_scenarios import SCENARIOS

        except Exception as exc:
            print(exc)
            print(
                "Could not import SCENARIOS from ege_parser.preprocessing_scenarios.py",
                "make sure that file exists and SCENARIOS list is not empty",
                sep="\n",
            )

        from .ocr_model import get_ocr_model
        from .utils import DataLoader

        data = DataLoader(myconfig.SCAN_PATH).load_data()
        test_image = list(data.values())[0]
        ocr_engine = get_ocr_model()
        preprocessor = PreprocessingConductor(SCENARIOS).get_best_preprocessor(
            test_image, ocr_engine
        )
        save_preprocessor_to_cache(preprocessor)

    return preprocessor


if __name__ == "__main__":
    pass
