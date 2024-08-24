from __future__ import annotations

import os
import pickle
from typing import Any, cast

import numpy as np
from auto_preprocessing.preprocessing_classes import (
    MemoryModule,
    PreprocessingConductor,
    Preprocessor,
    PreprocessScenario,
)
from auto_preprocessing.preprocessing_scenarios import SCENARIOS
from paddleocr import PaddleOCR

from ege_parser.ocr_model import get_ocr_model
from ege_parser.utils import DataLoader

PREPROCESSING_CACHE_PATH = "./preprocessor_cache"

def load_cached_parameters() -> MemoryModule | None:
    if os.path.exists(PREPROCESSING_CACHE_PATH):
        with open(PREPROCESSING_CACHE_PATH, 'rb') as f:
            memory_module = cast(MemoryModule, pickle.load(f))
            return memory_module

    return None

def save_parameters_to_cache(memory_module : MemoryModule):
    # Open a file for writing
    with open(PREPROCESSING_CACHE_PATH, 'w') as f:
        # Dump the dictionary to the file
        pickle.dump(memory_module, f)

def get_preprocessor(scenarios : list[PreprocessScenario], np_image : np.ndarray, ocr_engine: PaddleOCR = None) -> Preprocessor:
    # Try to load the cached parameters
    memory_module = load_cached_parameters()
    if memory_module is not None:

        # If there is cache, check if scenarios has changed at all
        unseen_scenarios = memory_module.filter(np_image, scenarios)

        # If no new scenarios were added, return an old Preprocessor
        if not unseen_scenarios:
            preprocessor = Preprocessor(param_dict["best_template"],
                                        param_dict["best_parameters"])
            return preprocessor

        else:


    preprocessor = pick_and_save_best_preprocessor()
    return preprocessor


def pick_and_save_best_preprocessor() -> Preprocessor:
    # Load test data
    data = DataLoader(myconfig.SCAN_PATH).load_data()
    test_image = data['someblanks.pdf']['page_0']

    # Load competing scenarios
    from auto_preprocessing.preprocessing_scenarios import SCENARIOS

    # Define ocr_engine
    ocr_engine = get_ocr_model()

    # Run competition
    conductor = PreprocessingConductor()
    preprocessor = conductor.get_best_preprocessor(
        SCENARIOS, test_image, ocr_engine
    )

    param_dict = {"best_template": preprocessor.template,
                  "best_parameters": preprocessor.parameters,
                  "seen_scenarios" : {??}}

    save_parameters_to_cache(param_dict)

    return preprocessor

# Picking and saving the best preprocessor
if __name__ == "__main__":
    pick_and_save_best_preprocessor()
