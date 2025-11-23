from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from prepCV import CacheManager, PipelineDescription, PipelineManager, Preprocessor

from ege_parser.ocr_model.make_ocr import PaddleOcrDrawBoxesOnImages
from ege_parser.utils.config import Config
from ege_parser.utils.dataloader import DataLoader


def retrieve_best_preprocessor(cache_filepath: str | Path) -> Optional[Preprocessor]:
    best_prep = CacheManager.load_best_preprocessor_from_cache(cache_filepath)
    if best_prep is None:
        print(
            "No best preprocessor found in Cache. \n",
            "Run 'make_preprocessor.py' module separately to define and save the best preprocessor.",
        )

    return best_prep


def main():
    myconfig = Config(PREP_CV_CACHE_FILEPATH=Path("_prepCV_cache.pkl"))

    def crop_image(image, minx, maxx, miny, maxy):
        """Crops an image using relative coordinates (0-1)."""
        height, width = image.shape[:2]
        x_start = int(width * minx)
        x_end = int(width * maxx)
        y_start = int(height * miny)
        y_end = int(height * maxy)
        return image[y_start:y_end, x_start:x_end]

    optimal_crop_args = {"minx": [0.0], "maxx": [0.95], "miny": [0.25], "maxy": [0.8]}

    def resize_image(img, scale_factor):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    pipeline_manage = PipelineManager()
    pipeline_manage.load_from_cache(myconfig.PREP_CV_CACHE_FILEPATH)
    pipeline1 = PipelineDescription(
        {
            cv2.cvtColor: {"code": [cv2.COLOR_BGR2GRAY]},
            crop_image: optimal_crop_args,
            resize_image: {
                "scale_factor": [1, 2],
            },
            cv2.adaptiveThreshold: {
                "maxValue": [255],
                "adaptiveMethod": [cv2.ADAPTIVE_THRESH_MEAN_C],
                "thresholdType": [cv2.THRESH_BINARY],
                "blockSize": [35, 51, 71],
                "C": [40, 60, 80],
            },
        }
    )

    dataloader = DataLoader(myconfig.SCAN_PATH)
    loaded_images = dataloader.load_data()
    test_image = loaded_images["someblanks.pdf"]["page_0"]
    pipeline_manage.add_pipeline(pipeline1)
    pipeline_manage.run_search(test_image, "GridSearch", ocr_engine=PaddleOcrDrawBoxesOnImages())
    pipeline_manage.save_to_cache(myconfig.PREP_CV_CACHE_FILEPATH)
    print(pipeline_manage.best_preprocessor)


if __name__ == "__main__":
    main()
