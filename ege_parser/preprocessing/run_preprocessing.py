from __future__ import annotations

from cv2 import cv2
from prepCV import PipelineManager, PipelineDescription, CacheManager, Preprocessor

from ege_parser.utils.dataloader import DataLoader
from ege_parser.utils.config import Config
from ege_parser.ocr_model.ocr_model import PaddleOcrDrawBoxesOnImages


def main():
    def crop_image(image, minx, maxx, miny, maxy):
        """Crops an image using relative coordinates (0-1)."""
        height, width = image.shape[:2]
        x_start = int(width * minx)
        x_end = int(width * maxx)
        y_start = int(height * miny)
        y_end = int(height * maxy)
        return image[y_start:y_end, x_start:x_end]

    optimal_crop_args = {"minx": [0.0],
                         "maxx": [0.95],
                         "miny": [0.25],
                         "maxy": [0.8]}

    def resize_image(img, scale_factor):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        # resize image
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    pipeline_manage = PipelineManager()
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

    myconfig = Config()
    myconfig.validate()
    dataloader = DataLoader(myconfig.SCAN_PATH)
    loaded_images = dataloader.load_data()
    test_image = loaded_images['someblanks.pdf']['page_0']
    pipeline_manage.add_pipeline(pipeline1)
    pipeline_manage.run_search(test_image, "GridSearch", ocr_engine = PaddleOcrDrawBoxesOnImages())
    print(pipeline_manage.best_preprocessor)

def retrieve_best_cached_preprocessor() -> Preprocessor:
    return CacheManager.load_best_preprocessor_from_cache()

if __name__ == '__main__':
    main()