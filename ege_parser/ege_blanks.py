from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2 as cv2
import numpy as np
from dotenv import load_dotenv
from paddleocr import PPStructure, draw_structure_result, save_structure_res
from PIL import Image

from ege_parser.utils import DataLoader, Preprocessing


# Function to check and load missing environment variables
def check_and_load_env_variables(required_env_variables):
    # Check for missing environment variables
    missing_args = [arg for arg in required_env_variables if os.getenv(arg) is None]

    if missing_args:
        # Print missing variables
        for arg in missing_args:
            print(f"Environment variable {arg} is not set")

        # Load environment variables from .env file
        print("Loading environmental variables from .env file...")
        load_dotenv()

        # Check again for missing variables after loading .env file
        missing_args_after_load = [arg for arg in required_env_variables if os.getenv(arg) is None]

        if missing_args_after_load:
            # Print missing variables after attempting to load .env file
            for arg in missing_args_after_load:
                print(f"Environment variable {arg} is missing in .env file")

            print("Please provide all required environmental variables and check the .env file")
            raise ValueError("Missing required environmental variables")


# Required variables list
required_env_variables = [
    "SCAN_PATH",
    "SAVE_FOLDER",
    "DET_MODEL_DIR",
    "REC_MODEL_DIR",
    "TABLE_MODEL_DIR",
    "REC_CHAR_DICT_PATH",
    "TABLE_CHAR_DICT_PATH",
    "FONT_PATH",
]


class EGE_processor(Preprocessing):
    """
    Child of Preprocessing class tuned to handle EGE-blanks related preprocessing tasks
    """

    def __call__(
        self, np_images: list[np.array], image_boundaries_reference: str | None = None
    ) -> list[np.array]:
        # Modify each entry of the original array
        return [
            self._process_single_image(np_image, image_boundaries_reference)
            for np_image in np_images
        ]

    def _process_single_image(
        self, np_image: np.array, image_template_path: str | None = None
    ) -> np.array:
        np_image = Preprocessing.convert_to_grayscale(np_image)
        np_image = Preprocessing.binary_threshold(np_image, n_neigbours=21, constant=40)

        # Extract image boundaries automatically if they are not provided
        if image_template_path is not None:
            try:
                image_boundaries = cv2.imread(image_template_path)

                # Preprocess template image
                image_boundaries = cv2.cvtColor(image_boundaries, cv2.COLOR_BGR2GRAY)
            except Exception as exc:
                print(f"Can not upload {image_template_path} image_boundaries template\n", exc)

        else:
            image_boundaries = Preprocessing.extract_image_grid(
                np_image,
                horiz_kernel_divider=10,
                vertic_kernel_divider=30,
                horiz_closing_iterations=3,
                vertical_closing_iterations=1,
            )

        # Binarize template image
        ret, image_boundaries = cv2.threshold(image_boundaries, 127, 255, cv2.THRESH_BINARY)

        # Extract grid entries from the provided image
        np_image = Preprocessing.extract_grid_entries(
            np_image, image_boundaries, mode="inseparable", verbose=False
        )

        return np_image


def main():
    check_and_load_env_variables(required_env_variables)

    SCAN_PATH = Path(str(os.getenv("SCAN_PATH")))
    SAVE_FOLDER = Path(str(os.getenv("SAVE_FOLDER")))
    DET_MODEL_DIR = Path(str(os.getenv("DET_MODEL_DIR")))
    REC_MODEL_DIR = Path(str(os.getenv("REC_MODEL_DIR")))
    TABLE_MODEL_DIR = Path(str(os.getenv("TABLE_MODEL_DIR")))
    REC_CHAR_DICT_PATH = Path(str(os.getenv("REC_CHAR_DICT_PATH")))
    TABLE_CHAR_DICT_PATH = Path(str(os.getenv("TABLE_CHAR_DICT_PATH")))
    FONT_PATH = Path(str(os.getenv("FONT_PATH")))

    # Load data
    data_loader = DataLoader()
    data = data_loader(SCAN_PATH)

    # Preprocess images
    engine = EGE_processor()
    data = dict((filename, engine(image)) for filename, image in data.items())
    # , './data/external/ege_grid.jpg'
    # Initialize OCR engine
    table_engine = PPStructure(
        det_model_dir=DET_MODEL_DIR.__str__(),
        rec_model_dir=REC_MODEL_DIR.__str__(),
        table_model_dir=TABLE_MODEL_DIR.__str__(),
        rec_char_dict_path=REC_CHAR_DICT_PATH.__str__(),
        table_char_dict_path=TABLE_CHAR_DICT_PATH.__str__(),
        image_orientation=True,
        layout=True,
        show_log=True,
    )

    # Create the reports directory if it doesn't exist
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    for filename, images_list in data.items():
        dir_path = SAVE_FOLDER / filename

        if dir_path.exists():
            # If it exists, delete it and recreate
            shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)

        # Assume every filename can have multiple images (like a multi-page PDF)
        for page, image in enumerate(images_list):
            result = table_engine(image)
            save_structure_res(result, SCAN_PATH.joinpath(SAVE_FOLDER), f"{filename}_page_{page}")

            image = Image.fromarray(image).convert("RGB")
            im_show = draw_structure_result(image, result, font_path=FONT_PATH)
            im_show = Image.fromarray(im_show)
            im_show.show()


if __name__ == "__main__":
    main()
