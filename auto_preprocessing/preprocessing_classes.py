from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import re
from typing import Any

import matplotlib
import numpy as np
from cv2 import cv2
from jinja2 import Template
from matplotlib import patches
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR

from ege_parser.utils import OcrResult


class BestPreprocessorFinder:
    def __init__(self):
        pass

    def find_best_preprocessor(
        self, preprocessors: list[Preprocessor], np_image: np.array, ocr_engine: PaddleOCR
    ) -> Preprocessor:
        processed_images = []

        # Gather Images from incoming scenarios
        for preprocessor in preprocessors:
            processed_image = preprocessor(np_image)

            # Optionally, perform an OCR task on each of compared images to find the best one
            if ocr_engine is not None:
                processed_image = OcrResult(
                    processed_image, ocr_engine.ocr(processed_image)
                ).processed_image

            processed_images.append(processed_image)

        # Define best image
        selector = ImageSelector(processed_images)
        best_image_index = selector.get_best_image_index()

        return preprocessors[best_image_index]


class ParameterTuner:
    def __init__(self):
        pass

    def tune(
        self, scenarios: list[PreprocessScenario], np_image: np.array, ocr_engine: PaddleOCR
    ) -> list[PreprocessScenario]:
        for scenario in scenarios:
            parameters, images = map(list, zip(*scenario.get_processed_images(np_image)))

            # Optionally, perform an OCR task on each of compared images to find the best one
            if ocr_engine is not None:
                for index, image in enumerate(images):
                    images[index] = OcrResult(image, ocr_engine.ocr(image)).processed_image

            # Define best image
            selector = ImageSelector(images)
            best_image_index = selector.get_best_image_index()
            scenario.best_params = parameters[best_image_index]

        return scenarios


class MemoryModule:
    def __init__(self):
        self.memory = {}
        self.best_jinja_template = None
        self.best_parameter_dict = None

    def load(self, filepath: str):
        """
        Loads the memory and best preprocessor attributes from a file.

        Args:
            filepath (str): Path to the file to load memory from.
            method (str): Method to use ('pickle' or 'json'). Defaults to 'pickle'.
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.memory = data.get("memory", {})
                self.best_jinja_template = data.get("best_jinja_template", None)
                self.best_parameter_dict = data.get("best_parameter_dict", None)

        except Exception as e:
            print(f"Error loading memory: {e}")

    def dump(self, filepath: str):
        """
        Dumps the memory and best preprocessor attributes to a file.

        Args:
            filepath (str): Path to the file to save memory to.
            method (str): Method to use ('pickle' or 'json'). Defaults to 'pickle'.
        """
        try:
            data = {
                "memory": self.memory,
                "best_jinja_template": self.best_jinja_template,
                "best_parameter_dict": self.best_parameter_dict,
            }

            with open(filepath, "w") as f:
                json.dump(data, f)

        except Exception as e:
            print(f"Error saving memory: {e}")

    def set_best_preprocessor(self, jinja_template: str, parameter_dict: dict):
        """
        Sets the best preprocessor attributes.

        Args:
            jinja_template (str): The best Jinja template as a string.
            parameter_dict (dict): The best parameter dictionary.
        """
        self.best_jinja_template = jinja_template
        self.best_parameter_dict = parameter_dict

    def get_best_preprocessor(self) -> tuple[str, dict]:
        """
        Returns the best preprocessor attributes.

        Returns:
            tuple: A tuple containing the best Jinja template and parameter dictionary.
        """
        return self.best_jinja_template, self.best_parameter_dict

    def filter(self, np_image: np.ndarray, scenarios: list) -> list:
        """
        Filters out known scenarios for a given image.

        Args:
            np_image (np.ndarray): The image being processed.
            scenarios (list): The list of scenarios to filter.

        Returns:
            list: A filtered list of scenarios.
        """
        image_hash = self._get_hashed_image(np_image)
        known_scenarios = self.memory.get(image_hash, set())

        new_scenarios = []
        for scenario in scenarios:
            scenario_hash = self._get_hashed_scenario(scenario)
            if scenario_hash not in known_scenarios:
                new_scenarios.append(scenario)
                self.memory.setdefault(image_hash, set()).add(scenario_hash)

        return new_scenarios

    @staticmethod
    def _get_hashed_image(image: np.ndarray) -> str:
        return image.tobytes().hex()

    @staticmethod
    def _get_hashed_scenario(scenario) -> str:
        scenario_str = scenario.template + str(sorted(scenario.parameters.items()))
        return hashlib.sha256(scenario_str.encode()).hexdigest()


class ImageSelector:
    def __init__(self, images: list[Any], batch_size: int = 4):
        self.images = images
        self.image_indexes = list(range(len(images)))
        self.current_batch_indexes: list[int] | None = None
        self.best_image_index: int | None = None
        self.batch_size = batch_size

    def get_best_image_index(self):
        # No competition required, if there is only one participant
        if len(self.images) <= 1:
            self.best_image_index = 0
            return self.best_image_index

        while len(self.image_indexes) > 0:
            self._set_figure_and_axs()
            self._show_next_batch()

        plt.close(self.fig)
        return self.best_image_index

    def _show_next_batch(self):
        # Get next batch of indexes
        if self.best_image_index is not None:
            batch_indexes = [self.best_image_index] + self.image_indexes[: self.batch_size - 1]
            self.image_indexes = self.image_indexes[self.batch_size - 1 :]
        else:
            batch_indexes = self.image_indexes[: self.batch_size]
            self.image_indexes = self.image_indexes[self.batch_size :]

        # Display images for the current batch
        for i, index in enumerate(batch_indexes):
            img = self.images[index]
            row, col = divmod(i, self.axs.shape[0])

            self.axs[row, col].set_title(f"Image {i + 1}")

            # Add black border around the image
            rect = patches.Rectangle(
                (0, 0),
                img.shape[1],
                img.shape[0],
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
            self.axs[row, col].add_patch(rect)

            self.axs[row, col].axis("off")

            self.axs[row, col].imshow(img)

        self.current_batch_indexes = batch_indexes

        # Bind the key press event
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show(block=True)  # Block to wait for user interaction

    def _on_key(self, event):
        if event.key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            selected_index = int(event.key) - 1

            assert (
                self.current_batch_indexes is not None
            ), "current_batch_indexes attribute is not initialized"
            self.best_image_index = self.current_batch_indexes[selected_index]
            plt.close()  # Close the plot after selection

        elif event.key.lower() == "c":
            plt.close()

    def _set_figure_and_axs(self):
        self.fig, self.axs = self._create_subplots(batch_size=self.batch_size)
        self.fig.text(
            0.5,
            0.01,
            "Enter the corresponding [1-9] key or press 'C' to exit",
            ha="center",
            fontsize=12,
        )
        plt.subplots_adjust(bottom=0.2, wspace=0.1, hspace=0.1)
        plt.tight_layout()

    @staticmethod
    def _create_subplots(batch_size):
        """
        Creates a 2x2 or 3x3 subplot grid based on batch size.

        Args:
            batch_size: The number of plots to create.

        Returns:
            A tuple of the figure and axes objects.
        """

        # Matplotlib backend
        matplotlib.use("TkAgg")

        if batch_size <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 3

        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        return fig, axs


class PreprocessingConductor:
    """
    This class only public method get_best_preprocessor() sets class instance best_preprocessor_parameters with use of passed
    np_image, scenarios list and ocr_engine. It also remembers all seen scenarios and runs a competition to find the best one
    only if unseen scenarios are coming.
    """

    def __init__(
        self,
        scenario_filter: ScenarioMemoryModule,
        tuner: ParameterTuner,
        finder: BestPreprocessorFinder,
    ):
        self.tuner = tuner
        self.finder = finder

    def get_best_preprocessor(
        self, scenarios: list[PreprocessScenario], np_image: np.array, ocr_engine: PaddleOCR
    ) -> Preprocessor:
        tuned_scenarios = self.tuner.tune(scenarios, np_image, ocr_engine)
        preprocessors = [x.best_preprocessor for x in tuned_scenarios]
        best_preprocessor = self.finder.find_best_preprocessor(preprocessors, np_image, ocr_engine)

        return best_preprocessor


class Preprocessor:
    """
    This class is executing given jinja_templated preprocessing pipeline using given parameters dictionary
    """

    def __init__(self, jinja_templated_str: str, param_dict: dict[str, str]):
        self.template = Template(jinja_templated_str)
        self.parameters = self._validate_param_dict(jinja_templated_str, param_dict)

    def __call__(self, np_image):
        rendered_code = self.template.render(self.parameters).splitlines()
        rendered_code = self._clean_rendered_code(rendered_code)
        preprocessed_image = self._run_preprocessing_pipeline(np_image, rendered_code)
        return preprocessed_image

    @staticmethod
    def _validate_param_dict(
        jinja_templated_str: str, param_dict: dict[str, str]
    ) -> dict[str, str]:
        # Checks that param_dict matches it's typing
        holded_dtypes = np.array(
            [type(variable) for option_list in param_dict.values() for variable in option_list]
        )
        if not (holded_dtypes == str).all():
            print(
                "Passed param_dict has wrong format.\n ",
                "Please check that all specified variables are strings!",
            )
            raise Exception

        # Checks that all {{ parameter }} presented in template are found in param_dict
        pattern = r"\{\{ ([^}]+) \}\}"
        matches = re.findall(pattern, jinja_templated_str)
        if not set(matches) - set(param_dict.keys()) == set():
            print(
                "Some parameters required by template are not presented into the parameter_dict:\n",
                f"{set(matches) - set(param_dict.keys())}",
            )
            raise Exception

        return param_dict

    @staticmethod
    def _clean_rendered_code(rendered_code: list[str]) -> list[str]:
        cleaned_list = [string.strip(" ") for string in rendered_code]
        cleaned_list = [string for string in cleaned_list if string != ""]
        return cleaned_list

    @staticmethod
    def _run_preprocessing_pipeline(np_image: np.array, rendered_code: list[str]) -> np.array:
        for line in rendered_code:
            preprocessing_layer = eval(line)
            np_image = preprocessing_layer(np_image)

        return np_image


class PreprocessScenario:
    def __init__(self, jinja_templated_str: str, param_dict: dict[str, list[str]]):
        self.template = jinja_templated_str
        self.parameters = param_dict
        self.parameter_combinations = self._generate_combinations()
        self._best_params: dict[str, str] = {}
        self._best_preprocessor: Preprocessor | None = None

    def get_processed_images(self, np_image: np.array):
        results = []
        for params in self.parameter_combinations:
            preprocessor = Preprocessor(self.template, params)
            preprocessed_image = preprocessor(np_image)

            results.append((params, preprocessed_image))

        return results

    @property
    def best_params(self):
        return self._best_params

    @best_params.setter
    def best_params(self, best_params: dict[str, str]):
        self._best_params = best_params
        self._best_preprocessor = Preprocessor(self.template, self._best_params)

    @best_params.getter
    def best_params(self):
        return self._best_params

    @property
    def best_preprocessor(self):
        return self._best_preprocessor

    def _generate_combinations(self):
        # Extract keys and values
        keys = self.parameters.keys()
        values = self.parameters.values()

        # Use itertools.product to generate all possible combinations
        combinations = itertools.product(*values)

        # Convert each combination into a dictionary
        return [dict(zip(keys, combination)) for combination in combinations]


class Crop:
    """
    Crop the image to the specified x and y ranges.

    :param x_crop: Tuple indicating the start and end positions for cropping along the x-axis.
    :param y_crop: Tuple indicating the start and end positions for cropping along the y-axis.
    """

    def __init__(
        self, x_crop: tuple[int | float, int | float], y_crop: tuple[int | float, int | float]
    ):
        self.x_crop = x_crop
        self.y_crop = y_crop

    def __call__(self, np_image: np.array) -> np.array:
        min_x = int(self.x_crop[0] * np_image.shape[0] if self.x_crop[0] < 1 else self.x_crop[0])
        max_x = int(self.x_crop[1] * np_image.shape[0] if self.x_crop[1] < 1 else self.x_crop[1])
        min_y = int(self.y_crop[0] * np_image.shape[1] if self.y_crop[0] < 1 else self.y_crop[0])
        max_y = int(self.y_crop[1] * np_image.shape[1] if self.y_crop[1] < 1 else self.y_crop[1])

        return np_image[min_y:max_y, min_x:max_x]


class Resize:
    """
    Resize the image by the given scale factor.

    :param scale_factor: The factor by which to scale the image.
    """

    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def __call__(self, np_image: np.array) -> np.array:
        new_dimensions = (
            int(np_image.shape[1] * self.scale_factor),
            int(np_image.shape[0] * self.scale_factor),
        )
        return cv2.resize(np_image, new_dimensions, interpolation=cv2.INTER_CUBIC)


class GaussianSmooth:
    """
    Apply Gaussian smoothing to the image.

    :param kernel_size: Size of the kernel used for Gaussian smoothing.
    """

    def __init__(self, kernel_size: tuple[int, int]):
        self.kernel_size = kernel_size

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.GaussianBlur(np_image, self.kernel_size, 0)


class ConvertToGray:
    """
    Convert the image to grayscale.

    No parameters required for this class.
    """

    def __init__(self):
        pass

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)


class BinaryThreshold:
    """
    Apply adaptive binary thresholding to the image.

    :param n_neighbors: The size of the neighborhood area used in thresholding.
    :param constant: A constant subtracted from the mean or weighted mean.
    """

    def __init__(self, n_neighbors: int, constant: int):
        self.n_neighbors = n_neighbors
        self.constant = constant

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.adaptiveThreshold(
            np_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.n_neighbors,
            self.constant,
        )


class Dilate:
    """
    Apply dilation to the image.

    :param kernel: The kernel used for the dilation operation.
    :param iterations: Number of times dilation is applied.
    """

    def __init__(self, kernel: np.array, iterations: int = 1):
        self.kernel = kernel
        self.iterations = iterations

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.dilate(np_image, self.kernel, iterations=self.iterations)


class Erode:
    """
    Apply erosion to the image.

    :param kernel: The kernel used for the erosion operation.
    :param iterations: Number of times erosion is applied.
    """

    def __init__(self, kernel: np.array, iterations: int = 1):
        self.kernel = kernel
        self.iterations = iterations

    def __call__(self, np_image: np.array) -> np.array:
        return cv2.erode(np_image, self.kernel, iterations=self.iterations)
