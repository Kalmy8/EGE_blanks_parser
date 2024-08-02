from __future__ import annotations

import os
from pathlib import Path

import cv2 as cv2
import fitz
import numpy as np


class Config:
    def __init__(self, **kwargs):
        self.SCAN_PATH = Path(kwargs.get("SCAN_PATH", os.getenv("SCAN_PATH")))
        self.SAVE_FOLDER = Path(kwargs.get("SAVE_FOLDER", os.getenv("SAVE_FOLDER")))
        self.DET_MODEL_DIR = Path(kwargs.get("DET_MODEL_DIR", os.getenv("DET_MODEL_DIR")))
        self.REC_MODEL_DIR = Path(kwargs.get("REC_MODEL_DIR", os.getenv("REC_MODEL_DIR")))
        self.TABLE_MODEL_DIR = Path(kwargs.get("TABLE_MODEL_DIR", os.getenv("TABLE_MODEL_DIR")))
        self.REC_CHAR_DICT_PATH = Path(
            kwargs.get("REC_CHAR_DICT_PATH", os.getenv("REC_CHAR_DICT_PATH"))
        )
        self.TABLE_CHAR_DICT_PATH = Path(
            kwargs.get("TABLE_CHAR_DICT_PATH", os.getenv("TABLE_CHAR_DICT_PATH"))
        )
        self.FONT_PATH = Path(kwargs.get("FONT_PATH", os.getenv("FONT_PATH")))

        self.validate()

    def validate(self):
        required_vars = [
            "SCAN_PATH",
            "SAVE_FOLDER",
            "DET_MODEL_DIR",
            "REC_MODEL_DIR",
            "TABLE_MODEL_DIR",
            "REC_CHAR_DICT_PATH",
            "TABLE_CHAR_DICT_PATH",
            "FONT_PATH",
        ]
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required configuration parameters: {', '.join(missing_vars)}"
            )


# class Reconstruct():
#     def __call__(self):
#         horiz_boxes = []
#         vert_boxes = []
#         for box in boxes:
#             x_h, x_v = 0, int(box[0][0])
#             y_h, y_v = int(box[0][1]), 0
#             width_h, width_v = img_width, int(box[2][0] - box[0][0])
#             height_h, height_v = int(box[2][1] - box[0][1]), img_height,
#
#             horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])
#             vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])
#
#             if verbose:
#                 img =
#             cv2.rectangle(img, (x_h, y_h), (x_h + width_h, y_h + height_h), (255, 0, 0), 1)
#             cv2.rectangle(img, (x_v, y_v), (x_v + width_v, y_v + height_v), (0, 255, 0), 1)


class DataLoader:
    """
    Examines specified folder, extracts all of the .pdf, .jpg, .png files as np.arrays,
    returns a python dictionary in a {'filename_str':[extracted_image1 , extracted_image2]} format
    """

    def __init__(self, config: Config):
        self.scan_path = config.SCAN_PATH

    def __call__(self) -> dict[str, list[np.array]]:
        # Get file_paths
        file_paths = self._get_pdfs_and_images(self.scan_path)
        data = {}

        # Convert to images
        for file_path in file_paths:
            file_name = file_path.name
            file_format = file_path.suffix

            if file_format == ".pdf":
                imgs = self._mine_pdf(file_path)
            else:
                imgs = [cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)]

            data[file_name] = imgs

        return data

    @staticmethod
    def _get_pdfs_and_images(scan_path: Path) -> list[Path]:
        """
        Extracts filepaths of all pdf/jpg/png files from specified folder
        :param scan_path: files folder path
        :return: filepaths of files that match specified extensions
        """

        extensions = [".pdf", ".png", ".jpg"]
        try:
            file_paths = [file for file in scan_path.glob("*") if file.suffix in extensions]

        except Exception:
            print(f"No avaliable images or pngs found with SCAN_PATH={scan_path}\n")

        return file_paths

    @staticmethod
    def _mine_pdf(pdf_path: Path) -> list[np.array]:
        """
        Extracts pages of given pdf-file
        """
        # Open the PDF file
        doc = fitz.open(pdf_path)

        extracted_images = []
        # Loop through each page
        for page in doc:
            # Extract images using list comprehension
            pixmap = page.get_pixmap()

            # Convert pixmap data to NumPy array
            np_image = np.frombuffer(pixmap.samples, dtype=np.uint8)

            # Reshape the array based on pixmap dimensions
            np_image = np_image.reshape(pixmap.height, pixmap.width, pixmap.n)

            extracted_images.append(np_image)

        return extracted_images


class ExtractImageGrid:
    def __init__(
        self,
        horiz_kernel_divider: int,
        vertic_kernel_divider: int,
        vertical_closing_iterations: int,
        horiz_closing_iterations: int,
    ):
        self.horiz_kernel_divider = horiz_kernel_divider
        self.vertic_kernel_divider = vertic_kernel_divider
        self.vertical_closing_iterations = vertical_closing_iterations
        self.horiz_closing_iterations = horiz_closing_iterations

    def __call__(self, np_image: np.array) -> np.array:
        inversed = cv2.bitwise_not(np_image)

        horizontal_lines_img = np.copy(inversed)
        verticle_lines_img = np.copy(inversed)

        horizontal_size = horizontal_lines_img.shape[1] // self.horiz_kernel_divider
        vertical_size = verticle_lines_img.shape[0] // self.vertic_kernel_divider

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        verticle_lines_img = cv2.erode(
            verticle_lines_img, vertical_kernel, iterations=self.vertical_closing_iterations
        )
        verticle_lines_img = cv2.dilate(
            verticle_lines_img, vertical_kernel, iterations=self.vertical_closing_iterations
        )

        horizontal_lines_img = cv2.erode(
            horizontal_lines_img, horizontal_kernel, iterations=self.horiz_closing_iterations
        )
        horizontal_lines_img = cv2.dilate(
            horizontal_lines_img, horizontal_kernel, iterations=self.horiz_closing_iterations
        )

        img_final = cv2.add(verticle_lines_img, horizontal_lines_img)

        img_final = cv2.bitwise_not(img_final)

        return img_final


class ExtractGridEntries:
    def __init__(self, mode: str, verbose: bool = False):
        self.mode = mode
        self.verbose = verbose

    @staticmethod
    def _sort_contours_and_hierarchy(
        contours: list[np.array], hierarchy: list[np.array]
    ) -> list[tuple[np.array]]:
        x_coords = []
        y_coords = []
        for cnt in contours:
            start_x, start_y, _, _ = cv2.boundingRect(cnt)
            x_coords.append(start_x)
            y_coords.append(start_y // 15)
        return list(
            zip(
                *sorted(
                    zip(contours, hierarchy, x_coords, y_coords),
                    key=lambda t: (t[3], t[2]),
                    reverse=False,
                )
            )
        )[:2]

    def __call__(
        self, np_image: np.array, image_boundaries: np.array
    ) -> list[np.array] | np.array:
        contours, hierarchy = cv2.findContours(
            image_boundaries, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = [x for x in hierarchy[0]]

        contours, hierarchy = self._sort_contours_and_hierarchy(contours, hierarchy)

        img_contours = np.uint8(np.zeros((np_image.shape[0], np_image.shape[1])))
        blank_list = np.uint8(np.full((np_image.shape[0], np_image.shape[1]), 255))

        regions_of_interest = []
        for cnt, hr in zip(contours, hierarchy):
            if hr[2] == -1 and hr[-1] != -1 and cv2.contourArea(cnt) >= 400:
                x, y, w, h = cv2.boundingRect(cnt)
                margin = 2
                if self.verbose:
                    cv2.rectangle(
                        img_contours,
                        (x + margin, y + margin),
                        (x + w - margin, y + h - margin),
                        (255, 255, 255),
                        -1,
                    )

                roi = np_image[y + margin : y + h - margin, x + margin : x + w - margin]
                blank_list[y + margin : y + h - margin, x + margin : x + w - margin] = roi
                regions_of_interest.append(roi)

        if self.mode == "piecewise":
            output = regions_of_interest

        elif self.mode == "inseparable":
            output = blank_list

        if self.verbose:
            cv2.imshow("origin", np_image)
            cv2.imshow("used grid", image_boundaries)
            cv2.imshow("contours", img_contours)
            cv2.imshow("intersection", cv2.bitwise_and(np_image, img_contours))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return output


class VisualizeBoundingBoxes:
    @staticmethod
    def __call__(image: np.array, result) -> np.array:
        # Extract the bounding boxes, text, and confidence scores
        boxes = [res[0] for res in result[0]]
        txts = [res[1][0] for res in result[0]]
        # scores = [res[1][1] for res in result[0]]

        image_boxes = image.copy()
        image_boxes = cv2.cvtColor(image_boxes, cv2.COLOR_GRAY2RGB)
        for box, txt in zip(boxes, txts):
            # Draw detected boxes
            cv2.rectangle(
                image_boxes,
                (int(box[0][0]), int(box[0][1])),
                (int(box[2][0]), int(box[2][1])),
                color=(0, 0, 255),
                thickness=1,
            )

            # Put the text near the bounding box
            cv2.putText(
                image_boxes,
                txt,
                (int(box[0][0]), int(box[0][1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=1,
            )

        return image_boxes
