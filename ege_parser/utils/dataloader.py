from pathlib import Path
from typing import Any

import numpy as np
from cv2 import cv2


class DataLoader:
    """
    Examines specified folder, extracts all of the .pdf, .jpg, .png files as np.arrays,
    returns a python dictionary in a {'filename_str':[extracted_image1 , extracted_image2]} format
    """

    def __init__(self, SCAN_PATH: Path):
        self.scan_path = SCAN_PATH

    def load_data(self) -> dict[str, dict[str, Any]]:
        # Get file_paths
        file_paths = self.get_pdfs_and_images()
        data = {}

        # Convert to images
        for file_path in file_paths:
            file_name = file_path.name
            file_format = file_path.suffix

            if file_format == ".pdf":
                imgs = self._mine_pdf(file_path)
            else:
                imgs = [cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)]

            data[file_name] = {f"page_{i}": image for i, image in enumerate(imgs)}

        return data

    def get_pdfs_and_images(self) -> list[Path]:
        """
        Extracts filepaths of all pdf/jpg/png files from specified folder
        :param scan_path: files folder path
        :return: filepaths of files that match specified extensions
        """

        extensions = [".pdf", ".png", ".jpg"]

        try:
            file_paths = [file for file in self.scan_path.glob("*") if file.suffix in extensions]
            return file_paths

        except FileNotFoundError:
            print(f"No avaliable images or pngs found with SCAN_PATH={self.scan_path}\n")
            return []

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