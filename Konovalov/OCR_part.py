"""
This module gets the last screenshot in /screenshots directory and extracts info
"""

from __future__ import annotations

import datetime
import glob
import os
from datetime import datetime
from paddleocr import PaddleOCR, draw_ocr
import paddleocr
import cv2 as cv2
import fitz
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

# your path here
global SCAN_PATH
SCAN_PATH = r'C:\Users\kalmy\OneDrive\Рабочий стол\Scans_folder'

OCR_PATH = os.path.dirname(paddleocr.__file__)


# Добавляет в словарь информацию о текущей дате
def append_actual_time(base_dict: dict) -> dict:
    base_dict['Time'] = datetime.now().replace(second=0, microsecond=0)
    return base_dict


# Находит в указанной директории последний из скриншотов
def get_latest_scan() -> np.array:
    list_of_files = glob.glob(SCAN_PATH + '\*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


class PdfParser:
    @staticmethod
    def _visualize_bounding_boxes(image: np.array, reader_data):
        """
        This function visualizes the bounding boxes of detected text in an image.

        Args:
            image: The original image.
            reader_data: The output of reader.readtext() function, a list of dictionaries containing text and bounding box information.
        """
        image = image.copy()
        # Loop through each detected text
        for data in reader_data:
            # Extract bounding box coordinates
            top_left, _, bottom_right, _ = data[0]
            top_left, bottom_right = list(map(int, top_left)), list(map(int, bottom_right))
            # Draw rectangle on the image
            cv2.rectangle(image, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0, 255, 0), 1)

            # Optionally, add text label under the bounding box
            text = data[1]
            cv2.putText(image, text, (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the image with bounding boxes
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _sort_contours_and_hierarchy(contours: list | tuple, hierarchy: list) -> list:
        x_coords = []
        y_coords = []
        for cnt in contours:
            start_x, start_y, _, _ = cv2.boundingRect(cnt)
            x_coords.append(start_x)
            y_coords.append(start_y // 15)
        return list(
            zip(*sorted(zip(contours, hierarchy, x_coords, y_coords), key=lambda t: (t[3], t[2]), reverse=False)))[:2]

    @staticmethod
    def extract_cells_boundaries(np_image: np.array) -> np.array:
        """
        Np_image is assumed to be single channel (grayscale) binarized image
        """
        inversed = cv2.bitwise_not(np_image)

        horizontal_lines_img = np.copy(inversed)
        verticle_lines_img = np.copy(inversed)

        # Размер ядра, чем меньше делитель, тем больше размер => только большие ровные полосы останутся на картинке
        horizontal_size = horizontal_lines_img.shape[1] // 40
        vertical_size = verticle_lines_img.shape[0] // 55

        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        # Число итераций, чем больше число итераций, тем более явно производится очистка
        # Morphological operation to detect verticle lines from an image
        verticle_lines_img = cv2.erode(verticle_lines_img, verticle_kernel, iterations=9)
        verticle_lines_img = cv2.dilate(verticle_lines_img, verticle_kernel, iterations=9)

        # Morphological operation to detect horizontal lines from an image
        horizontal_lines_img = cv2.erode(horizontal_lines_img, hori_kernel, iterations=4)
        horizontal_lines_img = cv2.dilate(horizontal_lines_img, hori_kernel, iterations=4)

        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final = cv2.add(verticle_lines_img, horizontal_lines_img)

        # Inverse image back and perform some smoothing optionally
        img_final = cv2.bitwise_not(img_final)

        return img_final

    @staticmethod
    def extract_cells_entries(image_list: list[np.array], mode: str = 'piecewise', verbose: bool = False) -> list[
        np.array]:
        """
            Extracts information within rectangle boundaries within given image
            'Mode' parameter stands for output list format:
                'piecewise' - return extracted regions as independent shards of the original image, top-left to bot-right
                'inseperable' - return extracted regions as the only non-blacked-out regions on the original image
        """
        data = []

        # Loop through every image
        for np_image in image_list:
            regions_of_interest = []

            # Detect grid-like structure
            np_image_boundaries = PdfParser.extract_cells_boundaries(np_image)

            # Find contours using the standard Canny edge detection and contour finding pipeline
            contours, hierarchy = cv2.findContours(np_image_boundaries, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = [x for x in hierarchy[0]]

            # Sort contours from top to bottom left to right
            contours, hierarchy = PdfParser._sort_contours_and_hierarchy(contours, hierarchy)

            # create an empty image for contours
            img_contours = np.uint8(np.zeros((np_image.shape[0], np_image.shape[1])))

            # create blank list to hold ROI of original data
            blank_list = np.uint8(np.full((np_image.shape[0], np_image.shape[1]), 255))

            for cnt, hr in zip(contours, hierarchy):
                # Рисуем только внутренние контуры, которые хорошо аппроксимируются прямоугольником
                # if hr[-1] != -1:
                #     perimeter = cv2.arcLength(cnt, True)
                #     approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                #     if len(approx) >= 4:

                # I have no children and I am internal Рисуем прямоуголььники вокруг самых младших элементов в иерархии
                if hr[2] == -1 and hr[-1] != -1:
                    if cv2.contourArea(cnt) >= 450:
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Margin for displaying only boxes content without box edges
                        margin = 2
                        if verbose:
                            cv2.rectangle(img_contours, (x + margin, y + margin), (x + w - margin, y + h - margin),
                                          (255, 255, 255), -1)

                        roi = np_image[y + margin:y + h - margin, x + margin:x + w - margin]

                        # Fill blank List with areas of original image
                        blank_list[y + margin:y + h - margin, x + margin:x + w - margin] = roi

                        regions_of_interest.append(roi)

            # Check if where are exactly 20 rectangular fields found
            if len(regions_of_interest) != 20:
                print('There are Corrupted Images. Try using another parameters for extracting function!')
                continue

            if mode == 'piecewise':
                data.append(regions_of_interest)

            elif mode == 'inseparable':
                data.append(blank_list)

            if verbose:
                cv2.imshow('origin', np_image)  # Выводим оригинальное изображение
                cv2.imshow('res', img_contours)  # Выводим контуры
                cv2.imshow('', cv2.bitwise_and(np_image, img_contours))  # Выводим наложение
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return data

    @staticmethod
    def extract_pngs(pdf_path: str, verbose: bool = False) -> list[np.array]:
        """
        Extracts and pre-process pages of the given p

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

            # Reshape the array based on pixmap dimensions, cut files heading to only OCR student`s names and marks
            np_image = np_image.reshape(pixmap.height, pixmap.width, pixmap.n)[:pixmap.height // 6, :, :]

            # Convert the image to grayscale
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

            # Resizing
            np_image = cv2.resize(np_image, (np_image.shape[1] * 2, np_image.shape[0] * 2),
                                  interpolation=cv2.INTER_CUBIC)

            # Gaussian Smoothing
            np_image = cv2.GaussianBlur(np_image, (3, 3), 0)

            # Apply binary thresholding with an adaptive method
            n_neighbors = 11
            constant = 2
            np_image = cv2.adaptiveThreshold(np_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             n_neighbors,
                                             constant)

            # Some dilation followed by erosion
            # kernel = np.ones((3,3),np.uint8)
            # np_image = cv2.morphologyEx(np_image, cv2.MORPH_CLOSE, kernel)

            extracted_images.append(np_image)

        # Display for pre-processing visualization
        if verbose:
            for image in extracted_images:
                cv2.imshow("Sharpened Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return extracted_images

    # Разбивает исходную картинку по bounding box`ам
    # Достает из картинки всю информацию
    @staticmethod
    def process_images(np_images: list[np.array] | list[list[np.array]], verbose: bool = False):
        """
        Process every image stored in the input array
        """
        # If images list contains only inseparable np.arrays
        if isinstance(np_images[0], np.ndarray):
            pass

        # If images list contains shredded np.arrays
        if isinstance(np_images[0], list):
            np_images = np_images[0]

        ocr = PaddleOCR(det_model_dir=OCR_PATH + '/multilang_det',
                        rec_model_dir=OCR_PATH + '/cyrillic_rec',
                        rec_char_dict_path= OCR_PATH + '/ppocr/utils/dict/cyrillic_dict.txt',
                        #сls_model_dir='{your_cls_model_dir}'
                       )

        image_data = []
        for piece in np_images:

            result = ocr.ocr(piece, det=True, rec=True, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(line)

            if verbose:
                # PdfParser._visualize_bounding_boxes(piece, piece_data)
                # draw result
                result = result[0]
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                piece = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
                im_show = draw_ocr(piece, boxes, txts=txts, scores=scores,
                                   font_path= OCR_PATH + r'\doc\fonts\simfang.ttf')
                cv2.imshow('result', im_show)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # image_data.append(piece_data)

        return image_data




def main():
    # Gets last scan path
    pdf_path = get_latest_scan()

    # Extract .png images from latest pdf_file in scans folder
    extracted_pngs = PdfParser.extract_pngs(pdf_path)

    # Extract only grid entries
    images_shredded = PdfParser.extract_cells_entries(extracted_pngs, mode='piecewise', verbose=False )
    image_data = PdfParser.process_images(images_shredded, verbose=True)

    pass


if __name__ == '__main__':
    main()