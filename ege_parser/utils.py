from __future__ import annotations

import numpy as np
import glob
from pathlib import Path
import os
import cv2 as cv2
import fitz

class DataLoader:
    '''
    Performs data loading and pdf mining
    '''
    def __call__(self, scan_path):
        # Get file_paths
        file_paths = DataLoader.get_pdfs_and_images(scan_path)

        data = {}
        # Convert to images
        for file_path in file_paths:
            file_name = file_path.name
            file_format = file_path.suffix

            if file_format == '.pdf':
                imgs = DataLoader.mine_pdf(file_path)
            else:
                imgs = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            data[file_name] = imgs

        return data

    @staticmethod
    def get_pdfs_and_images(scan_path : Path) -> list[Path]:
        '''
        Extracts all pdf/jpg/png files from specifies folder
        :param scan_path: scan-folder path
        :return: all files that match specified extensions
        '''

        extensions = ['.pdf', '.png', '.jpg']
        try:
            file_paths = [file for file in scan_path.glob('*') if file.suffix in extensions]

        except:
            print(f'No avaliable images or pngs found with SCAN_PATH={scan_path}\n',
                   'Please verify the config.yaml file')

        return file_paths

    @staticmethod
    def mine_pdf(pdf_path: Path) -> list[np.array]:
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

class Preprocessing:
    '''
    This class includes all preprocessing operations avaliable
    '''
    @staticmethod
    def crop(np_image, x_crop : tuple[int, int], y_crop : tuple[int, int]) -> np.array:
        return np_image[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1], :]

    @staticmethod
    def resize(np_image : np.array, scale_factor : float) -> np.array:
        # Resizing
        return cv2.resize(np_image, (np_image.shape[1] * scale_factor, np_image.shape[0] * scale_factor),
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def gaussian_smooth(np_image : np.array, kernel_size: tuple[int, int]) -> np.array:
        return cv2.GaussianBlur(np_image, kernel_size)

    @staticmethod
    def convert_to_grayscale(np_image : np.array):
        return cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def binary_threshold(np_image : np.array, n_neigbours : int, constant : int) -> np.array:
        # Apply binary thresholding with an adaptive method
        return cv2.adaptiveThreshold(np_image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     n_neigbours,
                                     constant)

    @staticmethod
    def extract_image_grid(np_image: np.array,
                           horiz_kernel_divider : int,
                           vertic_kernel_divider : int,
                           vertical_closing_iterations : int,
                           horiz_closing_iterations : int) -> np.array:
        '''
        Given numpy image, removes all objects excluding horizontal and vertical lines

        :param np_image: binarized/grayscaled numpy image
        :param horiz_kernel_divider: greater the number -> smaller horizontal lines will be extracted
        :param vertic_kernel_divider: greater the number -> smaller vertical lines will be extracted
        :param vertical_closing_iterations: greater the number -> cleaner the result
        :param horiz_closing_iterations: greater the number -> cleaner the result
        :return: 
        '''

        inversed = cv2.bitwise_not(np_image)

        horizontal_lines_img = np.copy(inversed)
        verticle_lines_img = np.copy(inversed)

        # Размер ядра, чем меньше делитель, тем больше размер => только большие ровные полосы останутся на картинке
        horizontal_size = horizontal_lines_img.shape[1] // horiz_kernel_divider
        vertical_size = verticle_lines_img.shape[0] // vertic_kernel_divider

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

        # Число итераций, чем больше число итераций, тем более явно производится очистка
        # Morphological operation to detect verticle lines from an image
        verticle_lines_img = cv2.erode(verticle_lines_img, vertical_kernel, iterations= vertical_closing_iterations)
        verticle_lines_img = cv2.dilate(verticle_lines_img, vertical_kernel, iterations= vertical_closing_iterations)

        # Morphological operation to detect horizontal lines from an image
        horizontal_lines_img = cv2.erode(horizontal_lines_img, horizontal_kernel, iterations= horiz_closing_iterations)
        horizontal_lines_img = cv2.dilate(horizontal_lines_img, horizontal_kernel, iterations= horiz_closing_iterations)

        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final = cv2.add(verticle_lines_img, horizontal_lines_img)

        # Inverse image back
        img_final = cv2.bitwise_not(img_final)

        return img_final

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
    def extract_grid_entries(np_image: np.array,
                             np_image_boundaries: np.array,
                             mode: str = 'inseparable',
                             verbose: bool = False) -> list[np.array]:
        '''
        Extracts information within rectangle boundaries from given image using grid given by second argument
        :param np_image:
        :param mode: 'Mode' parameter stands for output list format:
                     'piecewise' - return extracted regions as independent shards of the original image, top-left to bot-right
                     'inseperable' - return extracted regions as the only non-blacked-out regions on the original image
        :param verbose:
        :return: np.array
        '''
        data = []

        # Loop through every image
        regions_of_interest = []

        # Find contours using the standard Canny edge detection and contour finding pipeline
        contours, hierarchy = cv2.findContours(np_image_boundaries, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = [x for x in hierarchy[0]]

        # Sort contours from top to bottom left to right
        contours, hierarchy = Preprocessing._sort_contours_and_hierarchy(contours, hierarchy)

        # create an empty image for contours
        img_contours = np.uint8(np.zeros((np_image.shape[0], np_image.shape[1])))

        # Check if contours are found
        # if contours:
        #     # Draw all contours with different parameters
        #     for i, contour in enumerate(contours):
        #         if len(contour) > 0:  # Ensure the contour is not empty
        #             color = (i * 20 % 255, 255 - i * 10 % 255, i * 5 % 255)  # Unique color for each contour
        #             cv2.drawContours(img_contours, contours, i, color, 2, cv2.LINE_AA)
        # else:
        #     print("No contours found.")
        #
        # cv2.imshow('a', img_contours)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # create blank list to hold ROI of original data
        blank_list = np.uint8(np.full((np_image.shape[0], np_image.shape[1]), 255))

        for cnt, hr in zip(contours, hierarchy):
            # I have no children, I have a parent and big enough
            if hr[2] == -1 and hr[-1] != -1 and cv2.contourArea(cnt) >= 400:
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

        if mode == 'piecewise':
            data.append(regions_of_interest)

        elif mode == 'inseparable':
            data.append(blank_list)

        if verbose:
            cv2.imshow('origin', np_image)  # Выводим оригинальное изображение
            cv2.imshow('used grid', np_image_boundaries)
            cv2.imshow('contours', img_contours)  # Выводим контуры
            cv2.imshow('intersection', cv2.bitwise_and(np_image, img_contours))  # Выводим наложение
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return data

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

