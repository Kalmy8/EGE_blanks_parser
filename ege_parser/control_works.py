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


# Разбивает исходную картинку по bounding box`ам
# Достает из картинки всю информацию
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
                    rec_char_dict_path=OCR_PATH + '/ppocr/utils/dict/cyrillic_dict.txt',
                    # сls_model_dir='{your_cls_model_dir}'
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
                               font_path=OCR_PATH + r'\doc\fonts\simfang.ttf')
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




if __name__ == '__main__':
    main()