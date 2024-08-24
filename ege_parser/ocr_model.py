from paddleocr import PaddleOCR


def get_ocr_model() -> PaddleOCR:
    model = PaddleOCR(lang="en")
    return model


# Some testing routines
if __name__ == "__main__":
    pass
