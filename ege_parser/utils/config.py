from dotenv import load_dotenv

from pathlib import Path

import os
class Config:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, **kwargs)
        return cls._instance

    def __init__(self, **kwargs):
        # Loads required environmental variables
        load_dotenv()

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
