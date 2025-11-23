import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    _instance = None

    # Class variables to store configuration attributes
    SCAN_PATH: Path = None
    SAVE_FOLDER: Path = None
    PREP_CV_CACHE_FILEPATH: Path = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            # 1. Create the instance
            cls._instance = super(Config, cls).__new__(cls)

            # 2. Load environmental variables
            load_dotenv()

            # 3. Initialize attributes directly as class variables
            cls.SCAN_PATH = Path(kwargs.get("SCAN_PATH", os.getenv("SCAN_PATH")))
            cls.SAVE_FOLDER = Path(kwargs.get("SAVE_FOLDER", os.getenv("SAVE_FOLDER")))
            cls.PREP_CV_CACHE_FILEPATH = Path(
                kwargs.get("PREP_CV_CACHE_FILEPATH", os.getenv("PREP_CV_CACHE_FILEPATH"))
            )

            # 4. Validate config
            cls._instance.validate()

        return cls._instance

    def validate(self):
        required_vars = ["SCAN_PATH", "SAVE_FOLDER", "PREP_CV_CACHE_FILEPATH"]
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required configuration parameters: {', '.join(missing_vars)}"
            )
