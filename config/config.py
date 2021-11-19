# Configurations
import warnings
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

# import pretty_errors  # NOQA: F401 (imported but unused)
# from rich.logging import RichHandler
import torch

########################################################### Repository's Names ###########################################################
AUTHOR = "Hongnan G."
REPO = "reighns_cassava"

########################################################### Torch Device ###########################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################### Creating Directories ###########################################################

# This will create all the folders needed. Which begs the question on when should I execute this script?
BASE_DIR = Path(__file__).parent.parent.absolute()  # C:\Users\reigHns\mnist

CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_DIR = Path(BASE_DIR, "model")
STORES_DIR = Path(BASE_DIR, "stores")


# Local stores
BLOB_STORE = Path(STORES_DIR, "blob")
FEATURE_STORE = Path(STORES_DIR, "feature")
MODEL_REGISTRY = Path(STORES_DIR, "model")
TENSORBOARD = Path(STORES_DIR, "tensorboard")

# Data folders
RAW_DATA_DIR = Path(DATA_DIR, "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")
TRAIN_DATA_DIR = Path(DATA_DIR, "train")
TEST_DATA_DIR = Path(DATA_DIR, "test")


# Create dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
FEATURE_STORE.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
# new folder
TENSORBOARD.mkdir(parents=True, exist_ok=True)


# MLFlow model registry: Note the filepath is file:////C:\Users\reigHns\mnist\stores\model

# mlflow.set_tracking_uri(uri="file://" + str(MODEL_REGISTRY.absolute()))
# workaround for windows at the moment
# TODO : Switch to Linux.
# mlflow.set_tracking_uri(uri="file://" + "C:/Users/reigHns/mnist/stores/model")

########################################################### Suppress User Warnings ###########################################################
warnings.filterwarnings("ignore", category=UserWarning)


def init_logger(log_file: str = "info.log"):
    """
    Initialize logger.
    """

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter("%(asctime)s - %(message)s"))
    file_handler = FileHandler(filename=log_file)
    file_handler.setFormatter(Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = init_logger()
