from dataclasses import dataclass, field

import pathlib
from typing import Any, Dict, List
from config import config


@dataclass
class FilePaths:
    """Class to keep track of the files."""

    train_images: pathlib.Path = pathlib.Path(config.DATA_DIR, "train")
    test_images: pathlib.Path = pathlib.Path(config.DATA_DIR, "test")
    train_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "raw/train.csv")
    test_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "raw/test.csv")
    sub_csv: pathlib.Path = pathlib.Path(
        config.DATA_DIR, "raw/sample_submission.csv"
    )
    folds_csv: pathlib.Path = pathlib.Path(
        config.DATA_DIR, "processed/train.csv"
    )
    weight_path: pathlib.Path = pathlib.Path(config.MODEL_DIR)
    oof_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "processed")


@dataclass
class DataLoaderParams:
    """Class to keep track of the data loader parameters."""

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    test_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )


@dataclass
class MakeFolds:
    """A class to keep track of cross-validation schema.

    seed (int): random seed for reproducibility.
    num_folds (int): number of folds.
    cv_schema (str): cross-validation schema.
    class_col_name (str): name of the target column.
    image_col_name (str): name of the image column.
    folds_csv (str): path to the folds csv.
    """

    seed: int = 1992
    num_folds: int = 5
    cv_schema: str = "StratifiedKFold"
    class_col_name: str = "label"
    image_col_name: str = "image_id"
    folds_csv: pathlib.Path = pathlib.Path(
        config.DATA_DIR, "processed/train.csv"
    )


@dataclass
class AugmentationParams:
    """Class to keep track of the augmentation parameters."""

    mean: List[float]
    std: List[float]
    image_size: int = 224


@dataclass
class ModelParams:
    """A class to track model parameters.

    model_name (str): name of the model.
    pretrained (bool): If True, use pretrained model.
    input_channels (int): RGB image - 3 channels or Grayscale 1 channel
    output_dimension: Final output neuron.
                      It is the number of classes in classification.
                      Caution: If you use sigmoid layer for Binary, then it is 1.
    """

    # model_name: str = "efficientnet_b0"
    # model_name: str = "vit_small_patch16_224"
    # model_name: str = "resnet34d"
    model_name: str = "tf_efficientnet_b4_ns"
    pretrained: bool = True
    input_channels: int = 3
    output_dimension: int = 5


@dataclass
class OptimizerParams:
    """A class to track optimizer parameters.

    optimizer_name (str): name of the optimizer.
    lr (float): learning rate.
    weight_decay (float): weight decay.
    """

    optimizer_name: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {"lr": 1e-4, "weight_decay": 1e-6}
    )


@dataclass
class SchedulerParams:
    """A class to track Scheduler Params."""

    scheduler_name: str = "CosineAnnealingWarmRestarts"
    scheduler_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "T_0": 10,
            "T_mult": 1,
            "eta_min": 1e-6,
            "last_epoch": -1,
        }
    )


@dataclass
class GlobalTrainParams:
    debug: bool = True
    epochs: int = 3
    mixup: bool = False
    model_name: str = ModelParams().model_name

    num_classes: int = ModelParams().output_dimension
