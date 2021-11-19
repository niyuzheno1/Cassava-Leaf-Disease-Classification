import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from config import global_params

from src import dataset, make_folds, transformation, utils

FILES = global_params.FilePaths()
FOLDS = global_params.MakeFolds()
loader_params = global_params.DataLoaderParams()
train_params = global_params.GlobalTrainParams()


def return_filepath(image_id: str, folder: Path = FILES.train_images) -> str:
    """Add a new column image_path to the train and test csv so that we can call the images easily in __getitem__ in Dataset.

    Args:
        image_id (str): [description]
        folder (Path, optional): [description]. Defaults to FILES().train_images.

    Returns:
        str: [description]
    """
    path = os.path.join(folder, f"{image_id}.jpg")
    return path


def prepare_data() -> pd.DataFrame:
    """Call a sequential number of steps to prepare the data.

    Returns:
        pd.DataFrame: [description]
    """

    df_train = pd.read_csv(FILES.train_csv)
    df_test = pd.read_csv(FILES.test_csv)
    sub = pd.read_csv(FILES.sub_csv)

    df_train["image_path"] = df_train["Id"].apply(lambda x: return_filepath(image_id=x, folder=FILES.train_images))
    df_test["image_path"] = df_test["Id"].apply(lambda x: return_filepath(x, folder=FILES.test_images))

    df_folds = make_folds.make_folds(train_csv=df_train, config=FOLDS)
    return df_train, df_test, df_folds, sub


def prepare_loaders(
    df_folds: pd.DataFrame, fold: int
) -> Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare Data Loaders."""

    if train_params.debug:
        df_train = df_folds[df_folds["fold"] != fold].sample(loader_params.train_loader["batch_size"] * 32)
        df_valid = df_folds[df_folds["fold"] == fold].sample(loader_params.train_loader["batch_size"] * 32)
        df_oof = df_valid.copy()
    else:
        df_train = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
        df_valid = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
        # Initiate OOF dataframe for this fold (same as df_valid).
        df_oof = df_valid.copy()

    dataset_train = dataset.PawpularityDataset(df_train, transforms=transformation.get_train_transforms(), mode="train")
    dataset_valid = dataset.PawpularityDataset(df_valid, transforms=transformation.get_valid_transforms(), mode="train")

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, **loader_params.train_loader, worker_init_fn=utils.seed_worker, generator=g
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, **loader_params.valid_loader, worker_init_fn=utils.seed_worker, generator=g
    )

    return train_loader, valid_loader, df_oof
