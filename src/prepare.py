import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from config import global_params

from src import dataset, make_folds, transformation, utils

FILES = global_params.FilePaths()
FOLDS = global_params.MakeFolds()
LOADER_PARAMS = global_params.DataLoaderParams()
TRAIN_PARAMS = global_params.GlobalTrainParams()


def return_filepath(image_id: str, folder: Path = FILES.train_images) -> str:
    """Add a new column image_path to the train and test csv.
    We can call the images easily in __getitem__ in Dataset.

    We need to be careful if the image_id has extension already.
    In this case, there is no need to add the extension.

    Args:
        image_id (str): The unique image id: 1000015157.jpg
        folder (Path, optional): The train folder. Defaults to FILES().train_images.

    Returns:
        image_path (str): The path to the image: "c:\\users\\reighns\\kaggle_projects\\cassava\\data\\train\\1000015157.jpg"
    """
    image_path = os.path.join(folder, f"{image_id}")
    return image_path


def prepare_data(
    image_col_name: str = FOLDS.image_col_name,
) -> pd.DataFrame:
    """Call a sequential number of steps to prepare the data.


    Args:
        image_col_name (str): The column name of the unique image id.
                        In Cassava, it is "image_id".

    Returns:
        df_train, df_test, df_folds, sub (pd.DataFrame): Returns the dataframes for train, test, folds and submission.
    """

    df_train = pd.read_csv(FILES.train_csv)
    df_test = pd.read_csv(FILES.test_csv)
    sub = pd.read_csv(FILES.sub_csv)

    df_train["image_path"] = df_train[image_col_name].apply(
        lambda x: return_filepath(image_id=x, folder=FILES.train_images)
    )
    df_test["image_path"] = df_test[image_col_name].apply(
        lambda x: return_filepath(x, folder=FILES.test_images)
    )

    df_folds = make_folds.make_folds(train_csv=df_train, config=FOLDS)
    return df_train, df_test, df_folds, sub


def prepare_loaders(
    df_folds: pd.DataFrame, fold: int
) -> Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare Data Loaders."""

    if TRAIN_PARAMS.debug:
        df_train = df_folds[df_folds["fold"] != fold].sample(
            LOADER_PARAMS.train_loader["batch_size"] * 8
        )
        df_valid = df_folds[df_folds["fold"] == fold].sample(
            LOADER_PARAMS.train_loader["batch_size"] * 8
        )
        df_oof = df_valid.copy()
    else:
        df_train = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
        df_valid = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
        # Initiate OOF dataframe for this fold (same as df_valid).
        df_oof = df_valid.copy()

    dataset_train = dataset.CustomDataset(
        df_train, transforms=transformation.get_train_transforms(), mode="train"
    )
    dataset_valid = dataset.CustomDataset(
        df_valid, transforms=transformation.get_valid_transforms(), mode="train"
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        **LOADER_PARAMS.train_loader,
        worker_init_fn=utils.seed_worker,
        generator=g,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        **LOADER_PARAMS.valid_loader,
        worker_init_fn=utils.seed_worker,
        generator=g,
    )

    return train_loader, valid_loader, df_oof
