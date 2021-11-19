from __future__ import generators, print_function

import os
import sys

from pathlib import Path

BASE_DIR = (
    Path(__file__).parent.parent.absolute().__str__()
)  # C:\Users\reigHns\mnist
sys.path.append(BASE_DIR)


from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import torch

import typer
from sklearn import metrics

from src import (
    plot,
    prepare,
    transformation,
    utils,
    models,
    train,
    inference,
    trainer,
    dataset,
)
from config import config, global_params
from torch._C import device
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import gc
import wandb
import torchsummary

FILES = global_params.FilePaths()
FOLDS = global_params.MakeFolds()
MODEL = global_params.ModelParams()
LOADER_PARAMS = global_params.DataLoaderParams()
TRAIN_PARAMS = global_params.GlobalTrainParams()


device = config.DEVICE


# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Load data from URL and save to local drive."""
    # Download data, pre-caching.
    # datasets.MNIST(root=config.DATA_DIR.absolute(), train=True, download=True)
    # datasets.MNIST(root=config.DATA_DIR.absolute(), train=False, download=True)
    # Save data

    config.logger.info("Data downloaded!")


def train_one_fold(df_folds: pd.DataFrame, fold: int, is_plot: bool = False):
    """Train the model on the given fold."""

    # run = wandb.init(project="Petfinder", entity="reighns", job_type="Train", anonymous="must")
    # wandb.config = Dict or Yaml

    train_loader, valid_loader, df_oof = prepare.prepare_loaders(df_folds, fold)

    if is_plot:
        image_grid = plot.show_image(
            loader=train_loader,
            nrows=1,
            ncols=1,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # writer.add_image("name", image_grid)

    # Model, cost function and optimizer instancing
    model = models.PetNeuralNet().to(device)
    try:
        config.logger.info("Model Summary:")
        print(model)
        torchsummary.summary(model, (3, 224, 224))
    except RuntimeError:
        config.logger.debug("Check the channel number.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS.init_lr,
        weight_decay=TRAIN_PARAMS.weight_decay,
        amsgrad=False,
    )
    scheduler = train.get_scheduler(optimizer)
    reighns_trainer: trainer.Trainer = trainer.Trainer(
        params=TRAIN_PARAMS,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    curr_fold_best_checkpoint = reighns_trainer.fit(
        train_loader, valid_loader, fold
    )

    # TODO: Note that for sigmoid on one class, the OOF score is the positive class.
    df_oof[
        [str(c) + "_oof" for c in range(TRAIN_PARAMS.num_classes)]
    ] = curr_fold_best_checkpoint["oof_preds"]
    df_oof["oof_trues"] = curr_fold_best_checkpoint["oof_trues"]
    # df_oof['error_analysis'] = todo - sort the dataframe by the ones that the model got wrong to see where they are focusing.
    # df_oof["oof_preds"] = curr_fold_best_checkpoint["oof_preds"].argmax(1)

    df_oof.to_csv(Path(FILES.oof_csv, f"oof_fold_{fold}.csv"), index=False)
    del model
    del optimizer
    del scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return df_oof


def train_loop(df_folds: pd.DataFrame, is_plot: bool = False):
    """Perform the training loop on all folds. Here The CV score is the average of the validation fold metric.
    While the OOF score is the aggregation of all validation folds."""

    cv_score_list = []
    df_oof = pd.DataFrame()

    for fold in range(1, FOLDS.num_folds + 1):
        _df_oof = train_one_fold(df_folds=df_folds, fold=fold, is_plot=is_plot)
        df_oof = pd.concat([df_oof, _df_oof])

        # TODO: populate the cv_score_list using a dataframe like breast cancer project.
        # curr_fold_best_score_dict, curr_fold_best_score = get_oof_roc(config, _oof_df)
        # cv_score_list.append(curr_fold_best_score)
        # print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(fold, curr_fold_best_score))

    # print("CV score", np.mean(cv_score_list))
    # print("Variance", np.var(cv_score_list))
    # print("Five Folds OOF", get_oof_roc(config, oof_df))

    df_oof.to_csv(Path(FILES.oof_csv, "oof.csv"), index=False)


if __name__ == "__main__":
    utils.seed_all()

    # @Step 1: Download and load data.
    df_train, df_test, df_folds, sub = prepare.prepare_data()
    # forward_X, forward_y, model_summary = models.forward_pass(
    #     model=models.CustomNeuralNet()
    # )

    # # train_one_fold(df_folds=df_folds, fold=4)
    # train_loop(df_folds=df_folds, is_plot=False)

    # model_dir = Path(FILES.weight_path, MODEL.model_name).__str__()
    # model_dir = r"C:\Users\reighns\petfinder\model\weights\vit_small_patch16_224"
    # pred = inference.inference(df_test, model_dir, sub)
    # _ = inference.show_gradcam()
