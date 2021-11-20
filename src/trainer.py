import datetime
import gc
import os
import tempfile
import time
from collections import defaultdict

import numpy as np
import pytz
import torch
from torch._C import device
from config import config, global_params
from src import metrics, models

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from tqdm.auto import tqdm
from typing import List, Dict, Union
from src import dataset, callbacks
import cv2
from pathlib import Path

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

import torchmetrics

FILES = global_params.FilePaths()
TRAIN_PARAMS = global_params.GlobalTrainParams()


def get_optimizer(
    model: models.CustomNeuralNet,
    optimizer_params: global_params.OptimizerParams(),
):
    """Get the optimizer for the model.

    Args:
        model (models.CustomNeuralNet): [description]
        optimizer_params (global_params.OptimizerParams): [description]

    Returns:
        [type]: [description]
    """
    return getattr(torch.optim, optimizer_params.optimizer_name)(
        model.parameters(), **optimizer_params.optimizer_params
    )


def get_scheduler(
    optimizer: torch.optim,
    scheduler_params: global_params.SchedulerParams(),
):
    """Get the scheduler for the optimizer.

    Args:
        optimizer (torch.optim): [description]
        scheduler_params (global_params.SchedulerParams(), optional): [description]. Defaults to SCHEDULER_PARAMS.scheduler_params.

    Returns:
        [type]: [description]
    """

    return getattr(torch.optim.lr_scheduler, scheduler_params.scheduler_name)(
        optimizer=optimizer, **scheduler_params.scheduler_params
    )


class Trainer:
    """Object used to facilitate training."""

    def __init__(
        self,
        # params: Namespace,
        params,
        model,
        device=torch.device("cpu"),
        optimizer=None,
        scheduler=None,
        # trial=None,
        # writer=None,
        early_stopping: callbacks.EarlyStopping = None,
    ):
        # Set params
        self.params = params
        self.model = model
        self.device = device

        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.trial = trial
        # self.writer = writer
        self.early_stopping = early_stopping

        # list to contain various train metrics
        self.train_loss_history: List = []
        self.train_metric_history: List = []
        self.val_loss_history: List = []
        self.val_rmse_history: List = []

    def train_criterion(self, y_true, y_logits):
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.

        The below example applies for CrossEntropyLoss.

        Args:
            y_true ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is 0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C−1
                               If containing class probabilities, same shape as the input.

        Returns:
            [type]: [description]
        """

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_logits, y_true)
        return loss

    def valid_criterion(self, y_true, y_logits):
        """Validation Loss Function.

        Args:
            y_true ([type]): [description]
            y_logits ([type]): [description]

        Returns:
            [type]: [description]
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_logits, y_true)
        return loss

    def get_classification_metrics(self, y_trues, y_preds, y_probs):
        # TODO: To implement Ian's Results class here so that we can return as per the following link: https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/

        accuracy = accuracy_score(y_trues, y_preds)
        torchmetrics_accuracy = torchmetrics.Accuracy(
            threshold=0.5,
            num_classes=TRAIN_PARAMS.num_classes,
            average="micro",
            top_k=None,
        )(torch.as_tensor(y_trues), torch.as_tensor(y_preds))
        print(torchmetrics_accuracy)

        return {"accuracy": accuracy}

    def get_lr(self, optimizer) -> float:
        """Get the learning rate of the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        Args:
            self.optimizer (torch.optim): [description]
        Returns:
            float: [description]
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def run_train(self, train_loader: torch.utils.data.DataLoader) -> Dict:
        """[summary]
        Args:
            train_loader (torch.utils.data.DataLoader): [description]
        Returns:
            Dict: [description]
        """

        # train start time
        train_start_time = time.time()

        # get avg train loss for this epoch
        train_one_epoch_dict = self.train_one_epoch(train_loader)
        self.train_loss = train_one_epoch_dict["train_loss"]
        self.train_loss_history.append(self.train_loss)

        # total time elapsed for this epoch

        train_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - train_start_time)
        )

        return {
            "train_loss": self.train_loss_history,
            "time_elapsed": train_elapsed_time,
        }

    # def run_val(self, valid_loader: torch.utils.data.DataLoader) -> Dict:
    #     """[summary]
    #     Args:
    #         valid_loader (torch.utils.data.DataLoader): The validation data loader.
    #     Returns:
    #         Dict: [description]
    #     """

    #     # train start time
    #     val_start_time = time.time()

    #     # get avg train loss for this epoch
    #     valid_one_epoch_dict = self.valid_one_epoch(valid_loader)

    #     (
    #         self.valid_loss,
    #         self.valid_trues,
    #         self.valid_logits,
    #         self.valid_preds,
    #         self.valid_probs,
    #     ) = (
    #         valid_one_epoch_dict["valid_loss"],
    #         valid_one_epoch_dict["valid_trues"],
    #         valid_one_epoch_dict["valid_logits"],
    #         valid_one_epoch_dict["valid_preds"],
    #         valid_one_epoch_dict["valid_probs"],
    #     )

    #     # total time elapsed for this epoch
    #     val_elapsed_time = time.strftime(
    #         "%H:%M:%S", time.gmtime(time.time() - val_start_time)
    #     )

    #     return {
    #         "valid_loss_history": self.valid_loss,
    #         "valid_trues_history": self.valid_trues_history,
    #         "valid_preds_history": self.valid_preds_history,
    #         "valid_probs_history": self.valid_probs_history,
    #         "valid_time_elapsed": val_elapsed_time,
    #     }

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        fold: int = None,
    ):
        """[summary]

        Args:
            train_loader (torch.utils.data.DataLoader): [description]
            val_loader (torch.utils.data.DataLoader): [description]
            fold (int, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        best_val_loss = np.inf

        config.logger.info(
            f"Training on Fold {fold} and using {self.params.model_name}"
        )

        for _epoch in range(1, self.params.epochs + 1):
            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)
            # get current time
            timestamp = datetime.datetime.now(
                pytz.timezone("Asia/Singapore")
            ).strftime("%Y-%m-%d %H-%M-%S")
            # print current time and lr
            config.logger.info("\nTime: {}\nLR: {}".format(timestamp, curr_lr))

            train_dict: Dict = self.run_train(train_loader)

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.
            print(train_dict["train_loss"][_epoch - 1])
            config.logger.info(
                f"[RESULT]: Train. Epoch {_epoch} | Avg Train Summary Loss: {train_dict['train_loss'][_epoch-1]:.3f} | "
                f"Time Elapsed: {train_dict['time_elapsed']}"
            )

            ########################### Start of Validation ###########################
            # valid start time
            val_start_time = time.time()
            valid_one_epoch_dict = self.valid_one_epoch(valid_loader)

            (
                valid_loss,
                valid_trues,
                valid_logits,
                valid_preds,
                valid_probs,
            ) = (
                valid_one_epoch_dict["valid_loss"],
                valid_one_epoch_dict["valid_trues"],
                valid_one_epoch_dict["valid_logits"],
                valid_one_epoch_dict["valid_preds"],
                valid_one_epoch_dict["valid_probs"],
            )

            # total time elapsed for this epoch
            valid_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - val_start_time)
            )

            valid_accuracy = self.get_classification_metrics(
                valid_trues,
                valid_preds,
                valid_probs,
            )
            print(valid_loss)
            print(valid_accuracy)
            # TODO: Still need save each metric for each epoch into a list history. Rename properly
            # TODO: Log each metric to wandb and log file.

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.

            config.logger.info(
                f"[RESULT]: Validation. Epoch {_epoch} | Avg Val Summary Loss: {self.val_dict['val_loss'][_epoch-1]:.3f} | "
                f"Time Elapsed: {self.valid_time_elapsed}"
            )

            # self.log_scalar("Val Acc", val_dict["valid_rmse"][_epoch - 1], _epoch)

            # Early Stopping code block
            if self.early_stopping is not None:
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=self.val_dict["val_loss"][_epoch - 1]
                )
                self.best_loss = best_score
                # TODO: SAVE MODEL
                # self.save(
                #     "{self.param.model['model_name']}_best_loss_fold_{fold}.pt")
                if early_stop:
                    config.logger.info("Stopping Early!")
                    break
            else:
                if self.val_dict["val_loss"][_epoch - 1] < best_val_loss:
                    best_val_loss = self.val_dict["val_loss"][_epoch - 1]

                # if self.val_dict["valid_rmse"][_epoch - 1] < best_rmse:
                #     best_rmse = self.val_dict["valid_rmse"][_epoch - 1]
                #     self.save_model_artifacts(
                #         Path(
                #             FILES.weight_path,
                #             f"{self.params.model_name}_best_rmse_fold_{fold}.pt",
                #         )
                #     )
                #     config.logger.info(
                #         f"Saving model with best RMSE: {best_rmse}"
                #     )

            # Scheduler Step code block: note the special case for ReduceLROnplateau.
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.val_dict["val_acc"])
                else:
                    self.scheduler.step()

            # Load current checkpoint so we can get model's oof predictions, often in the form of probabilities.
            curr_fold_best_checkpoint = self.load(
                Path(
                    FILES.weight_path,
                    f"{self.params.model_name}_best_rmse_fold_{fold}.pt",
                )
            )
            return curr_fold_best_checkpoint

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> float:
        """Train one epoch of the model."""

        metric_monitor = metrics.MetricMonitor()

        # set to train mode
        self.model.train()
        average_cumulative_train_loss: float = 0.0
        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for step, data in enumerate(train_bar, start=1):
            if self.params.mixup:
                # TODO: Implement MIXUP logic.
                pass

            # unpack
            inputs = data["X"].to(self.device, non_blocking=True)

            # # .view(-1, 1)
            targets = data["y"].to(self.device, non_blocking=True)
            logits = self.model(inputs)  # Forward pass logits

            batch_size = inputs.shape[0]
            assert targets.size() == (batch_size,)
            assert logits.size() == (batch_size, TRAIN_PARAMS.num_classes)

            y_train_prob = torch.nn.Softmax(dim=1)(logits)
            # sigmoid_prob = torch.sigmoid(logits).detach().cpu().numpy()

            self.optimizer.zero_grad()  # reset gradients
            curr_batch_train_loss = self.train_criterion(targets, logits)
            curr_batch_train_loss.backward()  # Backward pass
            metric_monitor.update(
                "Loss", curr_batch_train_loss.item()
            )  # Update loss metric
            train_bar.set_description(f"Train. {metric_monitor}")

            self.optimizer.step()  # Update weights using the optimizer

            # Cumulative Loss
            # Batch/Step 1: curr_batch_train_loss = 10 -> average_cumulative_train_loss = (10-0)/1 = 10
            # Batch/Step 2: curr_batch_train_loss = 12 -> average_cumulative_train_loss = 10 + (12-10)/2 = 11 (Basically (10+12)/2=11)
            # Essentially, average_cumulative_train_loss = loss over all batches / batches
            average_cumulative_train_loss += (
                curr_batch_train_loss.detach().item()
                - average_cumulative_train_loss
            ) / (step)

        # self.log_weights(step)
        # running loss
        # self.log_scalar("running_train_loss", curr_batch_train_loss.data.item(), step)
        return {"train_loss": average_cumulative_train_loss}

    # @torch.no_grad
    def valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Validate the model on the validation set for one epoch.

        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.

        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """

        self.model.eval()  # set to eval mode
        metric_monitor = metrics.MetricMonitor()
        average_cumulative_valid_loss: float = 0.0
        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for step, data in enumerate(valid_bar, start=1):
                # unpack
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                batch_size = inputs.shape[0]
                assert targets.size() == (batch_size,)
                assert logits.size() == (batch_size, TRAIN_PARAMS.num_classes)

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = torch.nn.Softmax(dim=1)(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.valid_criterion(targets, logits)
                average_cumulative_valid_loss += (
                    curr_batch_val_loss.item() - average_cumulative_valid_loss
                ) / (step)
                valid_bar.set_description(f"Validation. {metric_monitor}")

                # For OOF score and other computation.
                # TODO: Consider giving numerical example.
                valid_trues.extend(targets.cpu().numpy())
                valid_logits.extend(logits.cpu().numpy())
                valid_preds.extend(y_valid_pred.cpu().numpy())
                valid_probs.extend(y_valid_prob.cpu().numpy())

                # TODO: To make this look like what Ian and me did in our breast cancer project.
                # TODO: For softmax predictions, then valid_probs must be of shape []

                # running loss
                # self.log_scalar("running_val_loss", curr_batch_val_loss.data.item(), step)

        # We should work with numpy arrays.
        # vstack here to stack the list of numpy arrays.
        # a = [np.asarray([1,2,3]), np.asarray([4,5,6])]
        # np.vstack(a) -> array([[1, 2, 3], [4, 5, 6]])
        valid_trues, valid_logits, valid_preds, valid_probs = (
            np.vstack(valid_trues),
            np.vstack(valid_logits),
            np.vstack(valid_preds),
            np.vstack(valid_probs),
        )
        num_valid_samples = len(valid_trues)
        assert valid_trues.shape == valid_preds.shape == (num_valid_samples, 1)
        assert (
            valid_logits.shape
            == valid_probs.shape
            == (num_valid_samples, TRAIN_PARAMS.num_classes)
        )

        return {
            "valid_loss": average_cumulative_valid_loss,
            "valid_trues": valid_trues,
            "valid_logits": valid_logits,
            "valid_preds": valid_preds,
            "valid_probs": valid_probs,
        }

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value to both MLflow and TensorBoard
        Args:
            name (str): name of the scalar metric
            value (float): value of the metric
            step (int): either epoch or each step
        """

        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)
        # mlflow.log_metric(name, value, step=step)

    def log_weights(self, step):
        self.writer.add_histogram(
            tag="conv1_weight",
            values=self.model.conv1.weight.data,
            global_step=step,
        )
        # writer.add_summary(writer.add_histogram('weights/conv1/bias',
        #                                         model.conv1.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/conv2/weight',
        #                                         model.conv2.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/conv2/bias',
        #                                                  model.conv2.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc1/weight',
        #                                         model.fc1.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc1/bias',
        #                                         model.fc1.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc2/weight',
        #                                         model.fc2.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc2/bias',
        #                                         model.fc2.bias.data).eval(), step)

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        self.model.eval()
        torch.save(self.model.state_dict(), path)

    def save_model_artifacts(self, path: str) -> None:
        """Save the weight for the best evaluation loss.
        oof_preds: np.array of shape [num_samples, num_classes] and represent the predictions for each fold.
        oof_trues: np.array of shape [num_samples, 1] and represent the true labels for each fold.
        """
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "oof_preds": self.val_dict["valid_probs"],
                "oof_trues": self.val_dict["valid_trues"],
            },
            path,
        )

    def load(self, path: str):
        """Load a model checkpoint from the given path."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        return checkpoint
