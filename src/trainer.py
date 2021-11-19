import datetime
import gc
import os
import tempfile
import time
from collections import defaultdict

import numpy as np
import pytz
import torch
from config import config, global_params
from sklearn import metrics

# Metrics
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from typing import List, Dict
from src import dataset, callbacks
import cv2
from pathlib import Path

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM

FILES = global_params.FilePaths()
train_params = global_params.GlobalTrainParams()


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

        Args:
            y_true ([type]): [description]
            y_logits ([type]): [description]

        Returns:
            [type]: [description]
        """
        loss_fn = torch.nn.BCEWithLogitsLoss()
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
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(y_logits, y_true)
        return loss

    def get_rmse(self, y_true, y_prob):
        """Calculate RMSE.
        We need to unnormalize both inputs by multiplying by 100.

        Args:
            y_true ([type]): [description]
            y_prob ([type]): [description]

        Returns:
            [type]: [description]
        """
        y_true = y_true * 100
        y_prob = y_prob * 100
        return metrics.mean_squared_error(y_true, y_prob, squared=False)

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
        self.train_loss, self.train_rmse = (
            train_one_epoch_dict["train_loss"],
            train_one_epoch_dict["train_rmse"],
        )
        self.train_loss_history.append(self.train_loss)

        # total time elapsed for this epoch

        train_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))

        return {"train_loss": self.train_loss_history, "time_elapsed": train_elapsed_time}

    def run_val(self, val_loader: torch.utils.data.DataLoader) -> Dict:
        """[summary]
        Args:
            val_loader (torch.utils.data.DataLoader): [description]
        Returns:
            Dict: [description]
        """

        # train start time
        val_start_time = time.time()

        # get avg train loss for this epoch
        valid_one_epoch_dict = self.valid_one_epoch(val_loader)
        self.valid_loss, self.valid_rmse, self.valid_probs, self.valid_trues = (
            valid_one_epoch_dict["valid_loss"],
            valid_one_epoch_dict["valid_rmse"],
            valid_one_epoch_dict["valid_probs"],
            valid_one_epoch_dict["valid_trues"],
        )
        self.val_loss_history.append(self.valid_loss)
        self.val_rmse_history.append(self.valid_rmse)

        # total time elapsed for this epoch

        val_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - val_start_time))

        return {
            "val_loss": self.val_loss_history,
            "valid_rmse": self.val_rmse_history,
            "time_elapsed": val_elapsed_time,
            "valid_trues": self.valid_trues,
            "valid_probs": self.valid_probs,
        }

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
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
        best_rmse = np.inf

        config.logger.info(f"Training on Fold {fold} and using {self.params.model_name}")

        for _epoch in range(1, self.params.epochs + 1):
            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)
            # get current time
            timestamp = datetime.datetime.now(pytz.timezone("Asia/Singapore")).strftime("%Y-%m-%d %H-%M-%S")
            # print current time and lr
            config.logger.info("\nTime: {}\nLR: {}".format(timestamp, curr_lr))

            train_dict: Dict = self.run_train(train_loader)

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.

            config.logger.info(
                f"[RESULT]: Train. Epoch {_epoch} | Avg Train Summary Loss: {train_dict['train_loss'][_epoch-1]:.3f} | "
                f"Time Elapsed: {train_dict['time_elapsed']}"
            )

            self.val_dict: Dict = self.run_val(val_loader)

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.

            config.logger.info(
                f"[RESULT]: Validation. Epoch {_epoch} | Avg Val Summary Loss: {self.val_dict['val_loss'][_epoch-1]:.3f} | "
                f"Val RMSE: {self.val_dict['valid_rmse'][_epoch-1]:.3f} | "
                f"Time Elapsed: {self.val_dict['time_elapsed']}"
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

                if self.val_dict["valid_rmse"][_epoch - 1] < best_rmse:
                    best_rmse = self.val_dict["valid_rmse"][_epoch - 1]
                    self.save_model_artifacts(
                        Path(FILES.weight_path, f"{self.params.model_name}_best_rmse_fold_{fold}.pt")
                    )
                    config.logger.info(f"Saving model with best RMSE: {best_rmse}")

            # Scheduler Step code block: note the special case for ReduceLROnplateau.
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.val_dict["val_acc"])
                else:
                    self.scheduler.step()

            # Load current checkpoint so we can get model's oof predictions, often in the form of probabilities.
            curr_fold_best_checkpoint = self.load(
                Path(FILES.weight_path, f"{self.params.model_name}_best_rmse_fold_{fold}.pt")
            )
            return curr_fold_best_checkpoint

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train one epoch of the model."""

        # metric_monitor = MetricMonitor()
        # set to train mode
        self.model.train()
        cumulative_train_loss: float = 0.0
        train_bar = tqdm(train_loader)

        # Iterate over train batches
        for step, data in enumerate(train_bar, start=1):
            if self.params.mixup:
                # TODO: Implement MIXUP logic.
                pass

            # unpack
            inputs = data["X"].to(self.device, non_blocking=True)
            targets = data["y"].to(self.device, non_blocking=True).view(-1, 1)

            self.optimizer.zero_grad()  # reset gradients
            logits = self.model(inputs)  # Forward pass logits

            sigmoid_prob = torch.sigmoid(logits).detach().cpu().numpy()

            train_rmse = self.get_rmse(targets.detach().cpu().numpy(), sigmoid_prob)

            curr_batch_train_loss = self.train_criterion(targets, logits)

            curr_batch_train_loss.backward()  # Backward pass

            self.optimizer.step()  # Update weights using the optimizer

            # Cumulative Loss
            # Batch/Step 1: curr_batch_train_loss = 10 -> cumulative_train_loss = (10-0)/1 = 10
            # Batch/Step 2: curr_batch_train_loss = 12 -> cumulative_train_loss = 10 + (12-10)/2 = 11
            # Essentially, cumulative train loss = loss over all batches / batches
            cumulative_train_loss += (curr_batch_train_loss.detach().item() - cumulative_train_loss) / (step)

            # self.log_weights(step)
            # running loss
            # self.log_scalar("running_train_loss", curr_batch_train_loss.data.item(), step)
        return {"train_loss": cumulative_train_loss, "train_rmse": train_rmse}

    # @torch.no_grad
    def valid_one_epoch(self, val_loader):
        """Validate one training epoch."""
        # TODO: Try to make results type more consistent. eg: sigmoid_prob is detached but targets is not detached.
        # TODO: Make names the same, for example valid_probs should be consistent throughout, however, we have y_probs and valid_probs.
        # set to eval mode
        self.model.eval()

        valid_loss: float = 0.0
        valid_bar = tqdm(val_loader)

        valid_logits, valid_trues, valid_probs = [], [], []

        with torch.no_grad():
            for step, data in enumerate(valid_bar, start=1):
                # unpack
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True).view(-1, 1)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                batch_size = inputs.shape[0]
                assert targets.size() == (batch_size, train_params.num_classes)
                assert logits.size() == (batch_size, train_params.num_classes)

                # Store outputs that are needed to compute various metrics
                # shape = [bs, 1] | shape = [bs, num_classes] if softmax
                # applied along dimension 1.
                sigmoid_prob = torch.sigmoid(logits).cpu().numpy()

                valid_rmse = self.get_rmse(targets.cpu().numpy(), sigmoid_prob)

                curr_batch_val_loss = self.valid_criterion(targets, logits)
                valid_loss += (curr_batch_val_loss.item() - valid_loss) / (step)

                # for OOF score and other computation
                valid_probs.extend(sigmoid_prob)
                valid_trues.extend(targets.cpu().numpy())
                valid_logits.extend(logits.cpu().numpy())
                # TODO: To make this look like what Ian and me did in our breast cancer project.
                # TODO: For softmax predictions, then valid_probs must be of shape []

                # running loss
                # self.log_scalar("running_val_loss", curr_batch_val_loss.data.item(), step)

            # argmax = np.argmax(Y_PROBS, axis=1)
            # correct = np.equal(argmax, np.asarray(Y_TRUES))
            # total = correct.shape[0]
            # # argmax = [1, 2, 1] Y_TRUES = [1, 1, 2] -> correct = [True, False, False] -> num_correct = 1 and total = 3 -> acc = 1/3
            # num_correct = np.sum(correct)
            # accuracy = (num_correct / total) * 100

        return {
            "valid_loss": valid_loss,
            "valid_rmse": valid_rmse,
            "valid_probs": valid_probs,
            "valid_trues": valid_trues,
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
        self.writer.add_histogram(tag="conv1_weight", values=self.model.conv1.weight.data, global_step=step)
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
