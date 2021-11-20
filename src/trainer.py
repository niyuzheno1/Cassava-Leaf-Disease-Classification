import time


import numpy as np

import torch

from config import config, global_params
from src import metrics, models


from tqdm.auto import tqdm
from typing import List, Dict, Union
from src import callbacks

from pathlib import Path

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM


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
        # TODO: how to add more metrics? wandb log too. Maybe save to model artifacts?
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

    def get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]

        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);

        Returns:
            [type]: [description]
        """
        # TODO: To implement Ian's Results class here so that we can return as per the following link: https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        # TODO: To think whether include num_classes, threshold etc in the arguments.
        torchmetrics_accuracy = metrics.accuracy_score_torch(
            y_trues,
            y_preds,
            num_classes=TRAIN_PARAMS.num_classes,
            threshold=0.5,
        )

        auroc_dict = metrics.multiclass_roc_auc_score_torch(
            y_trues,
            y_probs,
            num_classes=TRAIN_PARAMS.num_classes,
        )

        auroc_all_classes, macro_auc = (
            auroc_dict["auroc_per_class"],
            auroc_dict["macro_auc"],
        )

        # TODO: To check robustness of the code for confusion matrix.
        # macro_cm = metrics.tp_fp_tn_fn_binary(
        #     y_true=y_trues, y_prob=y_probs, class_labels=[0, 1, 2, 3, 4]
        # )

        return {"accuracy": torchmetrics_accuracy, "macro_auroc": macro_auc}

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
        self.best_valid_loss = np.inf

        config.logger.info(
            f"\nTraining on Fold {fold} and using {self.params.model_name}\n"
        )

        for _epoch in range(1, self.params.epochs + 1):

            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)

            ############################ Start of Training #############################

            train_start_time = time.time()

            train_one_epoch_dict = self.train_one_epoch(train_loader)
            train_loss = train_one_epoch_dict["train_loss"]

            # total time elapsed for this epoch
            train_time_elapsed = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - train_start_time)
            )

            config.logger.info(
                f"[RESULT]: Train. Epoch {_epoch} | Avg Train Summary Loss: {train_loss:.3f} | "
                f"Learning Rate: {curr_lr:.5f} | Time Elapsed: {train_time_elapsed}\n"
            )

            ########################### End of Training #################################

            ########################### Start of Validation #############################

            val_start_time = time.time()  # start time for validation
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

            valid_metrics_dict = self.get_classification_metrics(
                valid_trues,
                valid_preds,
                valid_probs,
            )

            valid_accuracy, valid_macro_auroc = (
                valid_metrics_dict["accuracy"],
                valid_metrics_dict["macro_auroc"],
            )

            # TODO: Still need save each metric for each epoch into a list history. Rename properly
            # TODO: Log each metric to wandb and log file.

            config.logger.info(
                f"[RESULT]: Validation. Epoch {_epoch} | Avg Val Summary Loss: {valid_loss:.3f} | "
                f"Avg Val Accuracy: {valid_accuracy:.3f} | Avg Val Macro AUROC: {valid_macro_auroc:.3f} | "
                f"Time Elapsed: {valid_elapsed_time}\n"
            )

            # TODO: Log into wandb or something.
            # self.log_scalar("Val Acc", val_dict["valid_rmse"][_epoch - 1], _epoch)

            ########################### End of Validation ##############################

            ########################## Start of Early Stopping ##########################
            ########################## Start of Model Saving ############################

            # User has to choose a few metrics to monitor.
            # Here I chose valid_loss and valid_macro_auroc.

            self.monitored_metric = {
                "metric_name": "valid_macro_auroc",
                "metric_score": torch.clone(valid_macro_auroc),
                "mode": "max",
            }
            # Metric to optimize, either min or max.
            self.best_valid_score = (
                -np.inf if self.monitored_metric["mode"] == "max" else np.inf
            )

            if self.early_stopping is not None:
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=valid_loss
                )
                self.best_valid_loss = best_score

                if early_stop:
                    config.logger.info("Stopping Early!")
                    break
                # TODO: Add save_model_artifacts here as well.
            else:
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss

                if self.monitored_metric["mode"] == "max":
                    if (
                        self.monitored_metric["metric_score"]
                        > self.best_valid_score
                    ):
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]
                else:
                    if (
                        self.monitored_metric["metric_score"]
                        < self.best_valid_score
                    ):
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]

                self.save_model_artifacts(
                    Path(
                        FILES.weight_path,
                        f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                    ),
                    valid_trues,
                    valid_probs,
                )

                config.logger.info(
                    f"\nSaving model with best valid AUROC score: {self.best_valid_score}"
                )

            ########################## End of Early Stopping ############################
            ########################## End of Model Saving ##############################

            ########################## Start of Scheduler ###############################

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

            ########################## End of Scheduler #################################

            ########################## Load Best Model ##################################
            # Load current checkpoint so we can get model's oof predictions, often in the form of probabilities.
            curr_fold_best_checkpoint = self.load(
                Path(
                    FILES.weight_path,
                    f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                )
            )
            ########################## End of Load Best Model ###########################

        return curr_fold_best_checkpoint

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
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
            # .view(-1, 1) if BCELoss
            targets = data["y"].to(self.device, non_blocking=True)
            logits = self.model(inputs)  # Forward pass logits

            batch_size = inputs.shape[0]
            assert targets.size() == (batch_size,)
            assert logits.size() == (batch_size, TRAIN_PARAMS.num_classes)

            y_train_prob = torch.nn.Softmax(dim=1)(logits)

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
        # TODO: Consider enhancement that returns the same dict as valid_one_epoch.
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
                valid_bar.set_description(f"Validation. Loss: {metric_monitor}")

                # For OOF score and other computation.
                # TODO: Consider giving numerical example. Consider rolling back to targets.cpu().numpy() if torch fails.
                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

                # TODO: To make this look like what Ian and me did in our breast cancer project.
                # TODO: For softmax predictions, then valid_probs must be of shape []

                # running loss
                # self.log_scalar("running_val_loss", curr_batch_val_loss.data.item(), step)

        # We should work with numpy arrays.
        # vstack here to stack the list of numpy arrays.
        # a = [np.asarray([1,2,3]), np.asarray([4,5,6])]
        # np.vstack(a) -> array([[1, 2, 3], [4, 5, 6]])
        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
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
        """Log the weights of the model to both MLflow and TensorBoard
        Args:
            step ([type]): [description]
        """
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

    def save_model_artifacts(
        self,
        path: str,
        valid_trues: torch.Tensor,
        valid_probs: torch.Tensor,
    ) -> None:
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
                "oof_trues": valid_trues,
                "oof_preds": valid_probs,
            },
            path,
        )

    def load(self, path: str):
        """Load a model checkpoint from the given path."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        return checkpoint
