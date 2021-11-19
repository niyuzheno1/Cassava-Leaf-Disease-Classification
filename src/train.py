from config import global_params
import torch

# Metrics
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from collections import defaultdict
import gc
from src import dataset
import numpy as np

scheduler_params = global_params.SchedulerParams()


def get_scheduler(optimizer, scheduler_params=scheduler_params):

    return getattr(torch.optim.lr_scheduler, scheduler_params.scheduler_name)(
        optimizer=optimizer, **scheduler_params.scheduler_params
    )


def use_rmse_score(output, target):
    y_pred = torch.sigmoid(output).cpu()
    y_pred = y_pred.detach().numpy() * 100
    target = target.cpu() * 100

    return mean_squared_error(target, y_pred, squared=False)


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def mixup_data(x, z, y, params):
    if params["mixup_alpha"] > 0:
        lam = np.random.beta(params["mixup_alpha"], params["mixup_alpha"])
    else:
        lam = 1

    batch_size = x.size()[0]
    if params["device"].type == "cuda":
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_z = lam * z + (1 - lam) * z[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_z, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_fn(
    train_loader: torch.utils.data.DataLoader,
    model,
    criterion,
    optimizer,
    epoch,
    params,
    scheduler=None,
):
    metric_monitor = MetricMonitor()
    model.train()
    train_bar = tqdm(train_loader)

    for step, data in enumerate(train_bar, start=1):
        images, target = data["X"], data["y"]
        if params["mixup"]:
            images, dense, target_a, target_b, lam = mixup_data(
                images, dense, target.view(-1, 1), params
            )
            images = images.to(params["device"], dtype=torch.float)
            # dense = dense.to(params["device"], dtype=torch.float)
            target_a = target_a.to(params["device"], dtype=torch.float)
            target_b = target_b.to(params["device"], dtype=torch.float)
        else:
            images = images.to(params["device"], non_blocking=True)
            # dense = dense.to(params["device"], non_blocking=True)

            target = (
                target.to(params["device"], non_blocking=True)
                .float()
                .view(-1, 1)
            )

        output = model(images)

        if params["mixup"]:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)

        rmse_score = use_rmse_score(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("RMSE", rmse_score)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        train_bar.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


def validate_fn(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, data in enumerate(stream, start=1):
            images, target = data["X"], data["y"]
            images = images.to(params["device"], non_blocking=True)
            # dense = dense.to(params["device"], non_blocking=True)
            target = (
                target.to(params["device"], non_blocking=True)
                .float()
                .view(-1, 1)
            )
            output = model(images)
            loss = criterion(output, target)
            rmse_score = use_rmse_score(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("RMSE", rmse_score)
            stream.set_description(
                f"Epoch: {epoch:02}. Valid. {metric_monitor}"
            )

            targets = (target.detach().cpu().numpy() * 100).tolist()
            outputs = (
                torch.sigmoid(output).detach().cpu().numpy() * 100
            ).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets
