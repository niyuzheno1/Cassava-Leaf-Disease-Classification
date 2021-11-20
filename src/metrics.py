from collections import defaultdict
import torch
from sklearn import metrics
from typing import List

import numpy as np
import torchmetrics
from torchmetrics.classification import auroc


class AverageLossMeter:
    """
    Computes and stores the average and current loss
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class MetricMonitor:
    """Monitor Metrics"""

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(
            lambda: {"metric_score": 0, "count": 0, "average_score": 0}
        )

    def update(self, metric_name, metric_score):
        metric = self.metrics[metric_name]

        metric["metric_score"] += metric_score
        metric["count"] += 1
        metric["average_score"] = metric["metric_score"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["average_score"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def multiclass_label_binarize(
    y: np.ndarray, class_labels: List[int], pos_label=1, neg_label=0
):
    """Binarize labels in one-vs-all fashion.
    # TODO: to replace with the above vstack method.

    Args:
        y (np.ndarray) Sequence of integer labels to encode
        class_labels (array-like) Labels for each class
        pos_label (int) Value for positive labels
        neg_label (int) Value for negative labels
    Returns:
        np.ndarray of shape (n_samples, n_classes) Encoded dataset
    """
    if isinstance(y, list):
        y = np.asarray(y)

    columns = [
        np.where(y == label, pos_label, neg_label) for label in class_labels
    ]

    return np.column_stack(columns)


def multiclass_roc_auc_score(y_true, y_score, classes=None):
    """Compute ROC-AUC score for each class in a multiclass dataset.

    Args:
        y_true (np.ndarray of shape (n_samples, n_classes)) True labels
        y_score (np.ndarray of shape (n_samples, n_classes)) Target scores
        classes (array-like of shape (n_classes,)) List of dataset classes. If `None`,
            the lexicographical order of the labels in `y_true` is used.

    Returns:
        array-like: ROC-AUC score for each class, in the same order as `classes`
    """
    classes = np.unique(y_true) if classes is None else classes

    y_true_multiclass = multiclass_label_binarize(y_true, class_labels=classes)

    def oneclass_roc_auc_score(class_id):
        y_true_class = y_true_multiclass[:, class_id]
        y_score_class = y_score[:, class_id]

        fpr, tpr, _ = metrics.roc_curve(
            y_true=y_true_class, y_score=y_score_class, pos_label=1
        )

        return metrics.auc(fpr, tpr)

    return [
        oneclass_roc_auc_score(class_id) for class_id in range(len(classes))
    ]


def multiclass_roc_auc_score_torch(y_true, y_prob, num_classes=None):
    """Compute ROC-AUC score for each class in a multiclass dataset.

    Args:
        y_true (np.ndarray of shape (n_samples, n_classes)) True labels
        y_prob (np.ndarray of shape (n_samples, n_classes)) Target scores
        classes (array-like of shape (n_classes,)) List of dataset classes. If `None`,
            the lexicographical order of the labels in `y_true` is used.

    Returns:
        array-like: ROC-AUC score for each class, in the same order as `classes`
    """

    auroc_all_classes = []
    roc = torchmetrics.ROC(num_classes=num_classes)
    y_true = torch.flatten(y_true)
    print(y_prob.shape)
    print(y_true.shape)
    fpr_all_classes, tpr_all_classes, _ = roc(y_prob, y_true)
    for fpr, tpr in zip(fpr_all_classes, tpr_all_classes):
        curr_class_auroc = torchmetrics.AUC(reorder=True)(fpr, tpr)
        auroc_all_classes.append(curr_class_auroc)

    return auroc_all_classes, torch.mean(torch.stack(auroc_all_classes), dim=0)
