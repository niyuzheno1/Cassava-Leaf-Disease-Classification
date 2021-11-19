# Utility functions.
import json
import os
import random
from typing import Dict
from urllib.request import urlopen

import mlflow
import numpy as np
import torch


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id) -> None:
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_json_from_url(url: str) -> Dict:
    """Load JSON data from a URL.
    Args:
        url (str): URL of the data source.
    Returns:
        A dictionary with the loaded JSON data.
    """
    data = json.loads(urlopen(url).read())
    return data


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.
    Args:
        filepath (str): JSON's filepath.
    Returns:
        A dictionary with the data loaded.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.
    Warning:
        This will overwrite any existing file at `filepath`.
    Args:
        d (Dict): dictionary to save.
        filepath (str): location to save the dictionary to as a JSON file.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): sort keys in dict alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_device(cuda: bool) -> torch.device:
    """Set the device for computation.
    Args:
        cuda (bool): Determine whether to use GPU or not (if available).
    Returns:
        Device that will be use for compute.
    """
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    if device.type == "cuda":  # pragma: no cover, simple tensor type setting
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    return device


def delete_experiment(experiment_name: str):
    """Delete an experiment with name `experiment_name`.
    Args:
        experiment_name (str): Name of the experiment.
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
