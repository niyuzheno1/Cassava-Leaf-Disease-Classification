from typing import Dict, Union

import cv2
import pandas as pd
import torch
from config import global_params

AUG = global_params.AugmentationParams(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], image_size=224)


class PawpularityDataset(torch.utils.data.Dataset):
    """Dataset class for the Pawpularity dataset."""

    def __init__(self, df: pd.DataFrame, transforms=None, mode: str = "train"):
        """Constructor for the dataset class.

        Args:
            df (pd.DataFrame): [description]
            transforms ([type], optional): [description]. Defaults to None.
            mode (str, optional): Defaults to "train". One of ['train', 'valid', 'test', 'gradcam']
        """
        self.image_path = df["image_path"].values
        self.image_ids = df["Id"].values
        self.df = df
        self.targets = torch.from_numpy(df["Pawpularity"].values).float() if mode != "test" else None

        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Union[Dict[str, torch.FloatTensor], Dict[str, torch.FloatTensor]]:
        """Implements the getitem method: https://www.geeksforgeeks.org/__getitem__-and-__setitem__-in-python/

        Args:
            index (int): index of the dataset.

        Returns:
            Union[Dict[str, torch.FloatTensor], Dict[str, torch.FloatTensor]]: dictionary containing the image and the target.
        """
        image_path = self.image_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(image, (AUG.image_size, AUG.image_size)).copy()  # needed for gradcam.

        target = self.targets[index] if self.mode != "test" else None  # Get target for all modes except for test.

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if (self.mode == "train") or (self.mode == "valid"):
            return {"X": torch.FloatTensor(image), "y": torch.FloatTensor(target)}
        elif self.mode == "test":
            return {"X": torch.FloatTensor(image)}
        elif self.mode == "gradcam":
            return {
                "X": torch.FloatTensor(image),
                "y": torch.FloatTensor(target),
                "original_image": torch.FloatTensor(original_image),
                "image_id": self.image_ids[index],
            }

        return None
