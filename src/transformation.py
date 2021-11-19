from typing import Dict

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config import global_params

AUG = global_params.AugmentationParams(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], image_size=224)


def get_train_transforms(image_size: int = AUG.image_size):
    """Performs Augmentation on training data.

    Args:
        image_size (int, optional): [description]. Defaults to AUG.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=AUG.mean, std=AUG.std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(image_size: int = AUG.image_size):
    """Performs Augmentation on validation data.

    Args:
        image_size (int, optional): [description]. Defaults to AUG.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=AUG.mean, std=AUG.std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def get_gradcam_transforms(image_size: int = AUG.image_size):
    """Performs Augmentation on gradcam data.

    Args:
        image_size (int, optional): [description]. Defaults to AUG.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=AUG.mean, std=AUG.std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ]
    )


def get_inference_transforms(image_size: int = AUG.image_size) -> Dict[str, albumentations.Compose]:
    """Performs Augmentation on test dataset.
    Returns the transforms for inference in a dictionary which can hold TTA transforms.

    Args:
        image_size (int, optional): [description]. Defaults to AUG.image_size.

    Returns:
        Dict[str, albumentations.Compose]: [description]
    """

    transforms_dict = {
        "transforms_test": albumentations.Compose(
            [
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(mean=AUG.mean, std=AUG.std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ]
        ),
        # "tta_hflip": albumentations.Compose(
        #     [
        #         albumentations.HorizontalFlip(p=1.0),
        #         albumentations.Resize(image_size, image_size),
        #         ToTensorV2(),
        #     ]
        # ),
    }

    return transforms_dict
