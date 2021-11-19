from typing import Dict

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config import global_params

TRANSFORMS = global_params.AugmentationParams(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], image_size=256
)


def get_train_transforms(image_size: int = TRANSFORMS.image_size):
    """Performs Augmentation on training data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.RandomResizedCrop(
                height=image_size, width=image_size
            ),
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Cutout(p=0.5),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(image_size: int = TRANSFORMS.image_size):
    """Performs Augmentation on validation data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_gradcam_transforms(image_size: int = TRANSFORMS.image_size):
    """Performs Augmentation on gradcam data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_inference_transforms(
    image_size: int = TRANSFORMS.image_size,
) -> Dict[str, albumentations.Compose]:
    """Performs Augmentation on test dataset.
    Returns the transforms for inference in a dictionary which can hold TTA transforms.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        Dict[str, albumentations.Compose]: [description]
    """

    transforms_dict = {
        "transforms_test": albumentations.Compose(
            [
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(
                    mean=TRANSFORMS.mean,
                    std=TRANSFORMS.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
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
