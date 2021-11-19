import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from typing import List
import torchvision


def unnormalize(normalized_img, mean, std, max_pixel_value=255.0) -> torch.Tensor:
    """TODO: Use https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/7 code and make it a class to include both Normalize and Unnormalize Method.

    Args:
        normalized_img ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]
        max_pixel_value (float, optional): [description]. Defaults to 255.0.

    Returns:
        torch.Tensor: [description]
    """
    # normalized_img = (unnormalized_img - mean * max_pixel_value) / (std * max_pixel_value)
    # unnormalized_img = normalized_img * (std * max_pixel_values) + mean * max_pixel_values

    unnormalized = torch.zeros(normalized_img.size(), dtype=torch.float64)
    unnormalized[0, :, :] = normalized_img[0, :, :] * (std[0] * max_pixel_value) + mean[0] * max_pixel_value
    unnormalized[1, :, :] = normalized_img[1, :, :] * (std[1] * max_pixel_value) + mean[1] * max_pixel_value
    unnormalized[2, :, :] = normalized_img[2, :, :] * (std[2] * max_pixel_value) + mean[2] * max_pixel_value

    return unnormalized


def show_image(
    loader: torch.utils.data.Dataset,
    nrows: int = 3,
    ncols: int = 4,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    one_channel: bool = False,
):
    """Plot a grid of image from Dataloader.

    Args:
        train_dataset (torch.utils.data.Dataset): [description]
        nrows (int, optional): [description]. Defaults to 3.
        ncols (int, optional): [description]. Defaults to 4.
        mean (List[float], optional): [description]. Defaults to None.
        std (List[float], optional): [description]. Defaults to None.
    """

    dataiter = iter(loader)

    one_batch_images, one_batch_targets = dataiter.next()["X"], dataiter.next()["y"]
    # TODO: FIX UNNORMALIZE not showing properly.
    # one_batch_images = [unnormalize(image, mean, std, max_pixel_value=255.0) for image in one_batch_images]

    # create grid of images
    image_grid = torchvision.utils.make_grid(one_batch_images)

    if one_channel:
        pass

    image_grid = image_grid.numpy()
    if one_channel:
        plt.imshow(image_grid, cmap="Greys")
    else:
        plt.imshow(np.transpose(image_grid, (1, 2, 0)))
    plt.show()

    return image_grid

    # show images
    # matplotlib_imshow(img_grid, one_channel=True)
    # plt.figure(figsize=(20, 10))

    # for _ in range(nrows):
    #     for col in range(ncols):

    #         rand = random.randint(0, len(train_dataset))
    #         image, label = train_dataset[rand]["X"], train_dataset[rand]["y"]
    #         image = unnormalize(image, mean, std, max_pixel_value=255.0)

    #         plt.subplot(1, ncols, col % ncols + 1)
    #         plt.axis("off")
    #         plt.imshow(image.permute(2, 1, 0))
    #         plt.title(f"Pawpularity: {label}")
    #         plt.show()
