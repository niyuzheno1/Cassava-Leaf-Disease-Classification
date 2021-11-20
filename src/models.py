import torch
import timm
from config import global_params, config
from typing import Dict, Callable, Tuple, Union
import functools
import torchsummary
from src import utils

MODEL_PARAMS = global_params.ModelParams

device = config.DEVICE


class CustomNeuralNet(torch.nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_PARAMS.model_name,
        out_features: int = MODEL_PARAMS.output_dimension,
        in_channels: int = MODEL_PARAMS.input_channels,
        pretrained: bool = MODEL_PARAMS.pretrained,
    ):
        """[summary]

        Args:
            model_name ([type], optional): [description]. Defaults to model_params.model_name.
            out_features ([type], optional): [description]. Defaults to model_params.output_dimension.
            in_channels ([type], optional): [description]. Defaults to model_params.input_channels.
            pretrained ([type], optional): [description]. Defaults to model_params.pretrained.
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_channels
        )
        config.logger.info(
            f"Model: {model_name} \nPretrained: {pretrained} \nIn Channels: {in_channels}"
        )

        self.backbone.reset_classifier(
            num_classes=0, global_pool="avg"
        )  # removes head from backbone

        self.in_features = self.backbone.num_features

        # self.single_head_fc = torch.nn.Sequential(
        #     torch.nn.Linear(self.in_features, self.in_features),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(self.in_features, out_features),
        # )
        self.single_head_fc = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, out_features),
        )
        self.architecture: Dict[str, Callable] = {
            "backbone": self.backbone,
            "bottleneck": None,
            "head": self.single_head_fc,
        }

    def extract_features(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the features mapping logits from the model. This is the output from the backbone of a CNN.

        Args:
            image (torch.FloatTensor): [description]

        Returns:
            [type]: [description]
        """
        feature_logits = self.architecture["backbone"](image)
        return feature_logits

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """The forward call of the model."""

        feature_logits = self.extract_features(image)
        classifier_logits = self.architecture["head"](feature_logits)

        return classifier_logits

    def get_last_layer(self):
        """Get the last layer information of TIMM Model.

        Returns:
            [type]: [description]
        """
        last_layer_name = None
        for name, param in self.model.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        linear_layer = functools.reduce(
            getattr, last_layer_attributes, self.model
        )
        # reduce applies to a list recursively and reduce
        in_features = functools.reduce(
            getattr, last_layer_attributes, self.model
        ).in_features
        return last_layer_attributes, in_features, linear_layer


# see pytorch model summary using torchsummary
def torchsummary_wrapper(
    model: CustomNeuralNet, image_size: Tuple[int, int, int]
) -> torchsummary.model_statistics.ModelStatistics:
    """A torch wrapper to print out layers of a Model.

    Args:
        model (CustomNeuralNet): Model.
        image_size (Tuple[int, int, int]): Image size as a tuple of (channels, height, width).

    Returns:
        model_summary (torchsummary.model_statistics.ModelStatistics): Model summary.
    """

    model_summary = torchsummary.summary(model, image_size)
    return model_summary


def forward_pass(
    model: CustomNeuralNet,
) -> Union[
    torch.FloatTensor,
    torch.LongTensor,
    torchsummary.model_statistics.ModelStatistics,
]:
    """Performs a forward pass of a tensor through the model.

    Args:
        model (CustomNeuralNet): Model to be used for the forward pass.

    Returns:
        X, y, summary (Union[torch.FloatTensor, torch.LongTensor, torchsummary.model_statistics.ModelStatistics]): Returns the X, y, and summary of the model.
    """
    utils.seed_all()
    model.to(device)

    batch_size = 4
    image_size = (3, 224, 224)
    try:
        config.logger.info("Model Summary:")
        torchsummary.summary(model, image_size)
    except RuntimeError:
        config.logger.debug("Check the channel number.")

    X = torch.randn((batch_size, *image_size)).to(device)
    y = model(image=X)
    config.logger.info("Forward Pass Successful!")
    config.logger.info(f"x: {X.shape} \ny: {y.shape}")
    config.logger.info(f"x[0][0][0]: {X[0][0][0][0]} \ny[0][0][0]: {y[0][0]}")

    return X, y
