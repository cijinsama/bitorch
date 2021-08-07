from bitorch.datasets.base import DatasetBaseClass
from .base import Model
from typing import List
import torch
import argparse
from torch import nn
from torch.nn import Module

from bitorch.layers import QConv2d
from bitorch.models.common_layers import make_initial_layers

__all__ = ['resnetE18', 'resnetE34', 'Resnet_E']


class BasicBlock(Module):
    """BasicBlock from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    This is used for ResNetE layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing
            ResNetE uses the full-precision downsampling layer
        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block, i.e. two binary convolutions with batchnorms in between. Check referenced paper for
        more details.

        Returns:
            nn.Sequential: the basic building block body model
        """
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False,
                    input_quantization="sign", weight_quantization="sign"),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x through the building block.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.
        """
        residual = x
        if self.shall_downsample:
            residual = self.downsample(x)
        x = self.body(x)

        return x + residual


class SpecificResnet(Module):
    """Superclass for ResNet models"""

    def __init__(self, classes: int, channels: list) -> None:
        """builds feature and output layers

        Args:
            classes (int): number of output classes
            channels (list): the channels used in the net
        """
        super(SpecificResnet, self).__init__()
        self.features = nn.Sequential()
        self.output_layer = nn.Linear(channels[-1], classes)

    def make_layer(self, layers: int, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """builds a layer by stacking blocks in a sequential models.

        Args:
            layers (int): the number of blocks to stack
            in_channels (int): the input channels of this layer
            out_channels (int): the output channels of this layer
            stride (int): the stride to be used in the convolution layers

        Returns:
            nn.Sequential: the model containing the building blocks
        """

        # this tricks adds shortcut connections between original resnet blocks
        # we multiple number of blocks by 2, but add only one layer instead of two in each block
        layers = layers * 2

        layer_list: List[nn.Module] = []
        layer_list.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(layers - 1):
            layer_list.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layer_list)

    def make_feature_layers(self, layers: list, channels: list) -> nn.Sequential:
        """builds the given layers with the specified block.

        Args:
            layers (list): the number of blocks each layer shall consist of
            channels (list): the channels

        Returns:
            nn.Sequential: [description]
        """
        feature_layers: List[nn.Module] = []
        for idx, num_layer in enumerate(layers):
            stride = 1 if idx == 0 else 2
            feature_layers.append(self.make_layer(num_layer, channels[idx], channels[idx + 1], stride))
        return nn.Sequential(*feature_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor through the resnet modules

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: forwarded tensor
        """
        x = self.features(x)
        x = self.output_layer(x)
        return x


class ResNetE(SpecificResnet):
    """ResNetE-18 model from
    `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    """

    def __init__(
            self,
            layers: list,
            channels: list,
            classes: int,
            initial_layers: str = "imagenet",
            image_channels: int = 3) -> None:
        """Creates ResNetE model.

        Args:
            layers (list): layer sizes
            channels (list): channel num used for input/output channel size of layers. there must always be one more
                channels than there are layers.
            classes (int): number of output classes
            initial_layers (str, optional): name of set for initial layers. refer to common_layers.py.
                Defaults to "imagenet".
            image_channels (int, optional): input channels of images. Defaults to 3.

        Raises:
            ValueError: raised if the number of channels does not match number of layer + 1
        """
        super(ResNetE, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!")

        feature_layers: List[nn.Module] = []
        feature_layers.append(nn.BatchNorm2d(image_channels, eps=2e-5))
        feature_layers.append(make_initial_layers(initial_layers, image_channels, channels[0]))
        feature_layers.append(nn.BatchNorm2d(channels[0]))

        feature_layers.append(self.make_feature_layers(layers, channels))

        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


"""
ResNet-e specifications
"""


class Resnet_E(Model):

    name = "ResnetE"

    resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
                   34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512])}

    def __init__(
            self,
            num_layers: int,
            dataset: DatasetBaseClass) -> None:
        super(Resnet_E, self).__init__(dataset)
        print(dataset.shape)
        self._model = self.create(num_layers, self._dataset.num_classes,
                                  self._dataset.name, self._dataset.shape[1])
        self.name += f"{str(num_layers)}"

    def create(self,
               num_layers: int,
               classes: int = 1000,
               initial_layers: str = "imagenet",
               image_channels: int = 3) -> Module:
        """Creates a ResNetE complying to given layer number.

        Args:
            num_layers (int): number of layers to be build.
            classes (int, optional): number of output classes. Defaults to 1000.
            initial_layers (str, optional): name of set of initial layers to be used. Defaults to "imagenet".
            image_channels (int, optional): number of channels of input images. Defaults to 3.

        Raises:
            ValueError: raised if no resnet specification for given num_layers is listed in the resnet_spec dict above

        Returns:
            Module: resnetE model
        """
        if num_layers not in self.resnet_spec:
            raise ValueError(f"No resnet spec for {num_layers} available!")

        block_type, layers, channels = self.resnet_spec[num_layers]

        return ResNetE(layers, channels, classes, initial_layers, image_channels)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--resnetE-num-layers", type=int, choices=[18, 34], required=True,
                            help="number of layers to be used inside resnetE")


def resnetE18(dataset: DatasetBaseClass) -> Module:
    """ResNetE-18 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet18_v1 model
    """
    return Resnet_E(18, dataset)


def resnetE34(dataset: DatasetBaseClass) -> Module:
    """ResNetE-34 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet34_v1 model
    """
    return Resnet_E(34, dataset)
