from argparse import ArgumentParser
from typing import Union, Type

import torch
from torch import nn

from bitorch.datasets.base import BasicDataset
from bitorch.layers.qconv import QConv1d, QConv2d, QConv3d
from bitorch.layers.qconv_noact import QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct


class Model(nn.Module):
    """Base class for Bitorch models"""
    name = "None"

    def __init__(self, dataset: Union[BasicDataset, Type[BasicDataset]]) -> None:
        super(Model, self).__init__()
        self._model = nn.Module()
        self._dataset = dataset

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        """allows additions to the argument parser if required, e.g. to add layer count, etc.

        ! please note that the inferred variable names of additional cli arguments are passed as
        keyword arguments to the constructor of this class !

        Args:
            parser (ArgumentParser): the argument parser
        """
        pass

    def model(self) -> nn.Module:
        """getter method for model

        Returns:
            Module: the main torch.nn.Module of this model
        """
        return self._model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor through the model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the model output
        """
        return self._model(x)

    def initialize(self):
        """initializes model weights a little differently for BNNs."""
        for m in self._model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if isinstance(m, (QConv1d, QConv2d, QConv3d, QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct)):
                    nn.init.xavier_normal_(m.weight)
                    # nn.init.xavier_normal_(m.weight, gain=1e-2)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight, gain=1e-2)
