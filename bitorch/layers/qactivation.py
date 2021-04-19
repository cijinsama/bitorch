import torch
from torch import nn

from bitorch.layers import layerconfig


class QActivation(nn.Module):
    def __init__(self, activation: str = None) -> None:
        super(QActivation, self).__init__()
        self.activation = layerconfig.config.get_activation_function(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # grad cancel on every quantization function? always ste?
        return self.activation(input_tensor)
