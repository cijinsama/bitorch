"""Module containing the quantized convolution layer"""

from typing import Type
from torch import Tensor
from bitorch.layers.qactivation import QActivation
from bitorch.layers.qconv_noact import QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct


def make_q_convolution(BaseClass: Type) -> Type:
    """creates a version with preactivation function of given baseclass

    Args:
        BaseClass (QConv-Subclass): The base class to add an activation layer to.

    Returns:
        Class: the activated version of the base class
    """
    class QConv(BaseClass):  # type: ignore
        def __init__(self,  # type: ignore
                     *args,  # type: ignore
                     input_quantization: str = None,
                     weight_quantization: str = None,
                     **kwargs) -> None:  # type: ignore
            """initialization function for quantization of inputs and weights.

            Args:
                input_quantization (str, optional): name of quantization function to apply on inputs before forwarding
                    through the qconvolution layer. Defaults to None.
                weight_quantization (str, optional): name of quantization function for weights. Defaults to None.
            """
            super(QConv, self).__init__(*args, weight_quantization=weight_quantization, **kwargs)
            self.activation = QActivation(input_quantization)

        def forward(self, input_tensor: Tensor) -> Tensor:
            """forward the input tensor through the activation and quantized convolution layer.

            Args:
                input_tensor (Tensor): input tensor

            Returns:
                Tensor: the activated and convoluted output tensor.
            """
            return super(QConv, self).forward(self.activation(input_tensor))

    return QConv


QConv1d = make_q_convolution(QConv1d_NoAct)
QConv2d = make_q_convolution(QConv2d_NoAct)
QConv3d = make_q_convolution(QConv3d_NoAct)