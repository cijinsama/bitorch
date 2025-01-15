import typing
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.autograd.function import Function

from bitorch.quantizations import Quantization
from .config import config


class GradientCancellation(Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], torch.tensor(inputs[1], device=inputs[0].device))

    @staticmethod
    @typing.no_type_check
    def forward(
        input_tensor: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        return input_tensor.view_as(input_tensor)

    @staticmethod
    @typing.no_type_check
    def backward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        output_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Apply straight through estimator.

        This passes the output gradient towards the input if the inputs are in the range [-1, 1].

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient (= the masked output gradient)
        """
        input_tensor, threshold = ctx.saved_tensors
        cancelled = torch.where(
            torch.abs(input_tensor) <= threshold, output_grad, torch.tensor(0.0, device=output_grad.device)
        )
        return cancelled, None

    @staticmethod
    def vmap(info, in_dims, input_tensor, threshold):
        """
        Define behavior of the autograd function under vmap.

        Args:
            info: Information about vmap (e.g., batch_size).
            in_dims: Tuple specifying the batch dimension for each input.
            input_tensor: Batched input tensor (with batch dimension at in_dims[0]).
            threshold: Scalar threshold (or batched threshold, depending on in_dims[1]).

        Returns:
            Tuple[output, out_dims]: Output tensor and its batch dimension.
        """
        # Ensure input_tensor has a batch dimension
        input_batch_dim = in_dims[0]
        if input_batch_dim is not None and input_batch_dim != 0:
            input_tensor = input_tensor.movedim(input_batch_dim, 0)

        # Ensure threshold has a batch dimension if provided
        if in_dims[1] is not None:
            threshold = threshold.movedim(in_dims[1], 0)

        # Forward pass is a view operation, so return the input directly
        output = input_tensor.view_as(input_tensor)
        out_dims = 0  # Output has batch dimension at index 0

        return output, out_dims

class QActivation(nn.Module):
    """Activation layer for quantization"""

    def __init__(
        self,
        activation: Optional[Union[str, Quantization]] = None,
        gradient_cancellation_threshold: Optional[float] = 0.0,
    ) -> None:
        """initialization function for fetching suitable activation function.

        Args:
            activation (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.
            gradient_cancellation_threshold (Optional[float], optional): threshold for input gradient
                cancellation. Disabled if threshold is 0.
        """
        super(QActivation, self).__init__()
        self.activation_function = config.get_quantization_function(activation or config.input_quantization)
        self.gradient_cancellation_threshold = gradient_cancellation_threshold or config.gradient_cancellation_threshold

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forwards input tensor through activation function.

        Args:
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized input tensor.
        """
        if self.gradient_cancellation_threshold > 0:
            input_tensor = GradientCancellation.apply(input_tensor, self.gradient_cancellation_threshold)
        return self.activation_function(input_tensor)
