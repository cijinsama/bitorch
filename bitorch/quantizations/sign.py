"""Sign Function Implementation"""

import typing

import torch

from .base import Quantization, STE


class SignFunction(STE):
    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx.save_for_backward(inputs[0], torch.tensor(inputs[1], device=inputs[0].device))
        pass

    @staticmethod
    @typing.no_type_check
    def forward(
        # ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Binarize the input tensor using the sign function.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the sign tensor
        """
        sign_tensor = torch.sign(input_tensor)
        sign_tensor = torch.where(sign_tensor == 0, torch.tensor(1.0, device=sign_tensor.device), sign_tensor)
        return sign_tensor

    @staticmethod
    def vmap(info, in_dims, input_tensor):
        """
        Vectorized implementation of the forward method for batched inputs.

        Args:
            info: Contains vmap-related information (batch_size, randomness).
            in_dims (tuple): Specifies which dimension of `input_tensor` is the batch dimension.
            input_tensor (torch.Tensor): Batched input tensor.

        Returns:
            Tuple[torch.Tensor, int]: The batched output tensor and the dimension of its batch.
        """
        # Ensure input_tensor has the batch dimension as the first dimension
        input_batch_dim = in_dims[0] if in_dims[0] is not None else 0
        if input_batch_dim != 0:
            input_tensor = input_tensor.movedim(input_batch_dim, 0)

        # Apply the sign and binarize operation across the batch
        sign_tensor = torch.sign(input_tensor)
        sign_tensor = torch.where(sign_tensor == 0, torch.tensor(1.0, device=sign_tensor.device), sign_tensor)

        # Output has batch dimension at index 0
        out_dims = 0
        return sign_tensor, out_dims


class Sign(Quantization):
    """Module for applying the sign function with straight through estimator in backward pass."""

    name = "sign"
    bit_width = 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SignFunction.apply(x)
