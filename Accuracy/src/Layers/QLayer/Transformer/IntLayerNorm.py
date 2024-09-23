import torch
import torch.nn as nn
from Accuracy.src.Modules.Transformer.Quantizer.IBERTQuantizer import *

class IntLayerNorm(nn.Module):
    """
    Quantized version of `torch.nn.LayerNorm`. Adds quantization-specific arguments on top of `torch.nn.LayerNorm`.

    Args:
        output_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the layer output activation.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "layernorm" or "nonlinear" is given.
    """

    def __init__(self, normalized_shape, eps, output_bit=8, quant_mode=False, force_dequant="none"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            # logger.info("Force dequantize layernorm")
            self.quant_mode = False

        self.register_buffer("shift", torch.zeros(1))
        self.output_bit = output_bit
        self.max_bit = 32
        self.dim_sqrt = None
        self.activation = QuantAct(self.output_bit, quant_mode=self.quant_mode)

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int**2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**self.max_bit)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            # logger.info(f"Dynamic shift adjustment: {int(shift_old)} -> {int(self.shift)}")

    def overflow_fallback(self, y_int):
        """
        This fallback function is called when overflow is detected during training time, and adjusts the `self.shift`
        to avoid overflow in the subsequent runs.
        """
        self.set_shift(y_int)  # adjusts `self.shift`
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y**2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        # compute sqrt of the feature dimension if it is the first run
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # overflow handling in training time
        if self.training:
            # if overflow is detected
            if var_int.max() >= 2**self.max_bit:
                var_int = self.overflow_fallback(y_int)
                assert var_int.max() < 2**self.max_bit + 0.1, (
                    "Error detected in overflow handling: "
                    "`var_int` exceeds `self.max_bit` (the maximum possible bit width)"
                )

        # To be replaced with integer-sqrt kernel that produces the same output
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2**self.shift
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor