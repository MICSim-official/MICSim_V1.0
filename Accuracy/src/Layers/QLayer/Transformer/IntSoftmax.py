import torch
import torch.nn as nn
from Accuracy.src.Modules.Transformer.Quantizer.IBERTQuantizer import *

class IntSoftmax(nn.Module):
    """
    Quantized version of `torch.nn.Softmax`. Adds quantization-specific arguments on top of `torch.nn.Softmax`.

    Args:
        output_bit (`int`):
            Bitwidth for the layer output activation.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "softmax" or "nonlinear" is given.
    """

    def __init__(self, output_bit, quant_mode=False, need_Oint=False,force_dequant="none"):
        super().__init__()
        self.output_bit = output_bit
        self.max_bit = 32
        self.quant_mode = quant_mode
        self.need_output_int=need_Oint

        if force_dequant in ["nonlinear", "softmax"]:
            # logger.info("Force dequantize softmax")
            self.quant_mode = False

        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931  # -ln2
        self.const = 30  # dummy integer constant
        self.coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor**2)
        z = (x_int + b_int) * x_int + c_int
        scaling_factor = self.coef[0] * scaling_factor**2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.const * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.const - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**self.const
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        if not self.quant_mode:
            return nn.functional.softmax(x, dim=-1), None

        x_int = x / scaling_factor

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)

        # Avoid overflow
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor

        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = floor_ste.apply(2**self.max_bit / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (self.max_bit - self.output_bit))
        scaling_factor = 1 / 2**self.output_bit
        if self.need_output_int:
            return exp_int, scaling_factor 
        else:
            return exp_int * scaling_factor, scaling_factor