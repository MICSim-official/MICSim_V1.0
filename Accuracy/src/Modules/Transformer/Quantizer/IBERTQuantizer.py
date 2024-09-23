import torch
import torch.nn as nn
import decimal
import numpy as np 
from torch.autograd import Function
from Accuracy.src.Modules.CNN.Quantizer.Quantizer import Quantizer
import configparser
import os
from Accuracy.src.utils import make_path

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Logdir = make_path.makepath_logdir()
bias_percision = int(config['Quantization']['biasprecision'])



class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):
        """
        Args:
            x (`torch.Tensor`):
                Floating point tensor to be quantized.
            k (`int`):
                Quantization bitwidth.
            percentile_mode (`bool`):
                Whether or not to use percentile calibration.
            scale (`torch.Tensor`):
                Pre-calculated scaling factor for *x*. Note that the current implementation of SymmetricQuantFunction
                requires pre-calculated scaling factor.

        Returns:
            `torch.Tensor`: Symmetric-quantized value of *input*.
        """
        zero_point = torch.tensor(0.0).to(scale.device)

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.

    Args:
        input (`torch.Tensor`):
            Single-precision input tensor to be quantized.
        scale (`torch.Tensor`):
            Scaling factor for quantization.
        zero_pint (`torch.Tensor`):
            Shift for quantization.
        inplace (`bool`, *optional*, defaults to `False`):
            Whether to compute inplace or not.

    Returns:
        `torch.Tensor`: Linearly quantized value of *input* according to *scale* and *zero_point*.
    """
    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # quantized = float / scale + zero_point
    if inplace:
        input.mul_(1.0 / scale).add_(zero_point).round_()
        return input
    return torch.round(1.0 / scale * input + zero_point)

    
class IBERTQuantizer(Quantizer):
    def __init__(self):
        super(IBERTQuantizer, self).__init__()
        self.percentile_mode = False
        self.bias_bits = bias_percision
        # self.per_channel = per_channel

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        return scale 
    
    def update_range(self, input):
        pass
    
    def input_clamp(self, input):
        return input 
                      
    def QuantizeWeight(self, weight, weight_bits=None, Wsigned=True, per_channel=None):
        if weight_bits is  None:
            weight_bits = self.weight_precision
        
        w = weight
        w_transform = w.data.detach()
        
        if per_channel == True:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            per_channel = False
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
            
            # w_min = w_transform.min().expand(1)
            # w_max = w_transform.max().expand(1)
        self.weightscale = get_scaling_factor(weight_bits, w_min, w_max, per_channel)
        Qweight = SymmetricQuantFunction.apply(weight, weight_bits, self.percentile_mode, self.weightscale)
        weightrange = []
        weightshift = 0.0
        
        if Wsigned == False:
            Qweight += 2 ** (weight_bits - 1) - 1
            weightshift  = -(2 ** (weight_bits - 1) - 1)
         
        return Qweight, self.weightscale, weightrange, weightshift 

    def QuantizeBias(self, bias, bias_scale):
        Qbias = SymmetricQuantFunction.apply(bias, self.bias_bits, False, bias_scale) 
        return Qbias
    
    def QuantizeInput(self, input, inputscale, bits=None, Isigned=True):
        Qinput = SymmetricQuantFunction.apply(input, self.input_precision, False, inputscale) 
        inputrange = []
        inputshift = 0.0
        return Qinput, inputscale, inputrange, inputshift
    
    def QuantizeError(self, error, bits=None, Esigned=True):
        pass

    def quantize_grad(self, x): 
        raise NotImplementedError("use QSGD")
    
def get_scaling_factor( num_bits, saturation_min, saturation_max, per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.

    Args:
        saturation_min (`torch.Tensor`):
            Lower bound for quantization range.
        saturation_max (`torch.Tensor`):
            Upper bound for quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.

    Returns:
        `torch.Tensor`: Scaling factor that linearly quantizes the given range between *saturation_min* and
        *saturation_max*.
    """
    # in this part, we do not need any gradient computation,
    # in order to enforce this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n
    return scale


class QuantAct(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit, need_Oint=False,act_range_momentum=0.95, per_channel=False, channel_len=None, quant_mode=False):
        super().__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.percentile = False
        self.need_output_int = need_Oint
        self.act_function = SymmetricQuantFunction.apply

        if not self.per_channel:
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
            self.x_min -= 1e-5
            self.x_max += 1e-5
        else:
            raise NotImplementedError("per-channel mode is not currently supported for activation.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
            f"quant_mode: {self.quant_mode}, Act_min: {self.x_min.item():.2f}, "
            f"Act_max: {self.x_max.item():.2f})"
        )

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        specified_min=None,
        specified_max=None,
    ):
        x_act = x if identity is None else identity + x
        # collect running stats if training
        if self.training:
            assert not self.percentile, "percentile mode is not currently supported for activation."
            assert not self.per_channel, "per-channel mode is not currently supported for activation."
            x_min = x_act.data.min()
            x_max = x_act.data.max()

            assert (
                x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0
            ), "NaN detected when computing min/max of the activation"

            # Initialization
            if self.x_min.min() > -1.1e-5 and self.x_max.max() < 1.1e-5:
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        if not self.quant_mode:
            return x_act, None

        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = get_scaling_factor(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(x, self.activation_bit, self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = FixedPointMul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        correct_output_scale = self.act_scaling_factor.view(-1)
        
        if self.need_output_int:
            return quant_act_int, self.act_scaling_factor 
        else:
            return quant_act_int * correct_output_scale, self.act_scaling_factor

class FixedPointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on hardware.

    Args:
        pre_act (`torch.Tensor`):
            Input tensor.
        pre_act_scaling_factor (`torch.Tensor`):
            Scaling factor of the input tensor *pre_act*.
        bit_num (`int`):
            Quantization bitwidth.
        z_scaling_factor (`torch.Tensor`):
            Scaling factor of the output tensor.
        identity (`torch.Tensor`, *optional*):
            Identity tensor, if exists.
        identity_scaling_factor (`torch.Tensor`, *optional*):
            Scaling factor of the identity tensor *identity*, if exists.

    Returns:
        `torch.Tensor`: Output tensor(*pre_act* if *identity* is not given, otherwise the addition of *pre_act* and
        *identity*), whose scale is rescaled to *z_scaling_factor*.
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        z_scaling_factor,
        identity=None,
        identity_scaling_factor=None,
    ):
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x: x  # noqa: E731
        else:
            reshape = lambda x: x.view(1, 1, -1)  # noqa: E731
        ctx.identity = identity

        n = 2 ** (bit_num - 1) - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor

            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output

            return torch.clamp(output.type(torch.float), -n - 1, n)

    @staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None

def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Args:
        scaling_factor (`torch.Tensor`):
            Target scaling factor to decompose.

    Returns:
        ``Tuple(torch.Tensor, torch.Tensor)`: mantisa and exponent
    """

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            decimal.Decimal(m * (2**max_bit)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

    return (
        torch.from_numpy(output_m).to(inputs.device).view(shape_of_input),
        torch.from_numpy(output_e).to(inputs.device).view(shape_of_input),
    )    