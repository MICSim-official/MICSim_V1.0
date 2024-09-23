import configparser
import os
import configparser
import torch
import torch.nn as nn
import numpy as np


from Accuracy.src.utils import make_path
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Logdir = make_path.makepath_logdir()

QuantizationMode = config['Quantization']['mode']
QuantizeEmbedding = config['Quantization']['embedding']

if QuantizeEmbedding == 'True':
    QuantIt = True
    if QuantizationMode == 'IBERT':
        from Accuracy.src.Modules.Transformer.Quantizer.IBERTQuantizer import IBERTQuantizer as Quantizer
else:
    QuantIt = False
    
    
class QEmbedding(nn.Module):
    """
    Quantized version of `torch.nn.Embedding`. Adds quantization-specific arguments on top of `torch.nn.Embedding`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        # weight_bit=8,
        momentum=0.95,
        # quant_mode=False,
    ):
        super().__init__()
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
        self.register_buffer("weight_scaling_factor", torch.zeros(1))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))

        self.Quantizer = Quantizer()
        # self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_it = QuantIt
        self.percentile_mode = False
        # self.weight_function = SymmetricQuantFunction.apply

    def forward(self, x, positions=None, incremental_state=None):
        if self.quant_it == False:
            return (
                nn.functional.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ),
                None,
            )

        self.weight_integer, self.weight_scaling_factor, weightrange, weightshift = self.Quantizer.QuantizeWeight(weight=self.weight) 

        emb_int = nn.functional.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return emb_int * self.weight_scaling_factor, self.weight_scaling_factor