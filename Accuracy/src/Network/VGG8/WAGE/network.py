import configparser
import os
import torch
import torch.nn as nn

config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Pretrained = config['Inference']['pretrained']
savedModel = config['Inference']['savedModel']
quantization_mode  =     config['Quantization']['mode']


from  Accuracy.src.Layers.QLayer.CNN.QConv2d import QConv2d
from  Accuracy.src.Layers.QLayer.CNN.QLinear import QLinear

class VGG8(nn.Module):
    def __init__(self):
        super(VGG8, self).__init__()
        self.features = nn.Sequential(
            QConv2d(3, 128, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=False, quantize_error=True, name="layer1"),
            nn.ReLU(inplace=True),
            QConv2d(128, 128, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=True,  quantize_error=True, name="layer2"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QConv2d(128, 256, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=True,  quantize_error=True, name="layer3"),
            nn.ReLU(inplace=True),
            QConv2d(256, 256, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=True,  quantize_error=True, name="layer4"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            QConv2d(256, 512, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=True, quantize_error=True, name="layer5"),
            nn.ReLU(inplace=True),
            QConv2d(512, 512, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False,
                 quantize_weight=True, quantize_input=True, quantize_error=True, name="layer6"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            QLinear(8192,1024,
                    quantize_weight=True, quantize_input=True, quantize_error=True, name="layer7"),
            nn.ReLU(inplace=True),
            QLinear(1024,10,
                    quantize_weight=True, quantize_input=True, quantize_error=True, name="layer8"))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
   


def vgg8_load():
    model = VGG8()
    print("Load VGG8 WAGE network")
    if Pretrained == 'True':
        print("load model: "+savedModel)
        model.load_state_dict(torch.load(savedModel))

    return model