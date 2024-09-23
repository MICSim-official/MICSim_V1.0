import torch.nn as nn
import torch
from Accuracy.src.Layers.FPLayer.Conv2d import Conv2d
from Accuracy.src.Layers.FPLayer.Linear import Linear


import configparser
import os
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
Pretrained = config['Inference']['pretrained']
savedModel = config['Inference']['savedModel']

class VGG8(nn.Module):
    def __init__(self):
        super(VGG8, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.features = nn.Sequential(
            Conv2d(3, 128, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            Conv2d(128, 128, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(128, 256, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            Conv2d(256, 256, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            Conv2d(256, 512, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            Conv2d(512, 512, 3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            Linear(8192,1024,
                 bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            Linear(1024,10,
                 bias=False))

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1/3)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg8_load():
    model = VGG8()
    if Pretrained == 'True':
        print("load model: "+savedModel)
        model.load_state_dict(torch.load(savedModel))

    return model