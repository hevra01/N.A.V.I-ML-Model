# this file includes the model architecture

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

""" 
Information about architecture config:
Tuple is structured by and signifies a convolutional block (filters, kernel_size, stride) 
Every convolutional layer is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats. 
"S" is for a scale prediction block and computing the yolo loss
"U" is for upsampling the feature map
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

# the most common building blocks of the architecture
# will be implemented as separate classes to avoid repeating code


# each tuple signifies a convolutional block with
# batch normalization and leaky relu added to it.
# batch normalization is making the value of the inputs
# of the layers have mean of 0 and variance of 1: helps
# with stability and speed of training.
# relu is rectified linear unit activation function.
# nn.module is the base class for all NN.
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        # batch normalization and activation function are skipped for the last layer
        # because the choice depends on the task being performed.
        # For example, in binary classification tasks, a sigmoid activation function
        # is often used to squash the output between 0 and 1. In multi-class classification
        # tasks, a softmax activation function is often used to produce a probability distribution over the classes.
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)



class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        # nn.ModuleList can be useful in cases where a custom or non-sequential
        # architecture is required, such as when using skip connections or residual
        # connections in a neural network.
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

            self.use_residual = use_residual
            self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x

        return x


class YOLOv5(nn.Module):
    # the number of classes will change for our case
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

