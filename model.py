import torch
import torch.nn as nn

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
    # until here is darknet-53
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
        self.leaky = nn.LeakyReLU(0.1, inplace=False)
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


# The residual block is a combination of two convolutional blocks (CNNBlock)
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        # nn.ModuleList can be useful in cases where a custom or non-sequential
        # architecture is required, such as when using skip connections or residual
        # connections in a neural network.
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers = self.layers + [
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


# The last predefined block we will use is the ScalePrediction which is the last two
# convolutional layers leading up to the prediction for each scale. 

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                # it is 6 + num_classes because we added distance to model's prediction
                # 0th index is objectness, 1st to 4th is bounding box infor, 5th is dist
                2 * in_channels, (num_classes + 6) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            # 3 represents anchors_per_scale
            .reshape(x.shape[0], 3, self.num_classes + 6, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)  # [[0.3, 0.5, 0.4, 0.2, 0.7, 0.8, 0.9], bb, c, d]
        )


# Putting it all together to YOLOv3
# We will now put it all together to the YOLOv3 model for the detection task and distance estimation. 
# Most of the action takes place in the _create_conv_layers function where
# we build the model using the blocks defined above. Essentially we will just
# loop through the config list that we created above and add the blocks defined
# above in the correct order.

class YOLOv3(nn.Module):
    # number of classes is based on KITTI that is 7
    def __init__(self, in_channels=3, num_classes=7):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    # The forward function is the method that defines the forward pass of a PyTorch neural network.
    # When we call the model object with an input tensor (x in this case),
    # PyTorch automatically calls the forward function of the model and passes the input tensor as an argument.
    # The output tensor out is the result of applying the forward function of the YOLOv3 model to the input tensor x.
    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                # append its output to a list and later on compute the loss for each of the predictions separetely.
                output = layer(x)
                outputs.append(torch.sigmoid(output))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                # Keep track of the layers that are routed forward
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                # When we encounter an upsamling layer we will concatenate
                # the output with the last route previously found
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            # tuple (filters, kernel size, stride)  is for CNNBlock
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            # list is for Residual Block
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(
                        in_channels,
                        num_repeats=num_repeats,
                    )
                )

            # string is either for Scale Prediction block or
            # Upsampling the feature map and concatenating with a previous layer. 
            elif isinstance(module, str):
                if module == "S":
                    layers = layers + [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(
                        nn.Upsample(scale_factor=2),
                    )
                    in_channels = in_channels * 3

        return layers
