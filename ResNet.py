import math

from torch import nn
import torch.nn.functional as F

# config
# ResNet18: [2, 2, 2, 2]
num_layers = [2, 2, 2]


class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class ResNet18_layer1(nn.Module):
    def __init__(self):
        super(ResNet18_layer1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class ResNet18_layer2(nn.Module):
    def __init__(self):
        super(ResNet18_layer2, self).__init__()
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        resudial1 = F.relu(x)
        out1 = self.layer2(resudial1)
        out1 = out1 + resudial1  # adding the resudial inputs -- downsampling not required in this layer
        resudial2 = F.relu(out1)
        return resudial2


class ResNet18_layer3(nn.Module):
    def __init__(self):
        super(ResNet18_layer3, self).__init__()
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        out1 = F.relu(x)
        out2 = self.layer3(out1)
        out2 = out2 + x
        x3 = F.relu(out2)
        return x3


class ResNet18_layer4(nn.Module):
    def __init__(self):
        super(ResNet18_layer4, self).__init__()
        self.layer4 = self._layer(Baseblock, 128, num_layers[0], stride=2)

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != 64 * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(64, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = [block(64, planes, stride=stride, dim_change=dim_change)]
        for i in range(1, num_layers):
            netLayers.append(block(planes, planes))
        return nn.Sequential(*netLayers)

    def forward(self, x):
        x4 = self.layer4(x)
        return x4


class ResNet18_layer5(nn.Module):
    def __init__(self):
        super(ResNet18_layer5, self).__init__()
        self.layer5 = self._layer(Baseblock, 256, num_layers[1], stride=2)

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != 128 * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(128, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = [block(128, planes, stride=stride, dim_change=dim_change)]
        for i in range(1, num_layers):
            netLayers.append(block(planes, planes))
        return nn.Sequential(*netLayers)

    def forward(self, x):
        x5 = self.layer5(x)
        return x5


class ResNet18_layer6(nn.Module):
    def __init__(self):
        super(ResNet18_layer6, self).__init__()
        self.layer6 = self._layer(Baseblock, 512, num_layers[2], stride=2)

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != 256 * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(256, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = [block(256, planes, stride=stride, dim_change=dim_change)]
        for i in range(1, num_layers):
            netLayers.append(block(planes, planes))
        return nn.Sequential(*netLayers)

    def forward(self, x):
        x6 = self.layer6(x)
        return x6


class ResNet18_layer7(nn.Module):
    def __init__(self, classes):
        super(ResNet18_layer7, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * Baseblock.expansion, classes)

    def forward(self, x):
        x7 = F.avg_pool2d(x, 2)
        x8 = x7.view(x7.size(0), -1)
        y_hat = self.fc(x8)
        return y_hat


# define client-side model
all_layers = [ResNet18_layer1, ResNet18_layer2, ResNet18_layer3, ResNet18_layer4, ResNet18_layer5, ResNet18_layer6,
              ResNet18_layer7]
