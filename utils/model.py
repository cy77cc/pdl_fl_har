import torch
import yaml
from torch import nn
import torch.nn.functional as F
from HyperPrams import get_parser
from utils.get_path import *
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 11), bias=False, padding=(0, 7))
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1, 7), bias=False, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HARRESNET(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(HARRESNET, self).__init__()

        args = get_parser()
        self.args = args
        config = yaml.load(
            open(get_config_path()),
            Loader=yaml.FullLoader,
        )
        in_channels = config["in_channels"]
        dim = 12416
        if args.dataset == "har":
            dim = 7808
        num_classes = config["num_classes"]

        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 7), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=(1, 7))
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=(1, 5))
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=(1, 3))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, num_sensors, num_axes, seq_len = x.shape
        x = x.reshape(batch_size, num_sensors*num_axes, 1, seq_len)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=(1, 2))
        out = out.view(out.size(0), -1)
        if 'dropout' in self.args.note:
            out = self.dropout(out)
        out = self.linear(out)
        return out

class HARCNN(nn.Module):
    def __init__(
        self,
        conv_kernel_size=(1, 7),
        pool_kernel_size=(1, 2),
    ):
        super().__init__()
        args = get_parser()
        self.args = args
        config = yaml.load(
            open(get_config_path()),
            Loader=yaml.FullLoader,
        )
        in_channels = config["in_channels"]
        dim = 2880
        if args.dataset == "har":
            dim = 1728
        num_classes = config["num_classes"]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(self.args.dropout_rate)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size, num_sensors, num_axes, seq_len = x.shape
        x = x.reshape(batch_size, num_sensors*num_axes, 1, seq_len)
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        if 'dropout' in self.args.note:
            out = self.dropout(out)
        out = self.fc2(out)
        if 'dropout' in self.args.note:
            out = self.dropout(out)
        out = self.fc(out)
        return out