import torch
import yaml
from torch import nn
import torch.nn.functional as F
from HyperPrams import get_parser
from utils.get_path import *

class HARCNN(nn.Module):
    def __init__(
        self,
        conv_kernel_size=(1, 7),
        pool_kernel_size=(1, 2),
    ):
        super().__init__()
        args = get_parser()
        config = yaml.load(
            open(get_config_path()),
            Loader=yaml.FullLoader,
        )
        in_channels = config["in_channels"]
        dim = config["dim"]
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
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size, num_sensors, num_axes, seq_len = x.shape
        x = x.reshape(batch_size, num_sensors*num_axes, 1, seq_len)
        # noise = torch.randn_like(x)
        # x = x + noise
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class MultiHARCNN(nn.Module):
    def __init__(
        self,
        channels=3,
        dim_hidden=64*26,
        num_classes=6,
        conv_kernel_size=(1, 9), 
        pool_kernel_size=(1, 2)
    ):
        super().__init__()


        # 设定每个传感器的卷积处理
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 512),
            nn.ReLU(), 
            nn.Linear(512, num_classes)
        )

        # 全连接层
        # 根据传感器数量调整全连接层的输入维度
        self.fc1 = nn.Linear(dim_hidden, 512)  # 每个传感器输出的特征数量乘以传感器数量
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, num_sensors, num_axes, seq_len = x.shape


        # 分别处理每个传感器的数据
        sensor_outputs = []
        for i in range(num_sensors):
            sensor_data = x[:, i, :, :]  # 获取第i个传感器的数据，形状为 (batch_size, 3, 200)
            sensor_data = sensor_data.unsqueeze(2)  # 增加通道维度 (batch_size, 1, 3, 200)

            # 通过卷积层处理每个传感器的数据
            conv_out = self.conv1(sensor_data)
            conv_out = self.conv2(conv_out)

            # 将每个传感器的输出保存
            sensor_outputs.append(conv_out)

        # 合并所有传感器的输出 (batch_size, 64, num_sensors, reduced_length)
        combined_output = torch.cat(sensor_outputs, dim=2)  # 沿着传感器维度拼接

        # 将每个传感器的输出展平成 (batch_size, 64 * num_sensors, reduced_length)
        combined_output = combined_output.view(batch_size, -1)  # 展平除时间维度外的其他维度

        # 全连接层
        x = torch.relu(self.fc1(combined_output))
        if 'dropout' in self.args.note:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

