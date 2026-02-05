import os.path

import yaml

from server import *
from client import *
from utils.data_utils import *
from HyperPrams import get_parser
import numpy as np
import random
from utils.get_path import *


if __name__ == "__main__":
    # 获取命令行参数
    args = get_parser()
    # 设置随机种子，保证结果可复现
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载配置文件
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    epochs = config["server_epochs"]

    # 创建必要的目录用于保存模型和结果
    os.makedirs(get_model_path(), exist_ok=True)
    os.makedirs(model_save_path(), exist_ok=True)
    os.makedirs(os.path.join(model_save_path(), "save_models"), exist_ok=True)
    #
    # plt_result()
    args = get_parser()
    # if not os.path.exists(os.path.join(get_data_path(), f"client_test.npz")):
    # 划分数据集
    data_split()
    
    # 初始化服务器，指定目标模型类型（这里似乎是"target"但实际代码中可能未充分利用该字符串，需结合Server类查看）
    server = Server("target")
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)

    # 开始联邦学习训练过程
    server.train()
    # 绘制损失函数曲线
    server.plt_loss()
    # server.test_mia() # 测试成员推断攻击（MIA）

