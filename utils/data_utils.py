import numpy as np
import torch
import yaml

from HyperPrams import get_parser
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.get_path import *
from os import path
import random


class HarData(Dataset):
    # x数据 y标签
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 加载数据集
def read_client_data(idx, data_type="train"):
    args = get_parser()
    data_path = get_data_path()
    if data_type == "train":
        x = np.load(f"{data_path}/{idx}/train.npz")["x"]
        y = np.load(f"{data_path}/{idx}/train.npz")["y"]
        return HarData(x, y)
    elif data_type == "test":
        x = np.load(f"{data_path}/{idx}/test.npz")["x"]
        y = np.load(f"{data_path}/{idx}/test.npz")["y"]
        return HarData(x, y)
    

def data_split():
    args = get_parser()
    data_path = get_data_path()
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    # 攻击模型认识的数据的所属客户端
    # print(config)
    malicious_client = config["malicious_client"]
    # 测试攻击模型的客户端
    attack_client = config['attack_client']
    attack_x = np.load(f"{data_path}/{attack_client}/data.npz")["x"]
    _, attack_test = train_test_split(attack_x, test_size=0.4, random_state=42)
    np.savez(f"{data_path}/client_test.npz", x=attack_test)
    # return
    for i in range(config["clients"]):
        x = np.load(f"{data_path}/{i}/data.npz")["x"]
        y = np.load(f"{data_path}/{i}/data.npz")["y"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        np.savez(f"{data_path}/{i}/train.npz", x=x_train, y=y_train)
        np.savez(f"{data_path}/{i}/test.npz", x=x_test, y=y_test)


    x= np.load(f"{data_path}/{malicious_client}/train.npz")['x']
    y = np.load(f"{data_path}/{malicious_client}/train.npz")['y']
    train_x, third_x, train_y, third_y = train_test_split(x, y, test_size=0.3)
    # np.savez(f"./harth/14/train.npz", x=train_x, y=train_y)
    np.savez(f"{data_path}/{malicious_client}/strain.npz", x=third_x, y=third_y)

    x = np.load(f"{data_path}/{malicious_client}/train.npz")['x']
    y = np.load(f"{data_path}/{malicious_client}/train.npz")['y']
    train_x, third_x, train_y, third_y = train_test_split(x, y, test_size=0.2)
    np.savez(f"{data_path}/{malicious_client}/train.npz", x=train_x, y=train_y)
    np.savez(f"{data_path}/{malicious_client}/nomember.npz", x=third_x, y=third_y)

    # MIA的训练数据
    attack_x = np.load(f"{data_path}/{malicious_client}/data.npz")["x"]

    client_list = [num for num in range(config['clients']) if num != malicious_client]


    # choice_client = np.random.choice(client_list, 1, replace=False)
    choice_client = np.random.choice(client_list, 2, replace=False)

    choice_x1 = np.load(f"{data_path}/{choice_client[0]}/data.npz")["x"]
    choice_x2 = np.load(f"{data_path}/{choice_client[1]}/data.npz")["x"]

    # choice_x1 += np.random.normal(0, 1, choice_x1.shape)
    # choice_x2 += np.random.normal(0, 1, choice_x2.shape)
    # attack_x += np.random.normal(0, 1, attack_x.shape)

    # client j 的数据
    attack_x_len = attack_x.shape[0]
    choice_x1_len = choice_x1.shape[0]
    choice_x2_len = choice_x2.shape[0]
    num_samples = int(min(attack_x_len * 0.4, choice_x1_len * 0.4, choice_x2_len * 0.4))
    num_samples = int(min(attack_x_len * 0.4, choice_x1_len * 0.4))
    in_data = np.random.permutation(attack_x)[:num_samples]
    out_data1 = np.random.permutation(choice_x1)[:num_samples//2]
    out_data2 = np.random.permutation(choice_x2)[:num_samples//2]

    intrain, intest = train_test_split(in_data, test_size=0.3, random_state=42)
    outtrain1, outtest1 = train_test_split(out_data1, test_size=0.3, random_state=42)
    outtrain2, outtest2 = train_test_split(out_data2, test_size=0.3, random_state=42)

    np.savez(f"{data_path}/attack_train.npz", intrain=intrain, outtrain=np.concatenate((outtrain1, outtrain2)))
    np.savez(f"{data_path}/attack_test.npz", intest=intest, outtest=np.concatenate((outtest1, outtest2)))
    # np.savez(f"{data_path}/attack_train.npz", intrain=intrain, outtrain=outtrain1)
    # np.savez(f"{data_path}/attack_test.npz", intest=intest, outtest=outtest1)


    
    


    