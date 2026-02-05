from HyperPrams import get_parser
import yaml
import os
from os import path

# 获取根目录路径
ROOT_PATH = "/home/zhangdongping/project/fl_sia_har"


def get_model_path():
    return path.join(ROOT_PATH, "result")


def get_config_path():
    dataset = get_parser().dataset
    return path.join(ROOT_PATH, "config", f"{dataset}.yaml")


def get_data_path():
    dataset = get_parser().dataset
    return path.join(ROOT_PATH, "npz_data", f"{dataset}")


# 获取数据集结果路径
def model_save_path():
    args = get_parser()
    dataset = args.dataset
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    return path.join(
        get_model_path(),
        dataset,
        f"{config['server_epochs']}_{config['client_epochs']}_"
        + f"{config['train_batch_size']}_"
        + f"{config['rate']}_{config['frac']}_{args.model_type}_{args.algorithm}_{args.note}",
    )


def graph_save_path():
    args = get_parser()
    dataset = args.dataset
    config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
    return path.join(
        get_model_path(),
        dataset,
        f"{config['server_epochs']}_{config['client_epochs']}_"
        + f"{config['train_batch_size']}_"
        + f"{config['rate']}_{config['frac']}_{args.model_type}_{args.algorithm}_{args.note}",
    )
