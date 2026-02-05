import copy

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import yaml
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, HarData
from utils.model import *
import pandas as pd
from HyperPrams import get_parser
from os import path
from utils.get_path import *
import seaborn as sns
import matplotlib.pyplot as plt
from utils.privacy import initialize_dp

sns.set_style("darkgrid")


class Client:
    """
    联邦学习中的客户端基类。
    负责本地数据加载、模型训练、权重更新和测试。
    """

    def __init__(self, id):
        self.args = get_parser()
        self.device = self.args.device
        # 初始化模型：CNN 或 ResNet
        if self.args.model_type == "cnn":
            self.model = HARCNN().to(self.device)
        else:
            self.model = HARRESNET().to(self.device)
        self.id = id  # 客户端ID
        self.config = yaml.load(
            open(get_config_path()),
            Loader=yaml.FullLoader,
        )

        self.learning_rate = self.config["rate"]
        self.loss = nn.CrossEntropyLoss().to(self.device)
        # 根据参数选择优化器：SGD 或 带L2正则的SGD
        if self.args.l2 == 1 and 'l2' in self.args.note:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
            )  # momentum=0.9, weight_decay=1e-4
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate
            )
        # 学习率调度器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.99
        )
        self.train_loader = None
        self.privacy_engine = None
        # 参与训练的次数
        self.p_num = 0
        # 初始化全局模型副本（用于FedProx等算法的对比）
        if self.args.model_type == "cnn":
            self.global_model = HARCNN().to(self.device)
        else:
            self.global_model = HARRESNET().to(self.device)


    # 加载客户端本地数据
    def load_data(self, batch_size=None, data_type=None):
        data = read_client_data(self.id, data_type)
        return DataLoader(data, batch_size, drop_last=True, shuffle=True)

    # 更新客户端模型权重为服务器下发的全局权重
    def update_weight(self, model):
        if 'dp' in self.args.note:
            self.model._module.load_state_dict(copy.deepcopy(model.state_dict()))
        else:
            self.model.load_state_dict(copy.deepcopy(model.state_dict()))
        if 'dp' in self.args.note:
            self.global_model._module.load_state_dict(copy.deepcopy(model.state_dict()))
        else:
            self.global_model.load_state_dict(copy.deepcopy(model.state_dict()))

    # 初始化差分隐私 (Differential Privacy)
    def init_dp(self, dp_sigma=1.0):
        """
        使用 Opacus 初始化差分隐私引擎
        dp_sigma: 噪声乘数
        """
        optimizer_kwargs = {}
        if self.args.l2 == 1 and 'l2' in self.args.note:
            optimizer_kwargs = {"lr": self.learning_rate, "momentum": 0.9, "weight_decay": 1e-4}
        else:
            optimizer_kwargs = {"lr": self.learning_rate}
        self.train_loader = self.load_data(self.config["train_batch_size"], "train")
        # 将模型、优化器和数据加载器包装在 PrivacyEngine 中
        self.model, self.optimizer, self.train_loader, self.privacy_engine = initialize_dp(
            model=self.model, 
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs=optimizer_kwargs, 
            data_loader=self.train_loader, 
            dp_sigma=dp_sigma
        )
        self.global_model = copy.deepcopy(self.model)

    # 客户端本地训练
    def train_client(self):
        self.p_num += 1
        epochs = self.config["client_epochs"]
        loss_arr = []
        # trainloader = self.train_loader
        
        # 每次训练前重新加载数据
        self.train_loader = self.load_data(self.config["train_batch_size"], "train")

        self.model.train()
        for i in range(epochs):

            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                proximal_term = 0.0
                # 如果使用 FedProx 算法，计算近端项 (proximal term)
                if self.args.algorithm == "prox":
                    for w, w_t in zip(self.model.parameters(), self.global_model.parameters()):
                            proximal_term += (w - w_t).norm(2)
                        
                    l = self.loss(output, y) + (self.args.mu / 2) * proximal_term
                else:
                    l = self.loss(output, y)
                loss_arr.append(l.item())
                l.backward()
                self.optimizer.step()
        return np.mean(loss_arr)

    # 客户端本地测试
    def test_client(self, data_type="test"):
        self.model.eval()

        all_preds = []
        all_targets = []

        testloaderfull = self.load_data(
            self.config["test_batch_size"], data_type
        )

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        if len(all_targets) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # =========================
        # 1. Accuracy
        # =========================
        acc = accuracy_score(all_targets, all_preds)

        # =========================
        # 2. Recall
        # =========================
        macro_recall = recall_score(
            all_targets, all_preds, average="macro", zero_division=0
        )

        weighted_recall = recall_score(
            all_targets, all_preds, average="weighted", zero_division=0
        )

        # =========================
        # 3. Macro FPR (One-vs-Rest)
        # =========================
        cm = confusion_matrix(
            all_targets,
            all_preds,
            labels=list(range(self.config["num_classes"]))
        )

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        with np.errstate(divide="ignore", invalid="ignore"):
            fpr_per_class = FP / (FP + TN)

        fpr_per_class = np.nan_to_num(fpr_per_class)
        macro_fpr = np.mean(fpr_per_class)

        return acc, macro_recall, weighted_recall, macro_fpr
        
