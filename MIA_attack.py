from os import path
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_utils import HarData
from utils.get_path import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore")

def MIA_member_data(shadow_model, in_data, args):
    """
    处理成员数据（参与训练的数据），生成攻击模型的输入特征。
    shadow_model: 影子模型（目标模型）
    in_data: 成员数据样本
    """
    shadow_model.eval()
    
    # 构造标签，成员数据标签为 1
    in_data, in_label = torch.from_numpy(in_data), torch.ones(size=(in_data.shape[0],))
    in_data = torch.tensor(in_data, dtype=torch.float32)
    InData = HarData(in_data, in_label)
    Inloader = DataLoader(InData, batch_size=16)

    attack_x = []
    attack_y = []
    for i, (x, y) in enumerate(tqdm(Inloader)):
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            pred = F.softmax(shadow_model(x))
            # 降序排序，输出预测
            pred, _ = torch.sort(pred, descending=True)
            # 取前5个概率最大的预测值作为特征
            pred = pred[:,:5]
        """
            pred是预测最大的三个 (注释说三个但代码是5)
            y是全1
        """
        attack_x.append(pred.detach())
        attack_y.append(y.detach())

    tensor_x = torch.cat(attack_x)
    tensor_y = torch.cat(attack_y)

    # return attackloader, attacktester
    data = tensor_x.detach().cpu().numpy()
    target = tensor_y.detach().cpu().numpy()

    return data, target

def MIA_nomember_data(shadow_model, out_data, args):
    """
    处理非成员数据（未参与训练的数据），生成攻击模型的输入特征。
    shadow_model: 影子模型（目标模型）
    out_data: 非成员数据样本
    """
    # very important！！！

    shadow_model.eval()

    """
        in_label 全1 标记为训练数据
        out_label 全0 标记为非训练数据
    """
    
    # 构造标签，非成员数据标签为 0
    out_data, out_label = torch.from_numpy(out_data), torch.zeros(
        size=(out_data.shape[0],)
    )
    
    out_data = torch.tensor(out_data, dtype=torch.float32)
    
    OutData = HarData(out_data, out_label)

    
    Outloader = DataLoader(OutData, batch_size=16)

    # 使用Din训练shadow_model (这里的注释可能不准确，实际上是用shadow_model预测OutData)
    attack_x = []
    attack_y = []

    # Dout
    for i, (x, y) in enumerate(tqdm(Outloader)):
        x = x.to(args.device)
        y = y.to(args.device)
        with torch.no_grad():
            pred = F.softmax(shadow_model(x))
            # 同样取前5个最大概率
            pred, _ = torch.sort(pred, descending=True) 
            pred = pred[:,:5]  

        attack_x.append(pred.detach())
        attack_y.append(y.detach())

    tensor_x = torch.cat(attack_x)
    tensor_y = torch.cat(attack_y)

    # return attackloader, attacktester
    data = tensor_x.detach().cpu().numpy()
    target = tensor_y.detach().cpu().numpy()

    return data, target


def train_attacker(data_x, data_y):
    """
    训练攻击者模型 (XGBoost Classifier)
    data_x: 攻击特征 (预测概率)
    data_y: 成员/非成员标签 (1/0)
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    attack_model = XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        # colsample_bytree= 0.6, gamma= 0.1, learning_rate= 0.1, max_depth= 3, n_estimators= 200, subsample= 0.6
        gamma=0.1, 
        learning_rate=0.1, 
        max_depth=3, 
        n_estimators=400, 
        n_jobs=2, 
        # reg_lambda=0.3,
    )


    if path.exists(path.join(model_save_path(), "attack_model.json")):
        attack_model.load_model(
            path.join(model_save_path(), "attack_model.json")
        )

    # attack_model = XGBClassifier()
    # 拟合攻击模型
    attack_model.fit(X_train, y_train)

    # 保存攻击模型
    attack_model.save_model(
        path.join(model_save_path(), "attack_model.json")
    )

    print("\n")
    print(
        "MIA Attacker training accuracy: {}".format(
            accuracy_score(y_train, attack_model.predict(X_train))
        )
    )
    print(
        "MIA Attacker testing accuracy: {}".format(
            accuracy_score(y_test, attack_model.predict(X_test))
        )
    )
    print("\n")



def test_attacker(data_x, data_y):
    """
    测试攻击者模型性能
    """
    attack_model = XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        gamma=0.1, 
        learning_rate=0.1, 
        max_depth=3, 
        n_estimators=400, 
        n_jobs=2, 
        # reg_lambda=0.3,
    )
    # attack_model = XGBClassifier(n_jobs=4, objective='binary:logistic', booster="gbtree")
    # 加载已训练的攻击模型
    attack_model.load_model(
        path.join(model_save_path(), "attack_model.json")
    )

    # print(classification_report(data_y, attack_model.predict(data_x)))

    predict_y = attack_model.predict(data_x)

    # 计算准确率和召回率
    accuracy = accuracy_score(data_y, predict_y)
    # print(f'MIA accuracy: {accuracy}')
    recall = recall_score(data_y, predict_y)
    fpr, tpr, thresholds = roc_curve(data_y, predict_y)
    return accuracy, recall, fpr, tpr, thresholds

