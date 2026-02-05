from utils.model import *
import copy
import time
import yaml
import numpy as np
from client import Client
from HyperPrams import get_parser
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
import seaborn as sns
from MIA_attack import *
from opacus.validators import ModuleValidator
# sns.set_theme()  
sns.set_style('whitegrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
# sns.set_palette("Purples")
# plt.figure()

class Server:
    def __init__(self, model_type):
        self.args = get_parser()
        self.device = self.args.device
        # 根据配置初始化模型：CNN 或 ResNet
        if self.args.model_type == "cnn":
            self.model = HARCNN().to(self.device)
        else:
            self.model = HARRESNET().to(self.device)

        print(self.model)
        if 'dp' in self.args.note:
            self.model = ModuleValidator.fix(self.model) # Opacus 隐私引擎修复（如果需要）
        self.config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
        self.global_loss = []
        self.model_type = model_type
        # 获取恶意客户端和攻击客户端的ID配置
        self.malicious_client = self.config['malicious_client']
        self.attack_client = self.config['attack_client']

    def fed_avg(self, w):
        """
        FedAvg 算法实现：对客户端上传的模型参数进行加权平均（此处为简单平均）。
        w: 包含所有被选中客户端模型状态字典的列表
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            # 计算平均值
            w_avg[key] = torch.div(w_avg[key], len(w))
        # 更新全局模型参数
        self.model.load_state_dict(copy.deepcopy(w_avg))

    def train(self):
        """
        服务器端的主训练循环
        """
        args = get_parser()
        clients = []
        select_clients = []
        df = pd.DataFrame(columns=['c_id', 'test_acc'])
        malicious_client_num = 0
        attack_client_num = 0

        # 初始化所有客户端
        for i in range(self.config["clients"]):
            client = Client(i)
            # 如果开启了差分隐私（DP），则初始化DP
            if 'dp' in self.args.note:
                client.init_dp()
            clients.append(client)

        # 全局训练轮次循环
        for epoch in tqdm(range(self.config["server_epochs"])):
            # print(f"========================{epoch}========================")
            local_loss = []
            # 随机选择客户端参与本轮训练
            m = 0
            if epoch == 0:
                # 第一轮可能选择所有客户端（根据配置）
                m = max((1 * self.config["clients"]), 1)
            else:
                # 后续轮次按比例随机选择
                m = max(int(self.config["frac"] * self.config["clients"]), 1)
            idxs_users = np.random.choice(
                range(self.config["clients"]), m, replace=False
            )
            idxs_users = idxs_users.tolist()
            # 确保恶意客户端在第一轮被选中（用于攻击目的？）
            if epoch == 0:
                idxs_users.append(self.malicious_client)
            select_clients.append(idxs_users)
            
            self.model.train()
            w = [] # 用于存储客户端的模型参数
            for i in idxs_users:
                # 将全局模型参数下发给客户端
                clients[i].update_weight(self.model)
                # 客户端本地训练，返回损失值
                loss = clients[i].train_client()

                # 创建保存模型的目录
                # os.makedirs(f"{model_save_path()}/save_models/{self.malicious_client}", exist_ok=True)
                os.makedirs(f"{model_save_path()}/save_models/{i}", exist_ok=True)
                os.makedirs(f"{model_save_path()}/save_models/{self.attack_client}", exist_ok=True)
                
                # 如果是恶意客户端，保存其模型参数（用于MIA攻击等分析）
                if i == self.config['malicious_client']:
                    malicious_client_num += 1
                    if 'dp' in self.args.note:
                        torch.save(clients[i].model._module.state_dict(), f"{model_save_path()}/save_models/{self.malicious_client}/client_{i}_{malicious_client_num}th_model.pth")
                    else:
                        torch.save(clients[i].model.state_dict(), f"{model_save_path()}/save_models/{self.malicious_client}/client_{i}_{malicious_client_num}th_model.pth")
                
                # 保存每个客户端在本轮训练后的模型
                # if i == self.config['attack_client']:
                #     attack_client_num += 1
                #     torch.save(clients[i].model.state_dict(), f"{model_save_path()}/save_models/{self.attack_client}/client_{i}_{attack_client_num}th_model.pth")
                # if not os.path.exists(f"{model_save_path()}/save_models/{i}/client_trained_model.pth"):
                if 'dp' in self.args.note:
                    torch.save(clients[i].model._module.state_dict(), f"{model_save_path()}/save_models/{i}/client_trained_model.pth")
                else:
                    torch.save(clients[i].model.state_dict(), f"{model_save_path()}/save_models/{i}/client_trained_model.pth")
                    
                
                # print("client %d selected num %d" % (i, clients[i].p_num))
                
                local_loss.append(loss)
                # 收集客户端模型参数用于聚合
                if 'dp' in self.args.note:
                    w.append(copy.deepcopy(clients[i].model._module.state_dict()))
                else:
                    w.append(copy.deepcopy(clients[i].model.state_dict()))
            
            # 服务器执行聚合操作 (FedAvg)
            self.fed_avg(w)
            self.global_loss.append(np.mean(local_loss))

        # 训练结束后，更新所有客户端的模型为最终全局模型（用于最终测试）
        for i in range(len(clients)):
            clients[i].update_weight(self.model)
            
        # 保存选中的客户端记录
        with open('array_data.json', 'w') as json_file:
            json.dump(select_clients, json_file)
        
        # 测试所有客户端的准确率
        for i in range(len(clients)):
            train_acc, train_macro_recall, train_weighted_recall, train_macro_fpr = clients[i].test_client("train")
            test_acc, test_macro_recall, test_weighted_recall, test_macro_fpr = clients[i].test_client("test")
            df = df._append({
                'c_id': str(i), 
                'train_acc': train_acc, 'train_macro_recall': train_macro_recall, 'train_weighted_recall': train_weighted_recall, 'train_macro_fpr': train_macro_fpr,
                'test_acc': test_acc, 'test_macro_recall': test_macro_recall, 'test_weighted_recall': test_weighted_recall, 'test_macro_fpr': test_macro_fpr
            }, ignore_index=True)

        # 保存测试结果并绘图
        df.to_csv(f'{model_save_path()}/test_result.csv', index=False, float_format='%.4f')
        plt.figure(figsize=(10, 5))
        # g = sns.lineplot(data=df, x='c_id', y='test_acc', markers=True, palette="Purples")
        plt.plot(df['c_id'], df['train_acc'], "+-", label="train accuracy")
        plt.plot(df['c_id'], df['test_acc'], "--", label="test accuracy")
        plt.legend()
        plt.savefig(f'{model_save_path()}/test_result.png', dpi=300, bbox_inches='tight')

        # 保存最终全局模型
        torch.save(self.model.state_dict(), f"{model_save_path()}/trained_model.pth")
        plt.close()

    def plt_loss(self):
        """
        绘制全局训练损失曲线
        """
        plt.figure(figsize=(4, 3))
        plt.plot(np.arange(0, self.config['server_epochs']), self.global_loss, label="loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"{model_save_path()}/{self.model_type}_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        df = pd.DataFrame({'loss': self.global_loss})
        df.to_csv(f"{model_save_path()}/{self.model_type}_loss.csv", index=True,)
    
        # 在客户端参与次数大于1，之后的每一轮都创建训练集

    # 数据在模型上的输出概率为构建攻击模型的数据
    def mia_attack(self, shadow_model):
        """
        执行成员推断攻击 (MIA)
        shadow_model: 影子模型，用于生成攻击模型的训练数据
        """
        # 加载非成员数据（未参与训练的数据）
        out_data = np.load(f"{get_data_path()}/{self.malicious_client}/nomember.npz")['x']
        # 加载成员数据（参与训练的数据）
        in_data = np.load(f"{get_data_path()}/{self.malicious_client}/train.npz")['x']
        # 确保成员和非成员数据量一致
        in_data = np.random.permutation(in_data)[:out_data.shape[0]]
        
        # 生成成员数据的攻击特征（Top-k 预测概率）
        data_in_x, data_in_y = MIA_member_data(shadow_model, in_data, self.args)
        # 生成非成员数据的攻击特征
        data_out_x, data_out_y = MIA_nomember_data(shadow_model, out_data, self.args)

        # 训练攻击者模型（XGBoost）
        train_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))


        # 测试攻击效果

        # out_data = np.load(f"{get_data_path()}/{self.malicious_client}/nomembertest.npz")['x']
        out_data = np.load(f"{get_data_path()}/{self.malicious_client}/nomember.npz")['x']
        in_data = np.load(f"{get_data_path()}/{self.malicious_client}/strain.npz")['x']
        in_data = np.random.permutation(in_data)[:out_data.shape[0]]
        # 成员数据
        data_in_x, data_in_y = MIA_member_data(shadow_model, in_data, self.args)
        # 非成员数据
        data_out_x, data_out_y = MIA_nomember_data(shadow_model, out_data, self.args)

        # 绘制预测向量分布图
        plt.figure()
        for i in range(data_in_x.shape[0]):
            plt.plot(range(0, len(data_in_x[i])), data_in_x[i], color="blue", alpha=0.5, linewidth=1)
            plt.plot(range(0, len(data_out_x[i])), data_out_x[i], color="red", alpha=0.5, linewidth=1)

        plt.savefig("./predict_vector.png", dpi=600, bbox_inches='tight')
        
        # 测试攻击模型准确率和召回率
        accuracy, recall = test_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y)))
        self.attack_accuracy.append(accuracy)
        self.recall_accuracy.append(recall)


    def test_mia(self):
        clients = []
        test_acc = []
        recall_acc = []
        base = []
        for i in range(self.config['clients']):
            c = Client(i)
            c.model.load_state_dict(self.model.state_dict())
            clients.append(c)
        
        for i in range(self.config['clients']):
            clients[i].train_client()
            out_data = np.load(f"{get_data_path()}/{self.malicious_client}/nomember.npz")['x']
            in_data = np.load(f"{get_data_path()}/{self.malicious_client}/strain.npz")['x']
            in_data = np.random.permutation(in_data)[:out_data.shape[0]]
            # 成员数据
            data_in_x, data_in_y = MIA_member_data(clients[i].model, in_data, self.args)
            # 非成员数据
            data_out_x, data_out_y = MIA_nomember_data(clients[i].model, out_data, self.args)
            
            accuracy, recall = test_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))

            test_acc.append(accuracy)
            recall_acc.append(recall)

        pd.DataFrame({'c_id': range(self.config['clients']), 'test_acc': test_acc, 'recall_acc': recall_acc}).to_csv(f'{model_save_path()}/mia_test_result.csv', index=False, float_format='%.4f')


