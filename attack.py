from MIA_attack import *
from utils.model import *
from HyperPrams import get_parser
import yaml
from utils.get_path import *
import os
import pandas as pd
from client import Client
from opacus.validators import ModuleValidator


args = get_parser()

config = yaml.load(open(get_config_path()), Loader=yaml.FullLoader)
# 获取恶意客户端和攻击客户端ID
malicious_client = config["malicious_client"]
attack_client = config["attack_client"]

device = args.device

dataset = args.dataset

# 初始化模型结构
if args.model_type == "cnn":
    model = HARCNN().to(device)
else:
    model = HARRESNET().to(device)

if 'dp' in args.note:
    model = ModuleValidator.fix(model)

# 获取恶意客户端保存的所有模型文件（不同轮次的模型）
files = os.listdir(f"{model_save_path()}/save_models/{malicious_client}/")

test_acc = []         
recall_acc = []
fpr_acc = []
tpr_acc = []
thresholds_arr = []

# 遍历所有保存的模型进行攻击测试
for i in range(1, len(files), 1):

    # 加载特定轮次的模型权重
    model.load_state_dict(torch.load(f"{model_save_path()}/save_models/{malicious_client}/client_{malicious_client}_{i}th_model.pth"))

    # 加载用于攻击的训练数据 (Shadow Dataset)
    in_train = np.load(f"{get_data_path()}/attack_train.npz")["intrain"]
    out_train = np.load(f"{get_data_path()}/attack_train.npz")["outtrain"]

    # 加载用于攻击的测试数据
    in_test = np.load(f"{get_data_path()}/attack_test.npz")["intest"]
    out_test = np.load(f"{get_data_path()}/attack_test.npz")["outtest"]

    # 准备训练攻击模型的数据 (成员和非成员)
    data_in_x, data_in_y = MIA_member_data(model, in_train, args)
    data_out_x, data_out_y = MIA_nomember_data(model, out_train, args)

    # 训练攻击模型
    train_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))

    # 准备测试攻击模型的数据
    data_in_x, data_in_y = MIA_member_data(model, in_test, args)
    data_out_x, data_out_y = MIA_nomember_data(model, out_test, args)

    # 测试攻击模型性能
    accuracy, recall, fpr, tpr, thresholds = test_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))

    test_acc.append(accuracy)
    recall_acc.append(recall)
    fpr_acc.append(fpr)
    tpr_acc.append(tpr)
    thresholds_arr.append(thresholds)

# 保存攻击结果
df = pd.DataFrame({'test_acc': test_acc, 'recall_acc': recall_acc, 'fpr_acc': fpr_acc, 'tpr_acc': tpr_acc, 'thresholds_arr': thresholds_arr})
df.to_csv(f'{model_save_path()}/attack_result.csv', index=True, float_format='%.4f')


test_acc = []       
# files = os.listdir(f"{model_save_path()}/save_models/{attack_client}/")
# 对所有客户端的最终模型进行攻击测试（模拟对系统中其他客户端的推断）
for i in range(config['clients']):
    # 加载每个客户端训练好的模型
    model.load_state_dict(torch.load(f"{model_save_path()}/save_models/{i}/client_trained_model.pth"))

    # 加载攻击目标数据
    attack_client_test = np.load(f"{get_data_path()}/{attack_client}/data.npz")['x']
    attack_client_test = np.random.permutation(attack_client_test)[:int(attack_client_test.shape[0]*0.4)]
    
    # 提取攻击特征
    data_x, data_y = MIA_member_data(model, attack_client_test, args)
    os.makedirs(f'save_vector/{dataset}/{i}', exist_ok=True)
    np.savez(f'save_vector/{dataset}/{i}/test.npz', x=data_x)

    # 测试攻击效果
    accuracy, recall, fpr, tpr, thresholds = test_attacker(data_x, data_y)
    test_acc.append(accuracy)

# 保存对各客户端的攻击准确率
df = pd.DataFrame({'accuracy': test_acc})
df.to_csv(f'{model_save_path()}/attack_client.csv', index=True, float_format='%.4f')


# c13 = Client(2)
# c13.model.load_state_dict(torch.load(f"result/harth/100_25_256_0.007_0.3_cnn_avg_mix2/trained_model.pth"))

# model.load_state_dict(torch.load(f"{model_save_path()}/save_models/client_{malicious_client}_{i}th_model.pth"))

#     # 训练数据
# in_train = np.load(f"{get_data_path()}/attack_train.npz")["intrain"]
# out_train = np.load(f"{get_data_path()}/attack_train.npz")["outtrain"]

# # 测试数据
# in_test = np.load(f"{get_data_path()}/attack_test.npz")["intest"]
# out_test = np.load(f"{get_data_path()}/attack_test.npz")["outtest"]

# data_in_x, data_in_y = MIA_member_data(c13.model, in_train, args)
# data_out_x, data_out_y = MIA_nomember_data(c13.model, out_train, args)

# train_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))

# data_in_x, data_in_y = MIA_member_data(c13.model, in_test, args)
# data_out_x, data_out_y = MIA_nomember_data(c13.model, out_test, args)

# accuracy, recall = test_attacker(np.concatenate((data_in_x, data_out_x), axis=0), np.concatenate((data_in_y, data_out_y), axis=0))

# print(accuracy)
# print(recall)