import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="har70")
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=4, help="random seed (default: 0)")

    parser.add_argument("--attack", type=bool, default=False)
    # 0表示不加l2,1表示加l2
    parser.add_argument("--l2", type=int, default=0)
    parser.add_argument("--mu", type=float, default=0.001)
    parser.add_argument("--note", type=str, default="mix3")
    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--algorithm", type=str, default="avg")
    parser.add_argument("--mix_clients", type=int, default=3)
    parser.add_argument("--dp_epsilon", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    args = parser.parse_args()
    device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
    args.device = device

    return args
