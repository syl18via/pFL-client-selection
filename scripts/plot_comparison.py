#!/usr/bin/env python
"""
对比实验画图脚本
用法: python scripts/plot_comparison.py --dataset Mnist --model dnn
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from loguru import logger

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = (8, 6)

def read_h5_file(filepath):
    """读取h5文件中的数据"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None, None, None
    
    hf = h5py.File(filepath, 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    hf.close()
    return rs_glob_acc, rs_train_acc, rs_train_loss

def smooth(data, window_len=20):
    """平滑曲线"""
    if window_len < 3 or len(data) < window_len:
        return data
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[window_len-1:]

def build_filename(dataset, algorithm, lr, beta, lamda, num_users, batch_size, local_epochs, 
                   K=None, personal_lr=None, times=None, personalized=False, averaged=False):
    """构建结果文件名
    格式: {dataset}_{algorithm}[_p]_{lr}_{beta}_{lamda}_{num_users}u_{batch_size}b_{local_epochs}[_{K}_{personal_lr}]_{times|avg}.h5
    """
    alg_name = algorithm + ("_p" if personalized else "")
    name = f"{dataset}_{alg_name}_{lr}_{beta}_{lamda}_{num_users}u_{batch_size}b_{local_epochs}"
    
    if algorithm in ["pFedMe"] and K is not None:
        name += f"_{K}_{personal_lr}"
    
    if averaged:
        name += "_avg"
    elif times is not None:
        name += f"_{times}"
    
    return name + ".h5"

def plot_comparison(dataset, model, output_dir="./figures", max_rounds=None):
    """绘制对比图
    
    Args:
        max_rounds: 横坐标最大值，None 表示显示全部轮次
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置参数 - 根据数据集调整
    if dataset == "Mnist":
        num_users = 5
        num_glob_iters = 800
        batch_size = 20
        local_epochs = 20
        lamda = 15
        if model == "dnn":
            lr = 0.005
            personal_lr = 0.09
        else:  # mclr
            lr = 0.005
            personal_lr = 0.1
    elif dataset == "Synthetic":
        num_users = 10
        num_glob_iters = 600
        batch_size = 20
        local_epochs = 20
        lamda = 20
        lr = 0.005
        personal_lr = 0.01
    elif dataset == "Cifar10":
        num_users = 5
        num_glob_iters = 800
        batch_size = 20
        local_epochs = 20
        lamda = 15
        lr = 0.01
        personal_lr = 0.01
    
    beta = 1.0
    K = 5
    
    # 算法配置
    # 注意：所有个性化联邦学习算法使用个性化模型结果（personalized: True）
    # FedAvg 没有个性化模型，使用全局模型结果（personalized: False）
    algorithms_config = [
        {"name": "MESA", "label": "MESA (Ours)", "color": "tab:red", "marker": "o", "personalized": True},
        {"name": "HiCS", "label": "HiCS", "color": "tab:cyan", "marker": "^", "personalized": True},
        {"name": "Oort", "label": "Oort", "color": "tab:blue", "marker": "v", "personalized": True},
        {"name": "PoC", "label": "PoC", "color": "tab:green", "marker": "s", "personalized": True},
        {"name": "pFedMe", "label": "pFedMe", "color": "tab:orange", "marker": "*", "personalized": True},
        {"name": "FedAvg", "label": "FedAvg", "color": "tab:purple", "marker": "x", "personalized": False},  # FedAvg 没有个性化模型
        {"name": "PerAvg", "label": "Per-FedAvg", "color": "tab:brown", "marker": "d", "personalized": True, "beta": 0.001},
    ]
    
    results = {}
    
    # 按模型类型分目录
    result_dir = f"./results/{model}"
    
    # 读取数据
    for alg in algorithms_config:
        alg_beta = alg.get("beta", beta)
        
        # 尝试读取平均后的结果文件
        filename = build_filename(
            dataset, alg["name"], lr, alg_beta, lamda, num_users, batch_size, local_epochs,
            K=K, personal_lr=personal_lr, personalized=alg["personalized"], averaged=True
        )
        filepath = f"./results/{filename}"
        
        # 如果没有平均文件，尝试读取第一次运行的结果
        if not os.path.exists(filepath):
            filename = build_filename(
                dataset, alg["name"], lr, alg_beta, lamda, num_users, batch_size, local_epochs,
                K=K, personal_lr=personal_lr, times=0, personalized=alg["personalized"]
            )
            filepath = f"{result_dir}/{filename}"
        
        test_acc, train_acc, train_loss = read_h5_file(filepath)
        
        if test_acc is not None:
            results[alg["name"]] = {
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "config": alg
            }
            logger.info(f"✓ Loaded {alg['name']}: max_acc={test_acc.max():.4f}, final_acc={test_acc[-1]:.4f}")
        else:
            logger.warning(f"✗ Failed to load {alg['name']}")
    
    if len(results) == 0:
        logger.error("No data found! Please run experiments first.")
        return

    markevery = num_glob_iters//10 if max_rounds is None else max_rounds//10
    
    # 绘制测试准确率
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        cfg = data["config"]
        smoothed = smooth(data["test_acc"])
        ax1.plot(smoothed, label=cfg["label"], color=cfg["color"], 
                marker=cfg["marker"], markevery=markevery, markersize=6, linewidth=1.5)
    
    ax1.set_xlabel('Global Rounds')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title(f'{dataset} - {model.upper()} - Test Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    if max_rounds is not None:
        ax1.set_xlim(0, max_rounds)
    
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/{dataset}_{model}_test_acc.pdf", bbox_inches="tight")
    fig1.savefig(f"{output_dir}/{dataset}_{model}_test_acc.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir}/{dataset}_{model}_test_acc.pdf")
    
    # 绘制训练损失
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        cfg = data["config"]
        smoothed = smooth(data["train_loss"])
        ax2.plot(smoothed, label=cfg["label"], color=cfg["color"],
                marker=cfg["marker"], markevery=markevery, markersize=6, linewidth=1.5)
    
    ax2.set_xlabel('Global Rounds')
    ax2.set_ylabel('Training Loss')
    ax2.set_title(f'{dataset} - {model.upper()} - Training Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    if max_rounds is not None:
        ax2.set_xlim(0, max_rounds)
    
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/{dataset}_{model}_train_loss.pdf", bbox_inches="tight")
    fig2.savefig(f"{output_dir}/{dataset}_{model}_train_loss.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir}/{dataset}_{model}_train_loss.pdf")
    
    # 打印最终结果汇总
    logger.info("\n" + "="*60)
    logger.info(f"Results Summary: {dataset} - {model.upper()}")
    logger.info("="*60)
    logger.info(f"{'Algorithm':<15} {'Max Acc':>10} {'Final Acc':>10} {'Min Loss':>10}")
    logger.info("-"*60)
    for name, data in results.items():
        cfg = data["config"]
        logger.info(f"{cfg['label']:<15} {data['test_acc'].max():>10.4f} {data['test_acc'][-1]:>10.4f} {data['train_loss'].min():>10.4f}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--output", type=str, default="./figures", help="Output directory for figures")
    parser.add_argument("--max_rounds", type=int, default=None, help="Max rounds to display (default: all)")
    args = parser.parse_args()
    
    plot_comparison(args.dataset, args.model, args.output, args.max_rounds)
