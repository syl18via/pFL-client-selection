#!/usr/bin/env python
"""
对比实验画图脚本
用法: python scripts/plot_comparison.py --result_dir results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs
"""

import sys
sys.path.insert(0, '.')  # 确保能导入项目模块

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from loguru import logger
from utils.path_utils import parse_result_dir, parse_result_filename

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = (8, 6)

# 算法显示配置
ALGORITHM_STYLES = {
    "MESA": {"label": "MESA (Ours)", "color": "tab:red", "marker": "o"},
    "HiCS": {"label": "HiCS", "color": "tab:cyan", "marker": "^"},
    "Oort": {"label": "Oort", "color": "tab:blue", "marker": "v"},
    "PoC": {"label": "PoC", "color": "tab:green", "marker": "s"},
    "pFedMe": {"label": "pFedMe", "color": "tab:orange", "marker": "*"},
    "FedAvg": {"label": "FedAvg", "color": "tab:purple", "marker": "x"},
    "PerAvg": {"label": "Per-FedAvg", "color": "tab:brown", "marker": "d"},
}

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

def scan_result_files(result_dir):
    """扫描结果目录，返回所有算法的结果文件信息
    
    Returns:
        dict: {algorithm_name: {"filepath": ..., "info": parsed_info, "personalized": bool}}
    """
    if not os.path.isdir(result_dir):
        logger.error(f"Directory not found: {result_dir}")
        return {}
    
    files = {}
    for filename in os.listdir(result_dir):
        if not filename.endswith('.h5'):
            continue
        
        info = parse_result_filename(filename)
        if info is None:
            logger.warning(f"Cannot parse filename: {filename}")
            continue
        
        alg = info['algorithm']
        personalized = info.get('personalized', False)
        current_time = info.get('current_time')
        
        # 构建唯一的 key（算法 + 是否个性化）
        key = f"{alg}_p" if personalized else alg
        
        # 优先使用 avg 文件，其次使用 current_time=0 的文件
        filepath = os.path.join(result_dir, filename)
        
        if key not in files:
            files[key] = {
                "filepath": filepath,
                "info": info,
                "algorithm": alg,
                "personalized": personalized,
            }
        else:
            # 如果已存在，优先保留 avg 文件
            existing_time = files[key]["info"].get('current_time')
            if current_time == "avg" and existing_time != "avg":
                files[key] = {
                    "filepath": filepath,
                    "info": info,
                    "algorithm": alg,
                    "personalized": personalized,
                }
            elif existing_time != "avg" and current_time == 0:
                # 如果都不是 avg，优先使用 current_time=0
                if existing_time != 0:
                    files[key] = {
                        "filepath": filepath,
                        "info": info,
                        "algorithm": alg,
                        "personalized": personalized,
                    }
    
    return files

def plot_from_result_dir(result_dir, output_dir="./figures", max_rounds=None):
    """从结果目录自动读取并绘制对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析目录名获取实验参数
    dir_name = os.path.basename(result_dir.rstrip('/'))
    dir_info = parse_result_dir(dir_name)
    
    if dir_info is None:
        logger.error(f"Cannot parse directory name: {dir_name}")
        return
    
    model = dir_info.get('model_name', 'unknown')
    dataset = dir_info.get('dataset', 'unknown')
    num_glob_iters = dir_info.get('num_glob_iters', 100)
    
    logger.info(f"Parsed experiment info: model={model}, dataset={dataset}, rounds={num_glob_iters}")
    
    # 扫描目录中的结果文件
    all_files = scan_result_files(result_dir)
    
    if not all_files:
        logger.error("No result files found!")
        return
    
    # 决定使用哪些文件（优先个性化模型，FedAvg 除外）
    # 对于同一个算法，如果有 _p 版本就用 _p，否则用普通版本
    algorithms_to_plot = {}
    
    for key, file_info in all_files.items():
        alg = file_info["algorithm"]
        personalized = file_info["personalized"]
        
        # FedAvg 没有个性化模型，直接使用
        if alg == "FedAvg":
            if alg not in algorithms_to_plot:
                algorithms_to_plot[alg] = file_info
        else:
            # 其他算法优先使用个性化模型
            if personalized:
                algorithms_to_plot[alg] = file_info
            elif alg not in algorithms_to_plot:
                algorithms_to_plot[alg] = file_info
    
    # 读取数据
    results = {}
    for alg, file_info in algorithms_to_plot.items():
        filepath = file_info["filepath"]
        test_acc, train_acc, train_loss = read_h5_file(filepath)
        
        if test_acc is not None:
            style = ALGORITHM_STYLES.get(alg, {"label": alg, "color": "gray", "marker": "."})
            results[alg] = {
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "config": style,
                "filepath": filepath,
            }
            logger.info(f"✓ Loaded {alg}: max_acc={test_acc.max():.4f}, final_acc={test_acc[-1]:.4f} ({os.path.basename(filepath)})")
        else:
            logger.warning(f"✗ Failed to load {alg}")
    
    if len(results) == 0:
        logger.error("No data loaded!")
        return
    
    # 计算标记间隔
    data_len = len(list(results.values())[0]["test_acc"])
    markevery = max(data_len // 10, 1) if max_rounds is None else max(max_rounds // 10, 1)
    
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
    parser = argparse.ArgumentParser(description="Plot comparison from result directory")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to result directory (e.g., results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs)")
    parser.add_argument("--output", type=str, default="./figures", help="Output directory for figures")
    parser.add_argument("--max_rounds", type=int, default=None, help="Max rounds to display (default: all)")
    args = parser.parse_args()
    
    plot_from_result_dir(args.result_dir, args.output, args.max_rounds)
