#!/usr/bin/env python
"""
绘制 Accuracy vs. Wall-clock Time 对比图
用法: python scripts/plot_time_comparison.py --result_dir results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs
"""

import sys
sys.path.insert(0, '.')

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

def read_h5_file_with_time(filepath):
    """读取h5文件中的数据，包括时间数据"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None, None, None, None, None
    
    hf = h5py.File(filepath, 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    
    wall_clock_times = None
    round_times = None
    if 'wall_clock_times' in hf.keys():
        wall_clock_times = np.array(hf.get('wall_clock_times')[:])
    if 'round_times' in hf.keys():
        round_times = np.array(hf.get('round_times')[:])
    
    hf.close()
    return rs_glob_acc, rs_train_acc, rs_train_loss, wall_clock_times, round_times

def smooth(data, window_len=20):
    """平滑曲线"""
    if window_len < 3 or len(data) < window_len:
        return data
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[window_len-1:]

def scan_result_files(result_dir, prefer_single_run=False):
    """扫描结果目录，返回所有算法的结果文件信息
    
    Args:
        result_dir: 结果目录路径
        prefer_single_run: 是否优先选择单次实验文件（用于时间数据分析，因为 avg 文件不包含时间数据）
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
            continue
        
        alg = info['algorithm']
        personalized = info.get('personalized', False)
        current_time = info.get('current_time')
        
        key = f"{alg}_p" if personalized else alg
        filepath = os.path.join(result_dir, filename)
        
        if key not in files:
            files[key] = {"filepath": filepath, "info": info, "algorithm": alg, "personalized": personalized}
        else:
            existing_time = files[key]["info"].get('current_time')
            
            if prefer_single_run:
                # 优先单次实验文件（current_time=0），因为 avg 文件不包含时间数据
                if current_time == 0 and existing_time != 0:
                    files[key] = {"filepath": filepath, "info": info, "algorithm": alg, "personalized": personalized}
                elif existing_time == "avg" and current_time != "avg":
                    files[key] = {"filepath": filepath, "info": info, "algorithm": alg, "personalized": personalized}
            else:
                # 优先 avg 文件
                if current_time == "avg" and existing_time != "avg":
                    files[key] = {"filepath": filepath, "info": info, "algorithm": alg, "personalized": personalized}
                elif existing_time != "avg" and current_time == 0 and existing_time != 0:
                    files[key] = {"filepath": filepath, "info": info, "algorithm": alg, "personalized": personalized}
    
    return files

def plot_time_comparison(result_dir, output_dir="./figures", max_time=None):
    """从结果目录绘制 Accuracy vs. Wall-clock Time 对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析目录名获取实验参数
    dir_name = os.path.basename(result_dir.rstrip('/'))
    dir_info = parse_result_dir(dir_name)
    
    if dir_info is None:
        logger.error(f"Cannot parse directory name: {dir_name}")
        return
    
    model = dir_info.get('model_name', 'unknown')
    dataset = dir_info.get('dataset', 'unknown')
    
    logger.info(f"Parsed experiment info: model={model}, dataset={dataset}")
    
    # 扫描目录中的结果文件
    # 优先使用 _avg 文件（多轮平均），如果 _avg 文件没有时间数据会自动回退到单次实验文件
    all_files = scan_result_files(result_dir, prefer_single_run=False)
    
    if not all_files:
        logger.error("No result files found!")
        return
    
    # 优先使用个性化模型（FedAvg 除外）
    algorithms_to_plot = {}
    for key, file_info in all_files.items():
        alg = file_info["algorithm"]
        personalized = file_info["personalized"]
        
        if alg == "FedAvg":
            if alg not in algorithms_to_plot:
                algorithms_to_plot[alg] = file_info
        else:
            if personalized:
                algorithms_to_plot[alg] = file_info
            elif alg not in algorithms_to_plot:
                algorithms_to_plot[alg] = file_info
    
    # 读取数据
    results = {}
    for alg, file_info in algorithms_to_plot.items():
        filepath = file_info["filepath"]
        test_acc, train_acc, train_loss, wall_clock_times, round_times = read_h5_file_with_time(filepath)
        
        if test_acc is not None and wall_clock_times is not None:
            style = ALGORITHM_STYLES.get(alg, {"label": alg, "color": "gray", "marker": "."})
            results[alg] = {
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "wall_clock_times": wall_clock_times,
                "round_times": round_times,
                "config": style,
            }
            logger.info(f"✓ Loaded {alg}: max_acc={test_acc.max():.4f}, total_time={wall_clock_times[-1]:.2f}s")
        else:
            if test_acc is None:
                logger.warning(f"✗ {alg}: no accuracy data")
            else:
                logger.warning(f"✗ {alg}: no time data")
    
    if len(results) == 0:
        logger.error("No data with time info found!")
        return
    
    # 绘制 Accuracy vs. Wall-clock Time
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, data in results.items():
        cfg = data["config"]
        accuracies = data["test_acc"]
        times = data["wall_clock_times"]
        
        min_len = min(len(accuracies), len(times))
        accuracies = accuracies[:min_len]
        times = times[:min_len]
        
        smoothed_acc = smooth(accuracies)
        smoothed_times = times[:len(smoothed_acc)]
        
        markevery = max(1, len(smoothed_times) // 10) if len(smoothed_times) > 0 else 1
        
        ax.plot(smoothed_times, smoothed_acc, label=cfg["label"], color=cfg["color"], 
                marker=cfg["marker"], markevery=markevery, markersize=6, linewidth=1.5)
    
    ax.set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'{dataset} - {model.upper()} - Test Accuracy vs. Wall-clock Time', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if max_time is not None:
        ax.set_xlim(0, max_time)
    
    # 如果时间超过60秒，用分钟显示
    all_times = [data["wall_clock_times"][-1] for data in results.values()]
    if max(all_times) >= 60:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/60:.1f}'))
        ax.set_xlabel('Wall-clock Time (minutes)', fontsize=12)
    
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{dataset}_{model}_acc_vs_time.pdf", bbox_inches="tight")
    fig.savefig(f"{output_dir}/{dataset}_{model}_acc_vs_time.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {output_dir}/{dataset}_{model}_acc_vs_time.pdf")
    
    # 打印结果汇总
    logger.info("\n" + "="*70)
    logger.info(f"Results Summary: {dataset} - {model.upper()} - Accuracy vs. Time")
    logger.info("="*70)
    logger.info(f"{'Algorithm':<15} {'Max Acc':>10} {'Final Acc':>10} {'Total Time':>12} {'Time to 90%':>12}")
    logger.info("-"*70)
    
    for name, data in results.items():
        cfg = data["config"]
        accuracies = data["test_acc"]
        times = data["wall_clock_times"]
        
        min_len = min(len(accuracies), len(times))
        accuracies = accuracies[:min_len]
        times = times[:min_len]
        
        max_acc = accuracies.max()
        final_acc = accuracies[-1]
        total_time = times[-1]
        
        target_acc = max_acc * 0.9
        time_to_90 = None
        for i, acc in enumerate(accuracies):
            if acc >= target_acc:
                time_to_90 = times[i]
                break
        
        time_to_90_str = f"{time_to_90:.1f}s" if time_to_90 is not None else "N/A"
        total_time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time/60:.1f}min"
        
        logger.info(f"{cfg['label']:<15} {max_acc:>10.4f} {final_acc:>10.4f} {total_time_str:>12} {time_to_90_str:>12}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Accuracy vs. Wall-clock Time")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to result directory")
    parser.add_argument("--output", type=str, default="./figures", help="Output directory for figures")
    parser.add_argument("--max_time", type=float, default=None, help="Max time to display in seconds")
    args = parser.parse_args()
    
    plot_time_comparison(args.result_dir, args.output, args.max_time)
