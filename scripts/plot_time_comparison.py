#!/usr/bin/env python
"""
绘制 Accuracy vs. Wall-clock Time 对比图
用法: python scripts/plot_time_comparison.py --dataset Mnist --model dnn
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = (8, 6)

def read_h5_file_with_time(filepath, use_personalized=False):
    """读取h5文件中的数据，包括时间数据
    
    Args:
        filepath: h5文件路径
        use_personalized: 是否读取个性化模型的结果（从 _p 文件中读取）
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None, None, None
    
    hf = h5py.File(filepath, 'r')
    
    # 读取准确率和损失
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    
    # 读取时间数据（全局模型和个性化模型使用相同的时间）
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

def plot_time_comparison(dataset, model, output_dir="./figures", max_time=None, use_personalized=True):
    """绘制 Accuracy vs. Wall-clock Time 对比图
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        output_dir: 输出目录
        max_time: 最大时间（秒），None 表示显示全部时间
        use_personalized: 是否使用个性化模型的准确率（True）还是全局模型（False）
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
    else:
        print(f"Unsupported dataset: {dataset}")
        return
    
    beta = 1.0
    K = 5
    
    # 算法配置 - 与 plot_comparison.py 保持一致
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
        filepath = f"{result_dir}/{filename}"
        
        # 如果没有平均文件，尝试读取第一次运行的结果
        if not os.path.exists(filepath):
            filename = build_filename(
                dataset, alg["name"], lr, alg_beta, lamda, num_users, batch_size, local_epochs,
                K=K, personal_lr=personal_lr, times=0, personalized=alg["personalized"]
            )
            filepath = f"{result_dir}/{filename}"
        
        # 对于个性化算法，如果需要使用个性化结果，直接读取 _p 文件
        if use_personalized and alg["personalized"]:
            # 先尝试读取个性化模型的结果文件（_p 后缀）
            filename_p = build_filename(
                dataset, alg["name"], lr, alg_beta, lamda, num_users, batch_size, local_epochs,
                K=K, personal_lr=personal_lr, personalized=True, averaged=True
            )
            filepath_p = f"{result_dir}/{filename_p}"
            
            if not os.path.exists(filepath_p):
                # 尝试读取非平均的个性化结果
                filename_p = build_filename(
                    dataset, alg["name"], lr, alg_beta, lamda, num_users, batch_size, local_epochs,
                    K=K, personal_lr=personal_lr, times=0, personalized=True
                )
                filepath_p = f"{result_dir}/{filename_p}"
            
            if os.path.exists(filepath_p):
                filepath = filepath_p
        
        test_acc, train_acc, train_loss, wall_clock_times, round_times = read_h5_file_with_time(filepath)
            
            results[alg["name"]] = {
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "wall_clock_times": wall_clock_times,
                "round_times": round_times,
                "config": alg
            }
            print(f"✓ Loaded {alg['name']}: max_acc={test_acc.max():.4f}, final_acc={test_acc[-1]:.4f}, total_time={wall_clock_times[-1]:.2f}s")
        else:
            if test_acc is None:
                print(f"✗ Failed to load {alg['name']}: no accuracy data")
            else:
                print(f"✗ Failed to load {alg['name']}: no time data")
    
    if len(results) == 0:
        print("No data found! Please run experiments first.")
        return
    
    # 绘制 Accuracy vs. Wall-clock Time
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, data in results.items():
        cfg = data["config"]
        accuracies = data["test_acc"]
        times = data["wall_clock_times"]
        
        # 确保时间数组和准确率数组长度一致
        min_len = min(len(accuracies), len(times))
        accuracies = accuracies[:min_len]
        times = times[:min_len]
        
        # 平滑处理（可选）
        smoothed_acc = smooth(accuracies)
        smoothed_times = times[:len(smoothed_acc)]
        
        # 计算标记点间隔（每10%的时间一个标记）
        if len(smoothed_times) > 0:
            max_time_val = smoothed_times[-1] if max_time is None else max_time
            markevery = max(1, len(smoothed_times) // 10)
        else:
            markevery = 1
        
        ax.plot(smoothed_times, smoothed_acc, label=cfg["label"], color=cfg["color"], 
                marker=cfg["marker"], markevery=markevery, markersize=6, linewidth=1.5)
    
    ax.set_xlabel('Wall-clock Time (seconds)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    title = f'{dataset} - {model.upper()} - Test Accuracy vs. Wall-clock Time'
    if use_personalized:
        title += ' (Personalized)'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if max_time is not None:
        ax.set_xlim(0, max_time)
    
    # 格式化 x 轴（时间）显示
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/60:.1f}' if x >= 60 else f'{x:.0f}'))
    if max_time is None or max_time >= 60:
        ax.set_xlabel('Wall-clock Time (minutes)', fontsize=12)
    
    fig.tight_layout()
    
    # 保存图片
    suffix = "_personalized" if use_personalized else "_global"
    fig.savefig(f"{output_dir}/{dataset}_{model}_acc_vs_time{suffix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_dir}/{dataset}_{model}_acc_vs_time{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir}/{dataset}_{model}_acc_vs_time{suffix}.pdf")
    
    # 打印结果汇总
    print("\n" + "="*70)
    print(f"Results Summary: {dataset} - {model.upper()} - Accuracy vs. Time")
    print("="*70)
    print(f"{'Algorithm':<15} {'Max Acc':>10} {'Final Acc':>10} {'Total Time':>12} {'Time to 90%':>12}")
    print("-"*70)
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
        
        # 计算达到90%最大准确率的时间
        target_acc = max_acc * 0.9
        time_to_90 = None
        for i, acc in enumerate(accuracies):
            if acc >= target_acc:
                time_to_90 = times[i]
                break
        
        time_to_90_str = f"{time_to_90:.1f}s" if time_to_90 is not None else "N/A"
        total_time_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time/60:.1f}min"
        
        print(f"{cfg['label']:<15} {max_acc:>10.4f} {final_acc:>10.4f} {total_time_str:>12} {time_to_90_str:>12}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Accuracy vs. Wall-clock Time")
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--output", type=str, default="./figures", help="Output directory for figures")
    parser.add_argument("--max_time", type=float, default=None, help="Max time to display in seconds (default: all)")
    parser.add_argument("--use_personalized", action="store_true", default=True, help="Use personalized model accuracy (default: True)")
    parser.add_argument("--use_global", action="store_true", default=False, help="Use global model accuracy instead of personalized")
    args = parser.parse_args()
    
    # 如果指定了 --use_global，则使用全局模型
    use_personalized = not args.use_global if args.use_global else args.use_personalized
    
    plot_time_comparison(args.dataset, args.model, args.output, args.max_time, use_personalized)
