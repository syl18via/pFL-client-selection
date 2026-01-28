#!/usr/bin/env python
"""
绘制被选中 Client 的 Loss 分布图（用于 Effectiveness of Selection 分析）
用法: python scripts/plot_selection_effectiveness.py --dataset Mnist --model dnn
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy import stats

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = (10, 6)

def read_h5_file_with_selection(filepath):
    """读取h5文件中的被选中 client loss 数据
    
    Args:
        filepath: h5文件路径
    Returns:
        selected_client_losses: 每轮被选中 client 的 loss 数组 (num_rounds, num_selected)
        selected_client_indices: 每轮被选中 client 的索引数组 (num_rounds, num_selected)
        rs_glob_acc: 全局模型准确率
        wall_clock_times: 累计时间（可选）
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None, None
    
    hf = h5py.File(filepath, 'r')
    
    # 读取被选中 client 的 loss
    selected_client_losses = None
    selected_client_indices = None
    if 'selected_client_losses' in hf.keys():
        selected_client_losses = np.array(hf.get('selected_client_losses')[:])
    if 'selected_client_indices' in hf.keys():
        selected_client_indices = np.array(hf.get('selected_client_indices')[:])
    
    # 读取准确率（用于参考）
    rs_glob_acc = None
    if 'rs_glob_acc' in hf.keys():
        rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    
    # 读取时间数据（可选）
    wall_clock_times = None
    if 'wall_clock_times' in hf.keys():
        wall_clock_times = np.array(hf.get('wall_clock_times')[:])
    
    hf.close()
    return selected_client_losses, selected_client_indices, rs_glob_acc, wall_clock_times

def build_filename(dataset, algorithm, lr, beta, lamda, num_users, batch_size, local_epochs, 
                   K=None, personal_lr=None, times=None, personalized=False, averaged=False):
    """构建结果文件名"""
    alg_name = algorithm + ("_p" if personalized else "")
    name = f"{dataset}_{alg_name}_{lr}_{beta}_{lamda}_{num_users}u_{batch_size}b_{local_epochs}"
    
    if algorithm in ["pFedMe"] and K is not None:
        name += f"_{K}_{personal_lr}"
    
    if averaged:
        name += "_avg"
    elif times is not None:
        name += f"_{times}"
    
    return name + ".h5"

def plot_loss_distribution_comparison(dataset, model, output_dir="./figures", 
                                     use_personalized=True, rounds_to_plot=None):
    """绘制不同算法的被选中 client loss 分布对比图
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        output_dir: 输出目录
        use_personalized: 是否使用个性化模型的结果
        rounds_to_plot: 要绘制的轮次列表，None 表示使用所有轮次
    """
    # 算法配置（与 plot_comparison.py 保持一致）
    algorithms_config = [
        {"name": "MESA", "label": "MESA (Ours)", "color": "tab:red", "marker": "o", "personalized": True, "metric": "Proximal Gap"},
        {"name": "HiCS", "label": "HiCS", "color": "tab:cyan", "marker": "^", "personalized": True},
        {"name": "Oort", "label": "Oort", "color": "tab:blue", "marker": "v", "personalized": True},
        {"name": "PoC", "label": "PoC", "color": "tab:green", "marker": "s", "personalized": True},
        {"name": "pFedMe", "label": "pFedMe", "color": "tab:orange", "marker": "*", "personalized": True},
        {"name": "FedAvg", "label": "FedAvg", "color": "tab:purple", "marker": "x", "personalized": False},
        {"name": "PerAvg", "label": "Per-FedAvg", "color": "tab:brown", "marker": "d", "personalized": True, "beta": 0.001},
    ]
    
    # 实验参数（需要与 main.py 中的参数匹配）
    # 根据数据集和模型自动设置参数
    if dataset == "Mnist":
        num_users = 5
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
        batch_size = 20
        local_epochs = 20
        lamda = 20
        lr = 0.005
        personal_lr = 0.01
    elif dataset == "Cifar10":
        num_users = 5
        batch_size = 20
        local_epochs = 20
        lamda = 15
        lr = 0.01
        personal_lr = 0.01
    else:
        # 默认值
        num_users = 5
        batch_size = 20
        local_epochs = 20
        lamda = 15
        lr = 0.005
        personal_lr = 0.09
    
    beta = 1.0
    K = 5
    
    result_dir = f"./results/{model}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有算法的 loss 数据
    all_losses = {}
    all_labels = {}
    all_colors = {}
    all_metrics = {}
    
    for cfg in algorithms_config:
        alg_name = cfg["name"]
        personalized = cfg.get("personalized", False) if use_personalized else False
        
        # 构建文件名
        if alg_name == "PerAvg" and "beta" in cfg:
            # PerAvg 使用不同的 beta
            filename = build_filename(dataset, alg_name, lr, cfg["beta"], lamda, 
                                     num_users, batch_size, local_epochs, 
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        else:
            filename = build_filename(dataset, alg_name, lr, beta, lamda, 
                                     num_users, batch_size, local_epochs, 
                                     K=K if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     personal_lr=personal_lr if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        
        filepath = os.path.join(result_dir, filename)
        
        # 尝试读取文件
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            # 处理 NaN 值：移除每轮中的 NaN
            processed_losses = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                if len(valid_losses) > 0:
                    processed_losses.extend(valid_losses.tolist())
            
            if len(processed_losses) > 0:
                all_losses[alg_name] = processed_losses
                all_labels[alg_name] = cfg["label"]
                all_colors[alg_name] = cfg["color"]
                all_metrics[alg_name] = cfg.get("metric", "Loss")
        else:
            print(f"Warning: Could not read selection data for {alg_name} from {filepath}")
    
    if len(all_losses) == 0:
        print("Error: No valid data found!")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: Loss 分布直方图对比
    ax1 = axes[0]
    bins = np.linspace(min(min(losses) for losses in all_losses.values()), 
                      max(max(losses) for losses in all_losses.values()), 
                      30)
    
    for alg_name, losses in all_losses.items():
        ax1.hist(losses, bins=bins, alpha=0.5, label=all_labels[alg_name], 
                color=all_colors[alg_name], density=True)
    
    ax1.set_xlabel(f'Selected Client {all_metrics[list(all_losses.keys())[0]]}')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Selected Client Losses')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Loss 分布箱线图对比
    ax2 = axes[1]
    box_data = [all_losses[alg_name] for alg_name in all_losses.keys()]
    box_labels = [all_labels[alg_name] for alg_name in all_losses.keys()]
    box_colors = [all_colors[alg_name] for alg_name in all_losses.keys()]
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel(f'Selected Client {all_metrics[list(all_losses.keys())[0]]}')
    ax2.set_title('Box Plot of Selected Client Losses')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    metric_name = "ProximalGap" if "MESA" in all_losses else "Loss"
    output_filename = f"{dataset}_{model}_selection_loss_distribution"
    if use_personalized:
        output_filename += "_personalized"
    output_filename += ".png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()

def plot_loss_over_rounds(dataset, model, output_dir="./figures", use_personalized=True):
    """绘制被选中 client loss 随轮次的变化
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        output_dir: 输出目录
        use_personalized: 是否使用个性化模型的结果
    """
    # 算法配置
    algorithms_config = [
        {"name": "MESA", "label": "MESA (Ours)", "color": "tab:red", "marker": "o", "personalized": True, "metric": "Proximal Gap"},
        {"name": "HiCS", "label": "HiCS", "color": "tab:cyan", "marker": "^", "personalized": True},
        {"name": "Oort", "label": "Oort", "color": "tab:blue", "marker": "v", "personalized": True},
        {"name": "PoC", "label": "PoC", "color": "tab:green", "marker": "s", "personalized": True},
        {"name": "pFedMe", "label": "pFedMe", "color": "tab:orange", "marker": "*", "personalized": True},
        {"name": "FedAvg", "label": "FedAvg", "color": "tab:purple", "marker": "x", "personalized": False},
        {"name": "PerAvg", "label": "Per-FedAvg", "color": "tab:brown", "marker": "d", "personalized": True, "beta": 0.001},
    ]
    
    # 实验参数
    lr = 0.005
    beta = 1.0
    lamda = 15
    num_users = 5
    batch_size = 20
    local_epochs = 20
    K = 5
    personal_lr = 0.09
    
    result_dir = f"./results/{model}"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cfg in algorithms_config:
        alg_name = cfg["name"]
        personalized = cfg.get("personalized", False) if use_personalized else False
        
        # 构建文件名
        if alg_name == "PerAvg" and "beta" in cfg:
            filename = build_filename(dataset, alg_name, lr, cfg["beta"], lamda, 
                                     num_users, batch_size, local_epochs, 
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        else:
            filename = build_filename(dataset, alg_name, lr, beta, lamda, 
                                     num_users, batch_size, local_epochs, 
                                     K=K if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     personal_lr=personal_lr if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        
        filepath = os.path.join(result_dir, filename)
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            # 计算每轮的平均 loss
            round_means = []
            round_stds = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                if len(valid_losses) > 0:
                    round_means.append(np.mean(valid_losses))
                    round_stds.append(np.std(valid_losses))
            
            if len(round_means) > 0:
                rounds = np.arange(len(round_means))
                round_means = np.array(round_means)
                round_stds = np.array(round_stds)
                
                # 绘制均值线和标准差阴影
                ax.plot(rounds, round_means, label=cfg["label"], 
                       color=cfg["color"], marker=cfg["marker"], 
                       markersize=6, linewidth=1.5, markevery=max(1, len(rounds)//20))
                ax.fill_between(rounds, round_means - round_stds, round_means + round_stds,
                               alpha=0.2, color=cfg["color"])
    
    metric_label = "Proximal Gap" if use_personalized else "Loss"
    ax.set_xlabel('Round')
    ax.set_ylabel(f'Mean Selected Client {metric_label}')
    ax.set_title(f'Selected Client {metric_label} Over Rounds')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_filename = f"{dataset}_{model}_selection_loss_over_rounds"
    if use_personalized:
        output_filename += "_personalized"
    output_filename += ".png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()

def plot_statistics_summary(dataset, model, output_dir="./figures", use_personalized=True):
    """绘制统计摘要表格
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        use_personalized: 是否使用个性化模型的结果
    """
    # 算法配置
    algorithms_config = [
        {"name": "MESA", "label": "MESA (Ours)", "color": "tab:red", "marker": "o", "personalized": True, "metric": "Proximal Gap"},
        {"name": "HiCS", "label": "HiCS", "color": "tab:cyan", "marker": "^", "personalized": True},
        {"name": "Oort", "label": "Oort", "color": "tab:blue", "marker": "v", "personalized": True},
        {"name": "PoC", "label": "PoC", "color": "tab:green", "marker": "s", "personalized": True},
        {"name": "pFedMe", "label": "pFedMe", "color": "tab:orange", "marker": "*", "personalized": True},
        {"name": "FedAvg", "label": "FedAvg", "color": "tab:purple", "marker": "x", "personalized": False},
        {"name": "PerAvg", "label": "Per-FedAvg", "color": "tab:brown", "marker": "d", "personalized": True, "beta": 0.001},
    ]
    
    # 实验参数
    lr = 0.005
    beta = 1.0
    lamda = 15
    num_users = 5
    batch_size = 20
    local_epochs = 20
    K = 5
    personal_lr = 0.09
    
    result_dir = f"./results/{model}"
    
    # 收集统计数据
    stats_data = []
    
    for cfg in algorithms_config:
        alg_name = cfg["name"]
        personalized = cfg.get("personalized", False) if use_personalized else False
        
        # 构建文件名
        if alg_name == "PerAvg" and "beta" in cfg:
            filename = build_filename(dataset, alg_name, lr, cfg["beta"], lamda, 
                                     num_users, batch_size, local_epochs, 
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        else:
            filename = build_filename(dataset, alg_name, lr, beta, lamda, 
                                     num_users, batch_size, local_epochs, 
                                     K=K if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     personal_lr=personal_lr if alg_name in ["pFedMe", "MESA", "HiCS", "Oort", "PoC"] else None,
                                     times="avg" if True else None, 
                                     personalized=personalized, averaged=True)
        
        filepath = os.path.join(result_dir, filename)
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            # 收集所有有效的 loss
            all_losses = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                all_losses.extend(valid_losses.tolist())
            
            if len(all_losses) > 0:
                all_losses = np.array(all_losses)
                stats_data.append({
                    "Algorithm": cfg["label"],
                    "Mean": np.mean(all_losses),
                    "Std": np.std(all_losses),
                    "Median": np.median(all_losses),
                    "Min": np.min(all_losses),
                    "Max": np.max(all_losses),
                    "Q25": np.percentile(all_losses, 25),
                    "Q75": np.percentile(all_losses, 75),
                })
    
    if len(stats_data) == 0:
        print("Error: No valid data found!")
        return
    
    # 打印统计摘要
    print("\n" + "="*80)
    print(f"Selected Client Loss Statistics Summary ({dataset}, {model})")
    print("="*80)
    print(f"{'Algorithm':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    
    for stats in stats_data:
        print(f"{stats['Algorithm']:<20} {stats['Mean']:<12.4f} {stats['Std']:<12.4f} "
              f"{stats['Median']:<12.4f} {stats['Min']:<12.4f} {stats['Max']:<12.4f}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot selected client loss distribution')
    parser.add_argument("--dataset", type=str, default="Mnist", help="Dataset name")
    parser.add_argument("--model", type=str, default="dnn", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./figures", help="Output directory")
    parser.add_argument("--use_personalized", action="store_true", default=True, 
                       help="Use personalized model results (default: True)")
    parser.add_argument("--use_global", action="store_true", default=False, 
                       help="Use global model results instead of personalized")
    parser.add_argument("--plot_type", type=str, default="all", 
                       choices=["all", "distribution", "over_rounds", "stats"],
                       help="Type of plot to generate")
    
    args = parser.parse_args()
    
    use_personalized = args.use_personalized and not args.use_global
    
    if args.plot_type in ["all", "distribution"]:
        plot_loss_distribution_comparison(args.dataset, args.model, args.output_dir, use_personalized)
    
    if args.plot_type in ["all", "over_rounds"]:
        plot_loss_over_rounds(args.dataset, args.model, args.output_dir, use_personalized)
    
    if args.plot_type in ["all", "stats"]:
        plot_statistics_summary(args.dataset, args.model, args.output_dir, use_personalized)
