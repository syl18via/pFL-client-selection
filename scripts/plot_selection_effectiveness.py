#!/usr/bin/env python
"""
绘制被选中 Client 的 Loss 分布图（用于 Effectiveness of Selection 分析）
用法: python scripts/plot_selection_effectiveness.py --result_dir results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs
"""

import sys
sys.path.insert(0, '.')

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
import pandas as pd
from loguru import logger
from utils.path_utils import parse_result_dir, parse_result_filename

plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

# 算法显示配置
ALGORITHM_STYLES = {
    "MESA": {"label": "MESA (Ours)", "color": "tab:red", "marker": "o", "metric": "Proximal Gap"},
    "HiCS": {"label": "HiCS", "color": "tab:cyan", "marker": "^"},
    "Oort": {"label": "Oort", "color": "tab:blue", "marker": "v"},
    "PoC": {"label": "PoC", "color": "tab:green", "marker": "s"},
    "pFedMe": {"label": "Random (pFedMe)", "color": "tab:orange", "marker": "*"},
    "FedAvg": {"label": "FedAvg", "color": "tab:purple", "marker": "x"},
    "PerAvg": {"label": "Per-FedAvg", "color": "tab:brown", "marker": "d"},
}

def read_h5_file_with_selection(filepath):
    """读取h5文件中的被选中 client loss 数据"""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return None, None, None, None
    
    hf = h5py.File(filepath, 'r')
    
    selected_client_losses = None
    selected_client_indices = None
    if 'selected_client_losses' in hf.keys():
        selected_client_losses = np.array(hf.get('selected_client_losses')[:])
    if 'selected_client_indices' in hf.keys():
        indices_data = np.array(hf.get('selected_client_indices')[:])
        # 如果保存的是字节字符串，需要解码
        if indices_data.dtype.kind == 'S':  # byte string
            # 处理多维数组：每轮的被选中 client 索引
            decoded_indices = []
            for row in indices_data:
                decoded_row = []
                for idx in row:
                    if isinstance(idx, bytes):
                        decoded_idx = idx.decode('utf-8').strip('\x00')
                    else:
                        decoded_idx = str(idx).strip('\x00')
                    # 过滤空字符串（填充值）
                    if decoded_idx and decoded_idx != '':
                        decoded_row.append(decoded_idx)
                decoded_indices.append(decoded_row)
            selected_client_indices = np.array(decoded_indices, dtype=object)
        else:
            # 非字节字符串，直接转换
            decoded_indices = []
            for row in indices_data:
                decoded_row = [str(idx) for idx in row if idx != -1 and not (isinstance(idx, str) and idx == '')]
                decoded_indices.append(decoded_row)
            selected_client_indices = np.array(decoded_indices, dtype=object)
    
    rs_glob_acc = None
    if 'rs_glob_acc' in hf.keys():
        rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    
    wall_clock_times = None
    if 'wall_clock_times' in hf.keys():
        wall_clock_times = np.array(hf.get('wall_clock_times')[:])
    
    hf.close()
    return selected_client_losses, selected_client_indices, rs_glob_acc, wall_clock_times

def scan_result_files(result_dir, prefer_single_run=False):
    """扫描结果目录，返回所有算法的结果文件信息
    
    Args:
        result_dir: 结果目录路径
        prefer_single_run: 是否优先选择单次实验文件（用于 selection data 分析，因为 avg 文件不包含 selection data）
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
                # 优先单次实验文件（current_time=0），因为 avg 文件不包含 selection data
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

def plot_loss_distribution(result_dir, output_dir="./figures"):
    """绘制不同算法的被选中 client loss 分布对比图"""
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
    all_files = scan_result_files(result_dir, prefer_single_run=True)
    
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
    
    # 收集所有算法的 loss 数据
    all_losses = {}
    all_labels = {}
    all_colors = {}
    all_metrics = {}
    
    for alg, file_info in algorithms_to_plot.items():
        filepath = file_info["filepath"]
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            processed_losses = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                if len(valid_losses) > 0:
                    processed_losses.extend(valid_losses.tolist())
            
            if len(processed_losses) > 0:
                style = ALGORITHM_STYLES.get(alg, {"label": alg, "color": "gray", "marker": "."})
                all_losses[alg] = processed_losses
                all_labels[alg] = style.get("label", alg)
                all_colors[alg] = style.get("color", "gray")
                all_metrics[alg] = style.get("metric", "Loss")
                logger.info(f"✓ Loaded {alg}: {len(processed_losses)} loss values")
        else:
            logger.warning(f"✗ No selection data for {alg}")
    
    if len(all_losses) == 0:
        logger.error("No valid selection data found!")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: Loss 分布直方图对比
    ax1 = axes[0]
    # 确保有数据才绘制
    if len(all_losses) > 0:
        all_loss_values = [loss for losses in all_losses.values() for loss in losses]
        if len(all_loss_values) > 0:
            bins = np.linspace(min(all_loss_values), max(all_loss_values), 30)
        else:
            bins = 30
    else:
        bins = 30
    
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
    
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel(f'Selected Client {all_metrics[list(all_losses.keys())[0]]}')
    ax2.set_title('Box Plot of Selected Client Losses')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = f"{output_dir}/{dataset}_{model}_selection_loss_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()

def plot_loss_over_rounds(result_dir, output_dir="./figures"):
    """绘制被选中 client loss 随轮次的变化"""
    os.makedirs(output_dir, exist_ok=True)
    
    dir_name = os.path.basename(result_dir.rstrip('/'))
    dir_info = parse_result_dir(dir_name)
    
    if dir_info is None:
        logger.error(f"Cannot parse directory name: {dir_name}")
        return
    
    model = dir_info.get('model_name', 'unknown')
    dataset = dir_info.get('dataset', 'unknown')
    
    all_files = scan_result_files(result_dir, prefer_single_run=True)
    
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alg, file_info in algorithms_to_plot.items():
        filepath = file_info["filepath"]
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            round_means = []
            round_stds = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                if len(valid_losses) > 0:
                    round_means.append(np.mean(valid_losses))
                    round_stds.append(np.std(valid_losses))
            
            if len(round_means) > 0:
                style = ALGORITHM_STYLES.get(alg, {"label": alg, "color": "gray", "marker": "."})
                rounds = np.arange(len(round_means))
                round_means = np.array(round_means)
                round_stds = np.array(round_stds)
                
                ax.plot(rounds, round_means, label=style["label"], 
                       color=style["color"], marker=style["marker"], 
                       markersize=6, linewidth=1.5, markevery=max(1, len(rounds)//20))
                ax.fill_between(rounds, round_means - round_stds, round_means + round_stds,
                               alpha=0.2, color=style["color"])
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Mean Selected Client Loss')
    ax.set_title(f'{dataset} - {model.upper()} - Selected Client Loss Over Rounds')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = f"{output_dir}/{dataset}_{model}_selection_loss_over_rounds.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()

def print_statistics_summary(result_dir):
    """打印统计摘要"""
    dir_name = os.path.basename(result_dir.rstrip('/'))
    dir_info = parse_result_dir(dir_name)
    
    if dir_info is None:
        logger.error(f"Cannot parse directory name: {dir_name}")
        return
    
    model = dir_info.get('model_name', 'unknown')
    dataset = dir_info.get('dataset', 'unknown')
    
    all_files = scan_result_files(result_dir, prefer_single_run=True)
    
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
    
    stats_data = []
    
    for alg, file_info in algorithms_to_plot.items():
        filepath = file_info["filepath"]
        selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
        
        if selected_losses is not None:
            all_losses = []
            for round_losses in selected_losses:
                valid_losses = round_losses[~np.isnan(round_losses)]
                all_losses.extend(valid_losses.tolist())
            
            if len(all_losses) > 0:
                all_losses = np.array(all_losses)
                style = ALGORITHM_STYLES.get(alg, {"label": alg})
                stats_data.append({
                    "Algorithm": style.get("label", alg),
                    "Mean": np.mean(all_losses),
                    "Std": np.std(all_losses),
                    "Median": np.median(all_losses),
                    "Min": np.min(all_losses),
                    "Max": np.max(all_losses),
                })
    
    if len(stats_data) == 0:
        logger.error("No valid data found!")
        return
    
    logger.info("\n" + "="*80)
    logger.info(f"Selected Client Loss Statistics Summary ({dataset}, {model})")
    logger.info("="*80)
    logger.info(f"{'Algorithm':<20} {'Mean':<12} {'Std':<12} {'Median':<12} {'Min':<12} {'Max':<12}")
    logger.info("-"*80)
    
    for stats in stats_data:
        logger.info(f"{stats['Algorithm']:<20} {stats['Mean']:<12.4f} {stats['Std']:<12.4f} "
              f"{stats['Median']:<12.4f} {stats['Min']:<12.4f} {stats['Max']:<12.4f}")
    
    logger.info("="*80)

def plot_round_specific_comparison(result_dir, target_rounds=[10, 50], 
                                    algorithms=["MESA", "pFedMe"], 
                                    output_dir="./figures", 
                                    include_all_clients=False):
    """绘制特定轮次的 Loss 分布对比图（小提琴图/箱线图）
    
    Args:
        result_dir: 结果目录路径
        target_rounds: 要对比的轮次列表（0-indexed，例如 [9, 49] 表示第10轮和第50轮）
        algorithms: 要对比的算法列表，默认 ["MESA", "pFedMe"]
        output_dir: 输出目录
        include_all_clients: 是否包含所有client的loss作为背景分布
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dir_name = os.path.basename(result_dir.rstrip('/'))
    dir_info = parse_result_dir(dir_name)
    
    if dir_info is None:
        logger.error(f"Cannot parse directory name: {dir_name}")
        return
    
    model = dir_info.get('model_name', 'unknown')
    dataset = dir_info.get('dataset', 'unknown')
    
    all_files = scan_result_files(result_dir, prefer_single_run=True)
    
    # 收集指定算法的数据
    algorithm_data = {}
    for alg_name in algorithms:
        # 优先使用个性化模型（pFedMe除外，因为它本身就是个性化）
        key_p = f"{alg_name}_p"
        key = alg_name
        
        file_info = None
        if key_p in all_files:
            file_info = all_files[key_p]
        elif key in all_files:
            file_info = all_files[key]
        
        if file_info:
            filepath = file_info["filepath"]
            selected_losses, selected_indices, rs_glob_acc, wall_clock_times = read_h5_file_with_selection(filepath)
            
            if selected_losses is not None:
                algorithm_data[alg_name] = {
                    "losses": selected_losses,
                    "indices": selected_indices,
                    "style": ALGORITHM_STYLES.get(alg_name, {"label": alg_name, "color": "gray"})
                }
                logger.info(f"✓ Loaded {alg_name}: {len(selected_losses)} rounds")
            else:
                logger.warning(f"✗ No selection data for {alg_name}")
        else:
            logger.warning(f"✗ File not found for {alg_name}")
    
    if len(algorithm_data) == 0:
        logger.error("No valid algorithm data found!")
        return
    
    # 获取所有client的loss（如果需要）
    all_clients_losses = None
    if include_all_clients:
        # 需要从某个算法的文件中读取所有client的loss
        # 这里假设我们可以从第一个算法的数据中获取
        # 实际上，我们需要在训练时记录所有client的loss
        logger.warning("include_all_clients=True is not yet implemented. Skipping background distribution.")
    
    # 为每个目标轮次创建图表
    for target_round in target_rounds:
        # 检查轮次是否有效
        max_rounds = min(len(data["losses"]) for data in algorithm_data.values())
        if target_round >= max_rounds:
            logger.warning(f"Round {target_round} is out of range (max: {max_rounds-1}). Skipping.")
            continue
        
        # 收集该轮次的数据
        round_data = []
        for alg_name, data in algorithm_data.items():
            round_losses = data["losses"][target_round]
            valid_losses = round_losses[~np.isnan(round_losses)]
            
            if len(valid_losses) > 0:
                for loss in valid_losses:
                    round_data.append({
                        'Loss': float(loss),
                        'Method': data["style"]["label"]
                    })
        
        if len(round_data) == 0:
            logger.warning(f"No valid data for round {target_round}. Skipping.")
            continue
        
        # 创建DataFrame
        df = pd.DataFrame(round_data)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1: 小提琴图（推荐）
        ax1 = axes[0]
        methods = df['Method'].unique()
        # 创建方法到颜色的映射
        method_to_color = {}
        for alg_name, data in algorithm_data.items():
            method_to_color[data["style"]["label"]] = data["style"]["color"]
        colors = [method_to_color.get(method, "gray") for method in methods]
        
        # 使用seaborn绘制小提琴图
        sns.violinplot(x='Method', y='Loss', data=df, inner="quartile", 
                      palette=colors, ax=ax1)
        
        # 添加抖动点展示具体样本
        sns.stripplot(x='Method', y='Loss', data=df, color='black', 
                     size=3, alpha=0.3, ax=ax1, jitter=True)
        
        ax1.set_title(f'Loss Distribution of Selected Clients (Round {target_round+1})', fontsize=14)
        ax1.set_ylabel('Local Training Loss', fontsize=12)
        ax1.set_xlabel('')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 子图2: 箱线图（传统方法）
        ax2 = axes[1]
        box_data = [df[df['Method'] == method]['Loss'].values for method in methods]
        bp = ax2.boxplot(box_data, labels=methods, patch_artist=True, 
                        showmeans=True, meanline=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title(f'Box Plot of Selected Client Losses (Round {target_round+1})', fontsize=14)
        ax2.set_ylabel('Local Training Loss', fontsize=12)
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息文本
        stats_text = []
        for method in methods:
            method_losses = df[df['Method'] == method]['Loss'].values
            mean_loss = np.mean(method_losses)
            median_loss = np.median(method_losses)
            stats_text.append(f"{method}: Mean={mean_loss:.4f}, Median={median_loss:.4f}")
        
        stats_str = "\n".join(stats_text)
        fig.text(0.5, 0.02, stats_str, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        output_path = f"{output_dir}/{dataset}_{model}_round_{target_round+1}_loss_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
        plt.close()
        
        # 打印统计信息
        logger.info(f"\nRound {target_round+1} Statistics:")
        logger.info("-" * 60)
        for method in methods:
            method_losses = df[df['Method'] == method]['Loss'].values
            logger.info(f"{method}:")
            logger.info(f"  Mean: {np.mean(method_losses):.4f}")
            logger.info(f"  Median: {np.median(method_losses):.4f}")
            logger.info(f"  Std: {np.std(method_losses):.4f}")
            logger.info(f"  Min: {np.min(method_losses):.4f}")
            logger.info(f"  Max: {np.max(method_losses):.4f}")
            logger.info(f"  Count: {len(method_losses)}")
        logger.info("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot selected client loss distribution')
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to result directory")
    parser.add_argument("--output", type=str, default="./figures", help="Output directory")
    parser.add_argument("--plot_type", type=str, default="all", 
                       choices=["all", "distribution", "over_rounds", "stats", "round_specific"],
                       help="Type of plot to generate")
    parser.add_argument("--target_rounds", type=int, nargs='+', default=[10, 50],
                       help="Target rounds for round-specific comparison (1-indexed, e.g., 10 50)")
    parser.add_argument("--algorithms", type=str, nargs='+', default=["MESA", "pFedMe"],
                       help="Algorithms to compare (default: MESA pFedMe)")
    parser.add_argument("--include_all_clients", action="store_true",
                       help="Include all clients' loss as background distribution (not yet implemented)")
    
    args = parser.parse_args()
    
    # Convert 1-indexed rounds to 0-indexed
    target_rounds = [r - 1 for r in args.target_rounds]
    
    if args.plot_type in ["all", "distribution"]:
        plot_loss_distribution(args.result_dir, args.output)
    
    if args.plot_type in ["all", "over_rounds"]:
        plot_loss_over_rounds(args.result_dir, args.output)
    
    if args.plot_type in ["all", "stats"]:
        print_statistics_summary(args.result_dir)
    
    if args.plot_type in ["all", "round_specific"]:
        plot_round_specific_comparison(
            args.result_dir, 
            target_rounds=target_rounds,
            algorithms=args.algorithms,
            output_dir=args.output,
            include_all_clients=args.include_all_clients
        )
