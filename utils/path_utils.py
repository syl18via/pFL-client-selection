"""
结果文件路径构建工具
统一管理 results 目录结构和文件名格式
"""

import os

# 需要额外参数 K 和 personal_learning_rate 的算法
PFEDME_BASED_ALGORITHMS = ["pFedMe", "pFedMe_p"]


def build_result_dir(model_name, dataset, num_users, num_glob_iters, total_times,
                     learning_rate, beta, lamda, batch_size, base_dir="./results"):
    """构建结果目录路径
    
    格式: {base_dir}/{model}_{dataset}_{num_users}u_{num_glob_iters}r_{total_times}t_{learning_rate}lr_{beta}beta_{lamda}lam_{batch_size}bs
    
    Args:
        model_name: 模型名称 (e.g., "dnn", "cnn", "mclr")
        dataset: 数据集名称 (e.g., "Mnist", "Cifar10")
        num_users: 每轮选择的用户数
        num_glob_iters: 全局迭代轮数
        total_times: 实验总次数 (e.g., 5 表示运行5次实验)
        learning_rate: 学习率
        beta: beta参数
        lamda: lambda参数
        batch_size: 批大小
        base_dir: 基础目录，默认 "./results"
    
    Returns:
        str: 结果目录路径
    """
    return f"{base_dir}/{model_name}_{dataset}_{num_users}u_{num_glob_iters}r_{total_times}t_{learning_rate}lr_{beta}beta_{lamda}lam_{batch_size}bs"


def build_result_filename(algorithm, local_epochs, current_time, K=None, personal_learning_rate=None, 
                          personalized=False):
    """构建结果文件名
    
    格式: {algorithm}[_p]_{local_epochs}e[_{K}_{personal_learning_rate}][_{current_time}|_avg].h5
    
    Args:
        algorithm: 算法名称 (e.g., "pFedMe", "FedAvg")
        local_epochs: 本地训练轮数
        current_time: 当前实验次数 (int: 0, 1, 2, ... 或 "avg" 表示平均结果)
        K: pFedMe系列算法的K参数
        personal_learning_rate: pFedMe系列算法的个性化学习率
        personalized: 是否为个性化模型结果 (会在算法名后加 "_p")
    
    Returns:
        str: 文件名 (含 .h5 后缀)
    """
    # 构建算法名部分
    alg_name = f"{algorithm}_p" if personalized else algorithm
    
    # 基础文件名
    filename = f"{alg_name}_{local_epochs}e"
    
    # pFedMe系列算法需要额外参数
    if algorithm in PFEDME_BASED_ALGORITHMS and K is not None and personal_learning_rate is not None:
        filename = f"{filename}_{K}_{personal_learning_rate}"
    
    # 添加当前次数或平均标记
    if isinstance(current_time, str) and current_time == "avg":
        filename = f"{filename}_avg"
    else:
        filename = f"{filename}_{current_time}"
    
    return f"{filename}.h5"


def build_result_filepath(model_name, dataset, num_users, num_glob_iters, total_times, current_time,
                          learning_rate, beta, lamda, batch_size,
                          algorithm, local_epochs, K=None, personal_learning_rate=None,
                          personalized=False, base_dir="./results"):
    """构建完整的结果文件路径
    
    Args:
        model_name: 模型名称
        dataset: 数据集名称
        num_users: 每轮选择的用户数
        num_glob_iters: 全局迭代轮数
        total_times: 实验总次数 (用于目录名)
        current_time: 当前实验次数 (int 或 "avg"，用于文件名)
        learning_rate: 学习率
        beta: beta参数
        lamda: lambda参数
        batch_size: 批大小
        algorithm: 算法名称
        local_epochs: 本地训练轮数
        K: pFedMe系列算法的K参数
        personal_learning_rate: pFedMe系列算法的个性化学习率
        personalized: 是否为个性化模型结果
        base_dir: 基础目录
    
    Returns:
        str: 完整的文件路径
    """
    result_dir = build_result_dir(
        model_name, dataset, num_users, num_glob_iters, total_times,
        learning_rate, beta, lamda, batch_size, base_dir
    )
    filename = build_result_filename(
        algorithm, local_epochs, current_time, K, personal_learning_rate, personalized
    )
    return os.path.join(result_dir, filename)


def parse_result_dir(dir_name):
    """解析结果目录名，提取参数
    
    Args:
        dir_name: 目录名 (e.g., "dnn_Mnist_5u_100r_5t_0.01lr_1.0beta_15lam_20bs")
    
    Returns:
        dict: 解析出的参数字典，解析失败返回 None
              total_times: 实验总次数
    """
    try:
        parts = dir_name.split('_')
        # 预期格式: model_dataset_Xu_Xr_Xt_Xlr_Xbeta_Xlam_Xbs
        params = {
            'model_name': parts[0],
            'dataset': parts[1],
        }
        
        for part in parts[2:]:
            if part.endswith('u'):
                params['num_users'] = int(part[:-1])
            elif part.endswith('r'):
                params['num_glob_iters'] = int(part[:-1])
            elif part.endswith('t'):
                params['total_times'] = int(part[:-1])
            elif part.endswith('lr'):
                params['learning_rate'] = float(part[:-2])
            elif part.endswith('beta'):
                params['beta'] = float(part[:-4])
            elif part.endswith('lam'):
                params['lamda'] = float(part[:-3])
            elif part.endswith('bs'):
                params['batch_size'] = int(part[:-2])
        
        return params
    except (IndexError, ValueError):
        return None


def parse_result_filename(filename):
    """解析结果文件名，提取参数
    
    Args:
        filename: 文件名 (e.g., "pFedMe_20e_5_0.01_0.h5", "FedAvg_20e_0.h5", "pFedMe_20e_5_0.01_avg.h5")
    
    Returns:
        dict: 解析出的参数字典，解析失败返回 None
              current_time: "avg" 表示平均文件，int 表示当前次数
    """
    try:
        # 去掉 .h5 后缀
        name = filename[:-3] if filename.endswith('.h5') else filename
        
        # 检查是否为平均文件
        is_avg = name.endswith('_avg')
        if is_avg:
            name = name[:-4]  # 去掉 _avg 后缀
        
        parts = name.split('_')
        
        # 检查是否为个性化模型
        personalized = False
        if len(parts) > 1 and parts[1] == 'p':
            personalized = True
            algorithm = parts[0]
            parts = [algorithm] + parts[2:]
        else:
            algorithm = parts[0]
        
        params = {
            'algorithm': algorithm,
            'personalized': personalized,
        }
        
        # 解析 local_epochs 和后续参数
        for i, part in enumerate(parts[1:], 1):
            if part.endswith('e'):
                params['local_epochs'] = int(part[:-1])
                remaining = parts[i+1:]
                
                # 解析剩余部分: 可能是 [K, personal_lr, current_time] 或 [current_time]
                if is_avg:
                    params['current_time'] = "avg"
                    if len(remaining) >= 2:
                        params['K'] = int(remaining[0])
                        params['personal_learning_rate'] = float(remaining[1])
                else:
                    # 最后一个是 current_time
                    if len(remaining) >= 1:
                        if len(remaining) >= 3:
                            # K, personal_lr, current_time
                            params['K'] = int(remaining[0])
                            params['personal_learning_rate'] = float(remaining[1])
                            params['current_time'] = int(remaining[2])
                        else:
                            # 只有 current_time
                            params['current_time'] = int(remaining[-1])
                break
        
        return params
    except (IndexError, ValueError):
        return None
