#!/usr/bin/env python3
"""
MNIST 数据集对比实验脚本
支持多GPU并行运行，每块GPU同时只运行一个任务
"""

import subprocess
import os
import time
from datetime import datetime

# ============ 配置参数 ============
NUM_GPUS = 8

DATASET = "Mnist"
BATCH_SIZE = 20
NUM_GLOBAL_ITERS = 10
LOCAL_EPOCHS = 20
NUM_USERS = 5
TIMES = 3

# pFedMe/MESA/Oort/PoC 专用参数
K = 5
LAMDA = 15

# ============ 创建日志目录 ============
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/{TIMESTAMP}-mnist"
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Log directory: {LOG_DIR}")

# ============ 定义所有任务 (name, log_file, command) ============
# {gpu} 会在运行时被替换为实际的 GPU ID

TASKS = []
TASKS_TODO = [
    "DNN-MESA",
    "DNN-HiCS",
    "DNN-Oort",
    "DNN-PoC",
    "DNN-pFedMe",
    "DNN-FedAvg",
    "DNN-PerAvg",
    "MCLR-MESA",
    "MCLR-HiCS",
    "MCLR-Oort",
    "MCLR-PoC",
    "MCLR-pFedMe",
    "MCLR-FedAvg",
    "MCLR-PerAvg",
]

# ============ 非凸场景 (DNN) ============
MODEL = "dnn"
LR = 0.005
PERSONAL_LR = 0.09
BETA = 1

# MESA
if "DNN-MESA" in TASKS_TODO:
    TASKS.append(("DNN-MESA", "dnn_MESA.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm MESA \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# HiCS
if "DNN-HiCS" in TASKS_TODO:
    TASKS.append(("DNN-HiCS", "dnn_HiCS.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm HiCS \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# Oort
if "DNN-Oort" in TASKS_TODO:
    TASKS.append(("DNN-Oort", "dnn_Oort.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm Oort \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# PoC
if "DNN-PoC" in TASKS_TODO:
    TASKS.append(("DNN-PoC", "dnn_PoC.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm PoC \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# HiCS
TASKS.append(("DNN-HiCS", "dnn_HiCS.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm HiCS \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# pFedMe
if "DNN-pFedMe" in TASKS_TODO:
    TASKS.append(("DNN-pFedMe", "dnn_pFedMe.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm pFedMe \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# FedAvg
if "DNN-FedAvg" in TASKS_TODO:
    TASKS.append(("DNN-FedAvg", "dnn_FedAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta {BETA} --lamda {LAMDA} \
    --num_global_iters {NUM_GLOBAL_ITERS} --local_epochs {LOCAL_EPOCHS} \
    --algorithm FedAvg --numusers {NUM_USERS} --times {TIMES} --gpu {{gpu}}
"""))

# PerAvg
if "DNN-PerAvg" in TASKS_TODO:
    TASKS.append(("DNN-PerAvg", "dnn_PerAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta 0.001 --lamda {LAMDA} \
    --num_global_iters {NUM_GLOBAL_ITERS} --local_epochs {LOCAL_EPOCHS} \
    --algorithm PerAvg --numusers {NUM_USERS} --times {TIMES} --gpu {{gpu}}
"""))

# ============ 凸场景 (MCLR) ============
MODEL = "mclr"
LR = 0.005
PERSONAL_LR = 0.1
BETA = 1

# MESA
if "MCLR-MESA" in TASKS_TODO:
    TASKS.append(("MCLR-MESA", "mclr_MESA.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm MESA \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# HiCS
if "MCLR-HiCS" in TASKS_TODO:
    TASKS.append(("MCLR-HiCS", "mclr_HiCS.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm HiCS \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# Oort
if "MCLR-Oort" in TASKS_TODO:
    TASKS.append(("MCLR-Oort", "mclr_Oort.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm Oort \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# PoC
if "MCLR-PoC" in TASKS_TODO:
    TASKS.append(("MCLR-PoC", "mclr_PoC.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm PoC \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# HiCS
TASKS.append(("MCLR-HiCS", "mclr_HiCS.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm HiCS \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# pFedMe
if "MCLR-pFedMe" in TASKS_TODO:
    TASKS.append(("MCLR-pFedMe", "mclr_pFedMe.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm pFedMe \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# FedAvg
if "MCLR-FedAvg" in TASKS_TODO:
    TASKS.append(("MCLR-FedAvg", "mclr_FedAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta {BETA} --lamda {LAMDA} \
    --num_global_iters {NUM_GLOBAL_ITERS} --local_epochs {LOCAL_EPOCHS} \
    --algorithm FedAvg --numusers {NUM_USERS} --times {TIMES} --gpu {{gpu}}
"""))

# PerAvg
if "MCLR-PerAvg" in TASKS_TODO:
    TASKS.append(("MCLR-PerAvg", "mclr_PerAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta 0.001 --lamda {LAMDA} \
    --num_global_iters {NUM_GLOBAL_ITERS} --local_epochs {LOCAL_EPOCHS} \
    --algorithm PerAvg --numusers {NUM_USERS} --times {TIMES} --gpu {{gpu}}
"""))


# ============ GPU 调度器 ============
class GPUScheduler:
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        # gpu_id -> (name, process, log_file) or None
        self.running = [None] * num_gpus
        self.completed = []
        self.failed = []
    
    def get_free_gpu(self):
        for i, task in enumerate(self.running):
            if task is None:
                return i
        return None
    
    def start_task(self, name: str, log_file: str, cmd: str, gpu_id: int):
        log_path = os.path.join(LOG_DIR, log_file)
        cmd = cmd.format(gpu=gpu_id).strip()
        
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
            )
        
        self.running[gpu_id] = (name, proc, log_path)
        print(f"[GPU {gpu_id}] Started: {name} -> {log_path}")
    
    def check_completed(self):
        for i, task in enumerate(self.running):
            if task is not None:
                name, proc, log_path = task
                ret = proc.poll()
                if ret is not None:
                    if ret == 0:
                        print(f"[GPU {i}] Completed: {name}")
                        self.completed.append(name)
                    else:
                        print(f"[GPU {i}] Failed (exit {ret}): {name}")
                        self.failed.append(name)
                    self.running[i] = None
    
    def has_running_tasks(self):
        return any(t is not None for t in self.running)
    
    def run_all(self, tasks):
        task_queue = list(tasks)
        total = len(task_queue)
        print(f"Total tasks: {total}, GPUs: {self.num_gpus}")
        print("=" * 50)
        
        while task_queue or self.has_running_tasks():
            self.check_completed()
            
            while task_queue:
                gpu_id = self.get_free_gpu()
                if gpu_id is None:
                    break
                name, log_file, cmd = task_queue.pop(0)
                self.start_task(name, log_file, cmd, gpu_id)
            
            time.sleep(2)
        
        print("=" * 50)
        print(f"All tasks completed!")
        print(f"  Successful: {len(self.completed)}")
        print(f"  Failed: {len(self.failed)}")
        if self.failed:
            print(f"  Failed tasks: {self.failed}")


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 50)
    print("Running MNIST Experiments (Multi-GPU)")
    print("=" * 50)
    
    scheduler = GPUScheduler(NUM_GPUS)
    scheduler.run_all(TASKS)
    
    print(f"Results saved in ./results/")
    print(f"Logs saved in {LOG_DIR}")
