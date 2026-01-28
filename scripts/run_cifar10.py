#!/usr/bin/env python3
"""
CIFAR-10 数据集对比实验脚本
支持多GPU并行运行，每块GPU同时只运行一个任务
"""

import subprocess
import os
import time
from datetime import datetime

# ============ 配置参数 ============
NUM_GPUS = 8

DATASET = "Cifar10"
BATCH_SIZE = 20
NUM_GLOBAL_ITERS = 500
LOCAL_EPOCHS = 20
NUM_USERS = 5
TIMES = 3

# pFedMe/MESA/Oort/PoC 专用参数
K = 5
LAMDA = 15

# ============ 创建日志目录 ============
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/{TIMESTAMP}-cifar10"
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Log directory: {LOG_DIR}")

# ============ 定义所有任务 (name, log_file, command) ============
# {gpu} 会在运行时被替换为实际的 GPU ID

TASKS = []
TASKS_TODO = [
    "CNN-MESA",
    "CNN-HiCS",
    "CNN-Oort",
    "CNN-PoC",
    "CNN-pFedMe",
    "CNN-FedAvg",
    "CNN-PerAvg",
]

# ============ CNN 模型 (CIFAR-10 使用 CNN) ============
MODEL = "cnn"
LR = 0.01
PERSONAL_LR = 0.01
BETA = 1

# MESA
if "CNN-MESA" in TASKS_TODO:
    TASKS.append(("CNN-MESA", "cnn_MESA.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm MESA \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# HiCS
if "CNN-HiCS" in TASKS_TODO:
    TASKS.append(("CNN-HiCS", "cnn_HiCS.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm HiCS \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# Oort
if "CNN-Oort" in TASKS_TODO:
    TASKS.append(("CNN-Oort", "cnn_Oort.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm Oort \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# PoC
if "CNN-PoC" in TASKS_TODO:
    TASKS.append(("CNN-PoC", "cnn_PoC.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm PoC \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# pFedMe
if "CNN-pFedMe" in TASKS_TODO:
    TASKS.append(("CNN-pFedMe", "cnn_pFedMe.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --personal_learning_rate {PERSONAL_LR} \
    --beta {BETA} --lamda {LAMDA} --num_global_iters {NUM_GLOBAL_ITERS} \
    --local_epochs {LOCAL_EPOCHS} --algorithm pFedMe \
    --numusers {NUM_USERS} --K {K} --times {TIMES} --gpu {{gpu}}
"""))

# FedAvg
if "CNN-FedAvg" in TASKS_TODO:
    TASKS.append(("CNN-FedAvg", "cnn_FedAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta {BETA} --lamda {LAMDA} \
    --num_global_iters {NUM_GLOBAL_ITERS} --local_epochs {LOCAL_EPOCHS} \
    --algorithm FedAvg --numusers {NUM_USERS} --times {TIMES} --gpu {{gpu}}
"""))

# PerAvg
if "CNN-PerAvg" in TASKS_TODO:
    TASKS.append(("CNN-PerAvg", "cnn_PerAvg.log", f"""
python3 -u main.py --dataset {DATASET} --model {MODEL} --batch_size {BATCH_SIZE} \
    --learning_rate {LR} --beta {BETA} --lamda {LAMDA} \
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
    print("Running CIFAR-10 Experiments (Multi-GPU)")
    print("=" * 50)
    
    scheduler = GPUScheduler(NUM_GPUS)
    scheduler.run_all(TASKS)
    
    print(f"Results saved in ./results/")
    print(f"Logs saved in {LOG_DIR}")
