#!/bin/bash
# MNIST 数据集对比实验脚本
# 运行 MESA, Oort_FedAvg, PoC_FedAvg, HiCS_FedAvg, pFedMe, FedAvg, PerAvg 算法
# 
# 注意：
# - 本脚本只运行 FedAvg 框架版本（标准训练）
# - Oort/PoC/HiCS 的 pFedMe 版本已被注释掉
# - 如需运行 pFedMe 版本，请取消注释相应的命令

set -e  # 出错时停止

# 终止所有与 "python3 main.py" 相关的进程（如有在运行）
PIDS=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$PIDS" ]; then
    echo "Killing processes: $PIDS"
    kill -9 $PIDS
fi

# ============ 创建日志目录 ============
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="mnist"
LOG_DIR="logs/${TIMESTAMP}-${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"
echo "Log directory: $LOG_DIR"

# ============ 配置参数 ============
DATASET="Mnist"
BATCH_SIZE=20
NUM_GLOBAL_ITERS=800
LOCAL_EPOCHS=20
NUM_USERS=5
TIMES=1
GPU=0

# pFedMe/MESA/Oort/PoC 专用参数
K=5
LAMDA=15

# ============ 非凸场景 (DNN) ============
echo "=========================================="
echo "Running MNIST Non-Convex (DNN) Experiments"
echo "=========================================="

MODEL="dnn"
LR=0.005
PERSONAL_LR=0.09
BETA=1

# MESA (你的算法)
echo "[1/7] Running MESA... (log: $LOG_DIR/dnn_MESA.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm MESA \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_MESA.log" 2>&1

# Oort (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[2/7] Running Oort (pFedMe)... (log: $LOG_DIR/dnn_Oort.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm Oort \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/dnn_Oort.log" 2>&1

# Oort (FedAvg framework)
echo "[2/7] Running Oort (FedAvg)... (log: $LOG_DIR/dnn_Oort_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm Oort --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_Oort_FedAvg.log" 2>&1

# PoC (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[3/7] Running PoC (pFedMe)... (log: $LOG_DIR/dnn_PoC.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm PoC \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/dnn_PoC.log" 2>&1

# PoC (FedAvg framework)
echo "[3/7] Running PoC (FedAvg)... (log: $LOG_DIR/dnn_PoC_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm PoC --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_PoC_FedAvg.log" 2>&1

# HiCS (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[4/7] Running HiCS (pFedMe)... (log: $LOG_DIR/dnn_HiCS.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm HiCS \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/dnn_HiCS.log" 2>&1

# HiCS (FedAvg framework)
echo "[4/7] Running HiCS (FedAvg)... (log: $LOG_DIR/dnn_HiCS_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm HiCS --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_HiCS_FedAvg.log" 2>&1

# pFedMe
echo "[5/7] Running pFedMe... (log: $LOG_DIR/dnn_pFedMe.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm pFedMe \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_pFedMe.log" 2>&1

# FedAvg
echo "[6/7] Running FedAvg... (log: $LOG_DIR/dnn_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm FedAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_FedAvg.log" 2>&1

# PerAvg
echo "[7/7] Running PerAvg... (log: $LOG_DIR/dnn_PerAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm PerAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/dnn_PerAvg.log" 2>&1

echo "=========================================="
echo "MNIST Non-Convex Experiments Completed!"
echo "=========================================="

# ============ 凸场景 (MCLR) ============
echo ""
echo "=========================================="
echo "Running MNIST Convex (MCLR) Experiments"
echo "=========================================="

MODEL="mclr"
LR=0.005
PERSONAL_LR=0.1
BETA=1

# MESA
echo "[1/7] Running MESA (Convex)... (log: $LOG_DIR/mclr_MESA.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm MESA \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_MESA.log" 2>&1

# Oort (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[2/7] Running Oort (Convex, pFedMe)... (log: $LOG_DIR/mclr_Oort.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm Oort \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/mclr_Oort.log" 2>&1

# Oort (FedAvg framework)
echo "[2/7] Running Oort (Convex, FedAvg)... (log: $LOG_DIR/mclr_Oort_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm Oort --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_Oort_FedAvg.log" 2>&1

# PoC (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[3/7] Running PoC (Convex, pFedMe)... (log: $LOG_DIR/mclr_PoC.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm PoC \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/mclr_PoC.log" 2>&1

# PoC (FedAvg framework)
echo "[3/7] Running PoC (Convex, FedAvg)... (log: $LOG_DIR/mclr_PoC_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm PoC --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_PoC_FedAvg.log" 2>&1

# HiCS (pFedMe framework - COMMENTED OUT, only running FedAvg version)
# echo "[4/7] Running HiCS (Convex, pFedMe)... (log: $LOG_DIR/mclr_HiCS.log)"
# nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
#     --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
#     --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
#     --local_epochs $LOCAL_EPOCHS --algorithm HiCS \
#     --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
#     > "$LOG_DIR/mclr_HiCS.log" 2>&1

# HiCS (FedAvg framework)
echo "[4/7] Running HiCS (Convex, FedAvg)... (log: $LOG_DIR/mclr_HiCS_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm HiCS --framework fedavg \
    --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_HiCS_FedAvg.log" 2>&1

# pFedMe
echo "[5/7] Running pFedMe (Convex)... (log: $LOG_DIR/mclr_pFedMe.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm pFedMe \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_pFedMe.log" 2>&1

# FedAvg
echo "[6/7] Running FedAvg (Convex)... (log: $LOG_DIR/mclr_FedAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm FedAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_FedAvg.log" 2>&1

# PerAvg
echo "[7/7] Running PerAvg (Convex)... (log: $LOG_DIR/mclr_PerAvg.log)"
nohup python3 -u main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm PerAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU \
    > "$LOG_DIR/mclr_PerAvg.log" 2>&1

echo "=========================================="
echo "All MNIST Experiments Completed!"
echo "Results saved in ./results/"
echo "Logs saved in $LOG_DIR"
echo "=========================================="
