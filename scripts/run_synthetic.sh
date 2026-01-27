#!/bin/bash
# Synthetic 数据集对比实验脚本
# 运行 MESA, Oort, PoC, pFedMe, FedAvg, PerAvg 算法

set -e  # 出错时停止

# ============ 配置参数 ============
DATASET="Synthetic"
BATCH_SIZE=20
NUM_GLOBAL_ITERS=600
LOCAL_EPOCHS=20
NUM_USERS=10
TIMES=10
GPU=0

# pFedMe/MESA/Oort/PoC 专用参数
K=5
LAMDA=20

# ============ 非凸场景 (DNN) ============
echo "=========================================="
echo "Running Synthetic Non-Convex (DNN) Experiments"
echo "=========================================="

MODEL="dnn"
LR=0.005
PERSONAL_LR=0.01
BETA=1

# MESA (你的算法)
echo "[1/6] Running MESA..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm MESA \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# Oort
echo "[2/6] Running Oort..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm Oort \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# PoC
echo "[3/6] Running PoC..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm PoC \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# pFedMe
echo "[4/6] Running pFedMe..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm pFedMe \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# FedAvg
echo "[5/6] Running FedAvg..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm FedAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU

# PerAvg
echo "[6/6] Running PerAvg..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta 0.001 --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm PerAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU

echo "=========================================="
echo "Synthetic Non-Convex Experiments Completed!"
echo "=========================================="

# ============ 凸场景 (MCLR) ============
echo ""
echo "=========================================="
echo "Running Synthetic Convex (MCLR) Experiments"
echo "=========================================="

MODEL="mclr"
LR=0.005
PERSONAL_LR=0.01
BETA=1

# MESA
echo "[1/6] Running MESA (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm MESA \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# Oort
echo "[2/6] Running Oort (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm Oort \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# PoC
echo "[3/6] Running PoC (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm PoC \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# pFedMe
echo "[4/6] Running pFedMe (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --personal_learning_rate $PERSONAL_LR \
    --beta $BETA --lamda $LAMDA --num_global_iters $NUM_GLOBAL_ITERS \
    --local_epochs $LOCAL_EPOCHS --algorithm pFedMe \
    --numusers $NUM_USERS --K $K --times $TIMES --gpu $GPU

# FedAvg
echo "[5/6] Running FedAvg (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta $BETA --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm FedAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU

# PerAvg
echo "[6/6] Running PerAvg (Convex)..."
python3 main.py --dataset $DATASET --model $MODEL --batch_size $BATCH_SIZE \
    --learning_rate $LR --beta 0.001 --lamda $LAMDA \
    --num_global_iters $NUM_GLOBAL_ITERS --local_epochs $LOCAL_EPOCHS \
    --algorithm PerAvg --numusers $NUM_USERS --times $TIMES --gpu $GPU

echo "=========================================="
echo "All Synthetic Experiments Completed!"
echo "Results saved in ./results/"
echo "=========================================="
