#!/bin/bash
# 运行所有数据集的对比实验
# 使用方法: bash scripts/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================"
echo "Starting All Experiments"
echo "Project Dir: $PROJECT_DIR"
echo "============================================"

# 检查数据是否已生成
if [ ! -f "data/Mnist/data/train/mnist_train.json" ]; then
    echo "Warning: MNIST data not found. Please run:"
    echo "  cd data/Mnist && python3 generate_niid_20users.py"
fi

if [ ! -f "data/Synthetic/data/train/train.json" ]; then
    echo "Warning: Synthetic data not found. Please run:"
    echo "  cd data/Synthetic && python3 generate_synthetic_05_05.py"
fi

# 运行各数据集实验
echo ""
echo ">>> Running MNIST experiments..."
bash scripts/run_mnist.sh

echo ""
echo ">>> Running Synthetic experiments..."
bash scripts/run_synthetic.sh

echo ""
echo ">>> Running CIFAR-10 experiments..."
bash scripts/run_cifar10.sh

echo ""
echo "============================================"
echo "All Experiments Completed!"
echo "Results saved in: $PROJECT_DIR/results/"
echo "============================================"
