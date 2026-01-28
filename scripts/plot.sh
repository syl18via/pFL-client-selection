#!/bin/bash
# 统一绘图脚本
# 用法: bash scripts/plot.sh <result_dir>
# 示例: bash scripts/plot.sh results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs

set -e

if [ -z "$1" ]; then
    echo "用法: bash scripts/plot.sh <result_dir>"
    echo "示例: bash scripts/plot.sh results/dnn_Mnist_5u_800r_5t_0.005lr_1.0beta_15lam_20bs"
    exit 1
fi

RESULT_DIR="$1"

if [ ! -d "$RESULT_DIR" ]; then
    echo "错误: 目录不存在: $RESULT_DIR"
    exit 1
fi

echo "=========================================="
echo "绘图目录: $RESULT_DIR"
echo "=========================================="

echo ""
echo "[1/3] 绘制准确率对比图..."
python scripts/plot_comparison.py --result_dir "$RESULT_DIR"

echo ""
echo "[2/3] 绘制时间对比图..."
python scripts/plot_time_comparison.py --result_dir "$RESULT_DIR"

echo ""
echo "[3/3] 绘制选择有效性分析图..."
python scripts/plot_selection_effectiveness.py --result_dir "$RESULT_DIR"

echo ""
echo "=========================================="
echo "所有图表已生成，保存在 ./figures/ 目录下"
echo "=========================================="
