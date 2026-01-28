# 实验需求修改清单

根据实验要求，以下是需要修改的地方：

## 1. 数据集支持

### 1.1 添加 CIFAR-100 支持
- [ ] 在 `utils/model_utils.py` 中添加 `read_cifar100_data()` 函数
- [ ] 在 `read_data()` 中添加 CIFAR-100 分支
- [ ] 在 `read_user_data()` 中添加 CIFAR-100 数据处理
- [ ] 在 `FLAlgorithms/trainmodel/models.py` 中添加 CIFAR-100 模型（如果需要）
- [ ] 创建 CIFAR-100 数据生成脚本（支持 Non-IID）

### 1.2 添加 Dirichlet Distribution (α=0.1) Non-IID 数据生成
- [ ] 创建 `data/Mnist/generate_niid_dirichlet.py` (α=0.1)
- [ ] 创建 `data/Cifar10/generate_niid_dirichlet.py` (α=0.1)
- [ ] 创建 `data/Cifar100/generate_niid_dirichlet.py` (α=0.1)
- [ ] 修改数据生成脚本，支持可配置的 α 参数

## 2. Baselines

### 2.1 实现 FedRep 算法
- [ ] 创建 `FLAlgorithms/servers/serverFedRep.py`
- [ ] 创建 `FLAlgorithms/users/userFedRep.py`
- [ ] 在 `main.py` 中添加 FedRep 支持
- [ ] FedRep 特点：先训练 representation，再训练 head

## 3. 时间记录（用于 Accuracy vs. Wall-clock Time）

### 3.1 在 Server 基类中添加时间记录
- [ ] 在 `serverbase.py` 的 `__init__` 中初始化 `self.wall_clock_times = []`
- [ ] 在 `train()` 方法开始时记录 `start_time = time.time()`
- [ ] 在每轮 `evaluate()` 后记录 `elapsed_time = time.time() - start_time`
- [ ] 将时间累积：`self.wall_clock_times.append(累计时间)`
- [ ] 在 `save_results()` 中保存时间数据到 h5 文件

### 3.2 修改所有 Server 子类
- [ ] `serverMESA.py`: 确保时间记录正确
- [ ] `serverOort.py`: 确保时间记录正确
- [ ] `serverPoC.py`: 确保时间记录正确（包括探测时间）
- [ ] `serverpFedMe.py`: 确保时间记录正确
- [ ] `serverFedRep.py`: 添加时间记录
- [ ] `serverHICS.py`: 确保时间记录正确

## 4. 记录被选中 Client 的 Loss 分布（用于 Effectiveness of Selection）

### 4.1 在 Server 基类中添加
- [ ] 在 `serverbase.py` 中添加 `self.selected_client_losses = []` (每轮记录)
- [ ] 在每轮选择后，记录被选中 client 的 loss：
  ```python
  selected_losses = []
  for user in self.selected_users:
      _, loss, _ = user.train_error_and_loss()
      selected_losses.append(loss.item())
  self.selected_client_losses.append(selected_losses)
  ```
- [ ] 在 `save_results()` 中保存到 h5 文件

### 4.2 特别处理 MESA
- [ ] MESA 已经计算了 proximal gap，可以直接使用
- [ ] 记录被选中 client 的 proximal gap 分布

## 5. 记录 Client 端计算耗时（用于 Overhead Analysis）

### 5.1 在 User 类中添加时间记录
- [ ] 在 `userbase.py` 的 `train()` 方法中添加：
  ```python
  import time
  start_time = time.time()
  # ... 训练代码 ...
  training_time = time.time() - start_time
  return training_time  # 或返回 (loss, training_time)
  ```

### 5.2 在 Server 中收集时间
- [ ] 在 `serverbase.py` 中添加 `self.client_training_times = []`
- [ ] 收集每个被选中 client 的训练时间
- [ ] 在 `save_results()` 中保存

### 5.3 特别处理 PoC
- [ ] PoC 需要额外记录探测时间（probe time）
- [ ] 区分完整训练时间和探测时间

## 6. Ablation Studies

### 6.1 Epsilon (探索率) 参数化
- [ ] 在 `serverMESA.py` 的 `__init__` 中添加 `epsilon` 参数
- [ ] 在 `main.py` 中添加 `--epsilon` 命令行参数（默认 0.1）
- [ ] 创建 ablation 实验脚本，测试不同 epsilon 值（0.0, 0.1, 0.2, 0.3, 0.5）

### 6.2 Staleness (历史信息过期) 机制
- [ ] 在 `serverMESA.py` 中添加 `staleness_window` 参数
- [ ] 修改 Scoreboard 更新逻辑：
  ```python
  # 只使用最近 staleness_window 轮的信息
  # 或者对历史分数进行衰减
  self.V[user_idx] = decay_factor * old_V + (1-decay_factor) * new_metric
  ```
- [ ] 在 `main.py` 中添加 `--staleness_window` 参数
- [ ] 创建 ablation 实验脚本

## 7. System Heterogeneity (模拟慢节点)

### 7.1 添加 Client 速度模拟
- [ ] 在 `userbase.py` 中添加 `speed_factor` 属性（0.1-1.0）
- [ ] 在训练时模拟延迟：
  ```python
  # 慢节点需要更多时间
  actual_time = base_time / speed_factor
  time.sleep(actual_time - base_time)  # 或使用更真实的模拟
  ```

### 7.2 在数据生成时分配速度
- [ ] 创建速度分配函数（例如：20% 慢节点，80% 正常节点）
- [ ] 在创建 User 时分配速度因子

### 7.3 在 Server 中考虑速度
- [ ] 选择策略可以考虑速度（但 MESA 主要基于难度）
- [ ] 记录每个 client 的实际训练时间（考虑速度）

## 8. 绘图脚本修改

### 8.1 Accuracy vs. Wall-clock Time
- [ ] 修改 `scripts/plot_comparison.py` 或创建新脚本
- [ ] 从 h5 文件读取 `wall_clock_times` 和 `rs_glob_acc`
- [ ] 绘制 Accuracy vs. Time 曲线

### 8.2 Effectiveness of Selection
- [ ] 创建新脚本 `scripts/plot_selection_effectiveness.py`
- [ ] 可视化被选中 client 的 loss 分布
- [ ] 对比不同算法的选择效果

### 8.3 Overhead Analysis
- [ ] 创建新脚本 `scripts/plot_overhead.py`
- [ ] 对比不同算法的计算耗时
- [ ] 特别标注 PoC 的探测开销

## 9. 实验脚本

### 9.1 创建主实验脚本
- [ ] `scripts/run_main_experiments.sh`: 运行所有主要实验
- [ ] `scripts/run_ablation.sh`: 运行 ablation studies
- [ ] 支持不同数据集（MNIST, CIFAR-10, CIFAR-100）
- [ ] 支持不同 Non-IID 程度（α=0.1）

### 9.2 更新现有脚本
- [ ] 更新 `run_mnist.sh` 支持 α 参数
- [ ] 更新 `run_cifar10.sh` 支持 α 参数
- [ ] 创建 `run_cifar100.sh`

## 10. 代码结构优化

### 10.1 统一时间记录接口
- [ ] 创建 `utils/timing_utils.py` 工具模块
- [ ] 统一时间记录格式

### 10.2 统一结果保存格式
- [ ] 确保所有算法保存相同格式的数据
- [ ] 包括：accuracy, loss, time, selected_losses, training_times

## 优先级排序

**高优先级（必须）：**
1. 时间记录（Accuracy vs. Time）
2. 被选中 client loss 记录
3. Epsilon 参数化
4. CIFAR-100 支持

**中优先级（重要）：**
5. FedRep 实现
6. Dirichlet Non-IID 数据生成
7. Client 计算耗时记录
8. System Heterogeneity

**低优先级（可选）：**
9. Staleness 机制
10. 绘图脚本优化
