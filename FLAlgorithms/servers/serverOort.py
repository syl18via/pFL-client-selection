import numpy as np
import time
from FLAlgorithms.servers.serverpFedMe import pFedMe

class Oort(pFedMe):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, K, personal_learning_rate, times)
        
        # Oort 需要维护历史 Utility
        self.client_utilities = np.zeros(len(self.users)) + 1e-5  # 初始化防止除零
        self.step_window = 10  # Oort 的滑动窗口，这里简化为只存最新的
        self.exploration = 0.1 # Oort 的 epsilon-greedy

    def train(self, save_model=False, current_time=0, total_times=1):
        # 必须重写 train，因为 Oort 需要在训练后收集 Loss 来更新 Utility
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()  # Start timing this round
            
            print(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (Oort)-------------")
            self.send_parameters()
            self.evaluate()

            # === Oort Selection Step ===
            # 1. 计算概率: p_i propto Utility_i
            # 简单实现：Utility = Loss^2 (Oort 论文中的统计 Utility)
            # 加上 epsilon 探索防止饿死
            total_util = np.sum(self.client_utilities)
            if total_util == 0:
                probs = np.ones(len(self.users)) / len(self.users)
            else:
                probs = (1 - self.exploration) * (self.client_utilities / total_util) + \
                        (self.exploration / len(self.users))
            probs = probs / np.sum(probs)

            # 2. 采样
            all_indices = np.arange(len(self.users))
            selected_indices = np.random.choice(all_indices, self.num_users, p=probs, replace=False)
            self.selected_users = [self.users[i] for i in selected_indices]

            # Record selected client losses before training (for Effectiveness of Selection analysis)
            # Note: We record loss before training to show selection criteria
            self.record_selected_client_losses(selected_indices.tolist())

            # 3. 训练 & 更新 Utility
            for user_idx in selected_indices:
                user = self.users[user_idx]
                # 先训练
                user.train(self.local_epochs)
                
                # 训练后获取 loss 来更新 Utility
                _, loss, _ = user.train_error_and_loss()
                # 将 tensor 转换为 Python float
                if hasattr(loss, 'item'):
                    loss = loss.detach().item()
                
                # Oort Update: 更新该用户的 Utility
                # 这里用 Loss^2 作为梯度范数的近似 (Oort 论文做法)
                self.client_utilities[user_idx] = loss ** 2

            # Evaluate personalized model
            self.evaluate_personalized_model()
            self.persionalized_aggregate_parameters()

            # Record time for this round
            round_time = time.time() - round_start_time
            self.record_time(round_time)

        self.save_results()
        if save_model:
            self.save_model()