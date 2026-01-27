import numpy as np
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

    def train(self, save_model=False):
        # 必须重写 train，因为 Oort 需要在训练后收集 Loss 来更新 Utility
        for glob_iter in range(self.num_glob_iters):
            print("-------------Oort Round number: ", glob_iter, " -------------")
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

            # 3. 训练 & 更新 Utility
            for user_idx in selected_indices:
                user = self.users[user_idx]
                # 注意：UserpFedMe.train 默认返回 LOSS (我们正好需要这个)
                loss, _ = user.train(self.local_epochs) 
                
                # Oort Update: 更新该用户的 Utility
                # 这里用 Loss^2 作为梯度范数的近似 (Oort 论文做法)
                self.client_utilities[user_idx] = loss ** 2

            self.persionalized_aggregate_parameters()

        self.save_results()
        if save_model:
            self.save_model()