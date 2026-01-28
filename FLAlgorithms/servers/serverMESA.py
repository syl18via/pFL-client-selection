import numpy as np
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.users.userMESA import UserMESA
from utils.model_utils import read_data, read_user_data

class MESA(pFedMe):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        # 调用父类初始化
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, K, personal_learning_rate, times)

        # === MESA 特有初始化 ===
        # 1. 重新生成 users 列表，使用 UserMESA 替代 UserpFedMe
        #    因为父类初始化时创建的是 UserpFedMe，我们需要替换成 UserMESA 以便能返回 metric
        self.users = []
        self.total_train_samples = 0
        data = read_data(dataset)
        for i in range(len(data[0])):
            id, train, test = read_user_data(i, data, dataset)
            # 使用 UserMESA !!!
            user = UserMESA(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        # 2. 初始化 Scoreboard (历史难度分数)
        self.V = np.ones(len(self.users)) 
        self.epsilon = 0.1 # 探索率

    def train(self, save_model=False, current_time=0, total_times=1):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (MESA)-------------")
            self.send_parameters()
            self.evaluate() # 记录全局精度

            # === MESA 核心逻辑 1: 基于 Scoreboard 计算概率 ===
            score_root = np.sqrt(self.V)
            total_score = np.sum(score_root)
            if total_score == 0:
                probs = np.ones(len(self.users)) / len(self.users)
            else:
                probs = (1 - self.epsilon) * (score_root / total_score) + (self.epsilon / len(self.users))
            probs = probs / np.sum(probs) # 归一化

            # === MESA 核心逻辑 2: 概率采样 ===
            all_indices = np.arange(len(self.users))
            # 这里的 num_users 对应 main.py 里的 args.numusers (每轮选多少人)
            selected_indices = np.random.choice(all_indices, self.num_users, p=probs, replace=False)
            self.selected_users = [self.users[i] for i in selected_indices]
            print(f"Selected Clients: {selected_indices}")

            # === MESA 核心逻辑 3: 只训练被选中的人 & 更新 Scoreboard ===
            for user_idx in selected_indices:
                user = self.users[user_idx]
                # UserMESA.train 返回 (loss, metric)
                _, metric_val = user.train(self.local_epochs)
                # 更新历史分数
                self.V[user_idx] = metric_val

            # === MESA 核心逻辑 4: 聚合 ===
            self.evaluate_personalized_model()
            self.persionalized_aggregate_parameters()

        self.save_results()
        if save_model:
            self.save_model()