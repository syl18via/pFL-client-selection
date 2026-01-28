import numpy as np
import time
from loguru import logger
from FLAlgorithms.servers.serverpFedMe import pFedMe

class PoC(pFedMe):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, current_time, total_times=1):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, K, personal_learning_rate, current_time, total_times)
        
        # d: 候选集大小 
        self.d = max(int(self.num_users * 1.5), len(self.users)) 
        # 防止 d 超过总用户数
        self.d = min(self.d, len(self.users))

    def train(self, save_model=False, current_time=0, total_times=1):
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()  # Start timing this round
            
            logger.info(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (PoC)-------------")
            self.send_parameters()
            self.evaluate()

            # === Power-of-Choice Selection Step ===
            
            # 1. 随机采样候选集 (Candidate Set)
            candidate_indices = np.random.choice(len(self.users), self.d, replace=False)
            
            # 2. 探测 Loss (Probe Step)
            # 在实际系统中这需要通信，但在仿真里我们可以直接偷看
            # 注意：这里会产生额外的计算开销，你在论文 Overhead Analysis 里要提到这点！
            probe_start_time = time.time()
            candidate_losses = []
            for idx in candidate_indices:
                # 这是一个轻量级估算，不进行完整训练，只算当前 Loss
                # 我们利用 User 类现有的 train_error_and_loss 方法
                _, loss, _ = self.users[idx].train_error_and_loss()
                # 将 tensor 转换为 Python float
                if hasattr(loss, 'item'):
                    loss = loss.item()
                candidate_losses.append(loss)
            probe_time = time.time() - probe_start_time
            
            # 3. 选 Top-m (Loss 最大的 m 个)
            # argsort 是从小到大，所以取最后 num_users 个
            top_m_indices_local = np.argsort(candidate_losses)[-self.num_users:]
            selected_indices = candidate_indices[top_m_indices_local]
            
            self.selected_users = [self.users[i] for i in selected_indices]

            # Record selected client losses (for Effectiveness of Selection analysis)
            # Note: PoC selects based on probe losses, but we record current loss for consistency with other algorithms
            self.record_selected_client_losses(selected_indices.tolist())

            # 4. 训练选中的用户
            for user in self.selected_users:
                user.train(self.local_epochs)

            # Evaluate personalized model
            self.evaluate_personalized_model()
            self.persionalized_aggregate_parameters()

            # Record time for this round (includes probe time)
            round_time = time.time() - round_start_time
            self.record_time(round_time)
            # Note: probe_time is included in round_time, but can be tracked separately if needed

        self.save_results()
        if save_model:
            self.save_model()