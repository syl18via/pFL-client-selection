import torch
from FLAlgorithms.users.userpFedMe import UserpFedMe

class UserMESA(UserpFedMe):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer, K, personal_learning_rate):
        # 直接调用父类初始化，完全复用 pFedMe 的属性
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                         local_epochs, optimizer, K, personal_learning_rate)

    # 重写 train 方法，加入 Piggyback 逻辑
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        
        # 复用 pFedMe 的训练逻辑
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()

            # K steps 个性化更新
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)

            # 更新本地 Anchor w
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

        self.update_parameters(self.local_model)

        # === MESA 核心修改: 计算并返回 Proximal Metric ===
        proximal_gap = 0.0
        # self.local_model 是 w_{i,K} (Anchor)
        # self.model.parameters() 是 \hat{\theta}_i (Personalized Model)
        for w_param, theta_param in zip(self.local_model, self.model.parameters()):
            proximal_gap += torch.sum((w_param.data - theta_param.data)**2).item()
            
        return LOSS, proximal_gap