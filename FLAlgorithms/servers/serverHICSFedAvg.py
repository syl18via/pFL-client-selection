import numpy as np
import time
import copy
from loguru import logger
from FLAlgorithms.servers.serveravg import FedAvg
# Import HiCS helper functions
from FLAlgorithms.servers.serverHICSpFedMe import get_gradients_fc, get_clusters_with_alg2, sample_clients
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from numpy.random import choice

class HiCSFedAvg(FedAvg):
    """HiCS: Hierarchical Clustering-based Client Selection based on FedAvg framework (not pFedMe)"""
    
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, current_time, total_times=1):
        # Note: No K and personal_learning_rate needed for FedAvg framework
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, current_time, total_times)
        # Update algorithm name to distinguish from pFedMe version
        self.algorithm = algorithm + "_FedAvg"
        
        # HiCS parameters (same as HiCS)
        if dataset in ["Cifar10", "cifar", "FedCIFAR100"]:
            self._temp = 0.015
        elif dataset == "fmnist":
            self._temp = 0.0025
        else:
            # Default temperature for other datasets
            self._temp = 0.01
        
        self.hics_alphas = [0.001, 0.002, 0.005, 0.01, 0.5]
        self._lambda = 10
        self._gamma = 4
        self.M = 5
        
        alphas = set(self.hics_alphas)
        self.multialpha = len(alphas) > 1
        
        self.num_round = num_glob_iters
        self.round = None
        self.warmup = None
        
        self.global_accu = 0
        self.global_loss = 1e6
        
        # Initialize gradients and magnitudes
        self.gradients = None
        self.magnitudes = None
        self.n_samples = None
        self.weights = None
        self.previous_global_model = None

    def _magnitude_gradient(self, gradients):
        """Calculate magnitude of gradients (same as HiCS)"""
        magnitudes = []
        for idx in range(len(gradients)):
            gradient = gradients[idx][0]
            m, n = gradient.shape
            magnitude = np.zeros(m)
            for c in range(m):
                magnitude[c] = np.sum(gradient[c])/n
            magnitudes.append(magnitude)
        return magnitudes

    def _estimated_entropy(self, estimated_H, Clusters):
        """Estimate entropy (same as HiCS)"""
        entropy = []
        for i in range(len(Clusters)):
            cluster_i = Clusters[i]
            entropy_i = 0
            for j in cluster_i:
                entropy_i += estimated_H[j]
            entropy.append(entropy_i / len(cluster_i))
        return entropy

    def _cluster_sampling(self, pool_gradients, pool_magnitudes, metric, round, pool_weights, pool_n_samples):
        """Cluster-based sampling (same as HiCS)"""
        n_sampled = self.num_users
        estimated_H = []
        
        for i in range(len(pool_gradients)):
            gradient = pool_gradients[i][0]
            magnitude = pool_magnitudes[i] if pool_magnitudes is not None else None
            estimated_H.append(np.sum(gradient) / (self._temp + 1e-8))
        
        if metric == "cosine":
            distances = pdist(pool_gradients, metric='cosine')
        else:
            distances = pdist(pool_gradients, metric=metric)
        
        linkage_matrix = linkage(distances, method='ward')
        labels = fcluster(linkage_matrix, n_sampled, criterion='maxclust')
        
        Clusters = []
        for i in range(self.M):
            cluster_i = np.where(labels == i)[0]
            Clusters.append(cluster_i)
        
        avg_entropy = self._estimated_entropy(estimated_H, Clusters)
        return self._sample_clients_entropy(avg_entropy, Clusters, pool_n_samples, round)

    def _sample_clients_entropy(self, entropy, Clusters, n_samples, round):
        """Sample clients based on entropy (same as HiCS)"""
        from numpy.random import choice
        
        n_sampled = max(self.num_users, 1)
        n_clustered = len(Clusters)
        entropy = np.exp(self._gamma * (self.num_round - round) * entropy / self.num_round)
    
        p_cluster = entropy/np.sum(entropy)
        sampled_clients = []
        clusters_selected = [0]*n_sampled
        
        for k in range(n_sampled):
            select_group = int(choice(n_clustered, 1, p=p_cluster))
            while clusters_selected[select_group] >= len(Clusters[select_group]):
                select_group = int(choice(n_clustered, 1, p=p_cluster)) 
            clusters_selected[select_group] += 1
        
        for k in range(len(clusters_selected)):
            if clusters_selected[k] == 0:
                continue
            select_clients = choice(
                Clusters[k], clusters_selected[k], replace=False
            )
            for i in range(clusters_selected[k]):
                sampled_clients.append(select_clients[i])
        return sampled_clients

    def train(self, save_model=False, current_time=0, total_times=1):
        """Train with HiCS selection using FedAvg framework"""
        # Initialize n_samples and weights
        n_samples_dict = {i: user.train_samples for i, user in enumerate(self.users)}
        client_ids = sorted(n_samples_dict.keys())
        self.n_samples = np.array([n_samples_dict[i] for i in client_ids])
        self.weights = self.n_samples / np.sum(self.n_samples)
        
        # Initialize gradients with dummy values
        self.gradients = get_gradients_fc(
            "clustered_2", 
            self.model, 
            [self.model] * len(self.users)
        )
        self.magnitudes = self._magnitude_gradient(self.gradients)
        
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()  # Start timing this round
            
            logger.info(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (HiCSFedAvg)-------------")
            self.send_parameters()
            self.evaluate()

            # === HiCS Selection Step ===
            self.round = glob_iter
            self.warmup = int(len(self.users) / self.num_users) if len(self.users) >= self.num_users else 1
            
            if self.round < self.warmup:
                # Warmup phase: round-robin selection
                pool = list(range(len(self.users)))
                sampled = pool[glob_iter*self.num_users:(glob_iter + 1)*self.num_users]
                if len(sampled) < self.num_users:
                    remain = self.num_users - len(sampled)
                    remaining_pool = [i for i in pool if i not in sampled]
                    if len(remaining_pool) > 0:
                        sampled += list(np.random.choice(remaining_pool, min(remain, len(remaining_pool)), replace=False))
                selected_indices = sampled
            else:
                # HiCS cluster-based selection
                pool = list(range(len(self.users)))
                pool_gradients = [self.gradients[i] for i in pool]
                pool_magnitudes = [self.magnitudes[i] for i in pool] if self.magnitudes is not None else None
                pool_n_samples = self.n_samples[pool] if hasattr(self, 'n_samples') and self.n_samples is not None else np.ones(len(pool))
                pool_weights = pool_n_samples / np.sum(pool_n_samples)
                
                sampled_idx = self._cluster_sampling(
                    pool_gradients,
                    pool_magnitudes,
                    "cosine",
                    glob_iter,
                    pool_weights,
                    pool_n_samples
                )
                selected_indices = [pool[i] for i in sampled_idx]
            
            self.selected_users = [self.users[i] for i in selected_indices]

            # Record selected client losses
            self.record_selected_client_losses(selected_indices)

            # Store local models before training for gradient calculation
            local_models = []
            for user_idx in selected_indices:
                user = self.users[user_idx]
                # Store model before training
                local_models.append(copy.deepcopy(user.model))
            
            # Train selected users (FedAvg standard training, not personalized)
            for user in self.selected_users:
                user.train(self.local_epochs)

            # Update gradients after training
            # Calculate gradients as difference between local and global models
            self.previous_global_model = copy.deepcopy(self.model)
            trained_local_models = [self.users[idx].model for idx in selected_indices]
            
            # Update gradients for selected clients
            gradients_i = get_gradients_fc(
                "clustered_2",
                self.previous_global_model,
                trained_local_models
            )
            for idx, gradient in zip(selected_indices, gradients_i):
                self.gradients[idx] = gradient
            
            self.magnitudes = self._magnitude_gradient(self.gradients)

            # Use FedAvg aggregation (not personalized aggregation)
            self.aggregate_parameters()

            # Record time for this round
            round_time = time.time() - round_start_time
            self.record_time(round_time)

        self.save_results()
        if save_model:
            self.save_model()
