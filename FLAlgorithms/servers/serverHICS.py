import numpy as np
import copy
import time
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.stats
from itertools import product
from sklearn.cluster import AgglomerativeClustering 
from numpy.random import choice
from copy import deepcopy
from loguru import logger

from FLAlgorithms.servers.serverpFedMe import pFedMe
from utils.model_utils import read_data, read_user_data
import torch

# HiCS helper functions (from hics.py)

def get_similarity(grad_1, grad_2, distance_type="L1"):
    """Calculate similarity between two gradients"""
    if distance_type == "L1":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm
    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm
    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)
        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)
            return np.arccos(norm)

def get_gradients_fc(sampling, global_m, local_models):
    """Return the `representative gradient` formed by the difference between
    the local work and the sent global model"""
    local_model_params = []
    for model in local_models:
        local_model_params += [
           [tens.detach().cpu().numpy() for tens in list(model.parameters())[-2:]]
        ]
            
    global_model_params = [
        tens.detach().cpu().numpy() for tens in list(global_m.parameters())[-2:]
    ]
    
    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                    local_params, global_model_params
                )
            ]
        ]
    return local_model_grads

def get_clusters_with_alg2(linkage_matrix: np.ndarray, n_sampled: int, weights: np.ndarray):
    """Algorithm 2 from HiCS paper"""
    epsilon = int(10 ** 10)

    # associate each client to a cluster
    link_matrix_p = deepcopy(linkage_matrix)
    augmented_weights = deepcopy(weights)

    for i in range(len(link_matrix_p)):
        idx_1, idx_2 = int(link_matrix_p[i, 0]), int(link_matrix_p[i, 1])
        new_weight = np.array(
            [augmented_weights[idx_1] + augmented_weights[idx_2]]
        )
        augmented_weights = np.concatenate((augmented_weights, new_weight))
        link_matrix_p[i, 2] = int(new_weight * epsilon)

    clusters = fcluster(
        link_matrix_p, int(epsilon / n_sampled), criterion="distance"
    )

    n_clients, n_clusters = len(clusters), len(set(clusters))

    # Associate each cluster to its number of clients in the cluster
    pop_clusters = np.zeros((n_clusters, 2)).astype(int)
    for i in range(n_clusters):
        pop_clusters[i, 0] = i + 1
        for client in np.where(clusters == i + 1)[0]:
            pop_clusters[i, 1] += int(weights[client] * epsilon * n_sampled)

    pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

    distri_clusters = np.zeros((n_sampled, n_clients)).astype(int)

    # n_sampled biggest clusters that will remain unchanged
    kept_clusters = pop_clusters[n_clusters - n_sampled :, 0]

    for idx, cluster in enumerate(kept_clusters):
        for client in np.where(clusters == cluster)[0]:
            distri_clusters[idx, client] = int(
                weights[client] * n_sampled * epsilon
            )

    k = 0
    for j in pop_clusters[: n_clusters - n_sampled, 0]:
        clients_in_j = np.where(clusters == j)[0]
        np.random.shuffle(clients_in_j)

        for client in clients_in_j:
            weight_client = int(weights[client] * epsilon * n_sampled)

            while weight_client > 0:
                sum_proba_in_k = np.sum(distri_clusters[k])

                u_i = min(epsilon - sum_proba_in_k, weight_client)

                distri_clusters[k, client] = u_i
                weight_client += -u_i

                sum_proba_in_k = np.sum(distri_clusters[k])
                if sum_proba_in_k == 1 * epsilon:
                    k += 1

    distri_clusters = distri_clusters.astype(float)
    for l in range(n_sampled):
        distri_clusters[l] /= np.sum(distri_clusters[l])

    return distri_clusters

def sample_clients(distri_clusters):
    """Sample clients from distribution clusters"""
    n_clients = len(distri_clusters[0])
    n_sampled = len(distri_clusters)

    sampled_clients = np.zeros(len(distri_clusters), dtype=int)

    for k in range(n_sampled):
        sampled_clients[k] = int(choice(n_clients, 1, p=distri_clusters[k]))

    return sampled_clients


class HiCS(pFedMe):
    """HiCS: Hierarchical Clustering-based Client Selection for Federated Learning"""
    
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, K, personal_learning_rate, times)
        
        # HiCS parameters
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
        """Calculate magnitude of gradients"""
        magnitudes = []
        for idx in range(len(gradients)):
            gradient = gradients[idx][0]
            m, n = gradient.shape
            magnitude = np.zeros(m)
            for c in range(m):
                magnitude[c] = np.sum(gradient[c])/n
            magnitudes.append(magnitude)
        return magnitudes

    def _estimated_entropy_from_grad(self, magnitudes):
        """Estimate entropy from gradient magnitudes"""
        estimated_H = []
        T = self._temp
        for idx in range(len(magnitudes)):
            magnitudes_exp = np.exp(magnitudes[idx]/T) / np.sum(np.exp(magnitudes[idx]/T))
            pk = np.array(magnitudes_exp)
            estimated_h = scipy.stats.entropy(pk)
            estimated_H.append(estimated_h)
        return estimated_H

    def _estimated_entropy(self, estimated_H, Clusters):
        """Estimate entropy for clusters"""
        Entropys = []
        for k in range(len(Clusters)):
            group_entropy = 0
            cluster = Clusters[k]
            for idx in cluster:
                group_entropy += estimated_H[idx]
            if len(cluster) > 0:
                group_entropy = group_entropy/len(cluster)
            Entropys.append(group_entropy)
        Entropys = np.array(Entropys)
        return Entropys

    def _get_matrix_similarity_from_grads_entropy(self, local_model_grads, estimated_H, distance_type):
        """Get similarity matrix from gradients and entropy"""
        n_clients = len(local_model_grads)
        metric_matrix = np.zeros((n_clients, n_clients))
        for i, j in product(range(n_clients), range(n_clients)):
            metric = get_similarity(local_model_grads[i], local_model_grads[j], distance_type) 
            metric_matrix[i, j] = metric + self._lambda * abs(estimated_H[i] - estimated_H[j])
        return metric_matrix

    def _cluster_sampling(self, gradients, magnitudes, sim_type, round, weights, n_samples):
        """Perform cluster-based sampling"""
        Clusters = []
        n_sampled = max(self.num_users, 1)

        magnitudes = self._magnitude_gradient(gradients)
        estimated_H = self._estimated_entropy_from_grad(magnitudes)
        sim_matrix = self._get_matrix_similarity_from_grads_entropy(gradients, estimated_H, distance_type=sim_type)
        linkage_matrix = linkage(sim_matrix, "ward") 

        if np.array(estimated_H).var() < 0.1: 
            hc = AgglomerativeClustering(
                n_clusters=self.M, metric="euclidean", linkage='ward'
            ) 
            
            hc.fit_predict(sim_matrix)
            labels = hc.labels_
            for i in range(self.M):
                cluster_i = np.where(labels == i)[0]
                Clusters.append(cluster_i)    
            avg_entropy = self._estimated_entropy(estimated_H, Clusters)
            return self._sample_clients_entropy(avg_entropy, Clusters, n_samples, round)
        else:
            distri_clusters = get_clusters_with_alg2(linkage_matrix, n_sampled, weights)
            return sample_clients(distri_clusters)

    def _sample_clients_entropy(self, entropy, Clusters, n_samples, round):
        """Sample clients based on entropy"""
        n_sampled = max(self.num_users, 1)
        n_clients = len(n_samples)
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

    def before_train(self, n_samples, global_m):
        """Initialize before training"""
        client_ids = sorted(n_samples.keys())
        self.n_samples = np.array([n_samples[i] for i in client_ids])
        self.weights = self.n_samples / np.sum(self.n_samples)
        
        # Initialize gradients with dummy values (will be updated during training)
        self.gradients = get_gradients_fc(
            "clustered_2", 
            global_m, 
            [global_m] * len(self.users)
        )
        self.magnitudes = self._magnitude_gradient(self.gradients)

    def before_step(self, global_m, local_models=None):
        """Called before each training step"""
        self.previous_global_model = copy.deepcopy(global_m)

    def select_clients_hics(self, round, client_idxs):
        """HiCS client selection method"""
        self.round = round
        self.selected_client_num = self.num_users

        self.warmup = int(len(client_idxs) / self.num_users) if len(client_idxs) >= self.num_users else 1

        pool = list(client_idxs)
        pool_size = len(pool)

        # Get gradients, magnitudes, weights for current pool
        pool_gradients = [self.gradients[i] for i in pool]
        pool_magnitudes = [self.magnitudes[i] for i in pool] if self.magnitudes is not None else None
        pool_n_samples = self.n_samples[pool] if hasattr(self, 'n_samples') and self.n_samples is not None else np.ones(pool_size)
        pool_weights = pool_n_samples / np.sum(pool_n_samples)

        if self.round < self.warmup:
            # Warmup phase: round-robin selection
            sampled = pool[round*self.num_users:(round + 1)*self.num_users]
            if len(sampled) < self.num_users:
                remain = self.num_users - len(sampled)
                remaining_pool = [i for i in pool if i not in sampled]
                if len(remaining_pool) > 0:
                    sampled += list(np.random.choice(remaining_pool, min(remain, len(remaining_pool)), replace=False))
            sampled_clients = sampled
        else:
            # HiCS cluster-based selection
            sampled_idx = self._cluster_sampling(
                pool_gradients,
                pool_magnitudes,
                "cosine",
                round,
                pool_weights,
                pool_n_samples
            )
            # Map back to original client indices
            sampled_clients = [pool[i] for i in sampled_idx]
        
        return sampled_clients

    def after_step(self, client_idxs, local_models, global_m, loss, acc):
        """Update gradients after training step"""
        self.global_loss = loss
        self.global_accu = acc
        
        if self.multialpha:
            if self.round < self.warmup:
                gradients_i = get_gradients_fc(
                    "clustered_2", 
                    self.previous_global_model, local_models
                )
                for idx, gradient in zip(client_idxs, gradients_i):
                    self.gradients[idx] = gradient
        else:
            gradients_i = get_gradients_fc(
                "clustered_2", 
                self.previous_global_model, 
                local_models
            )
            for idx, gradient in zip(client_idxs, gradients_i):
                self.gradients[idx] = gradient
                
        self.magnitudes = self._magnitude_gradient(self.gradients)

    def train(self, save_model=False, current_time=0, total_times=1):
        """HiCS training loop"""
        # Initialize gradients before training starts
        n_samples_dict = {i: user.train_samples for i, user in enumerate(self.users)}
        self.before_train(n_samples_dict, self.model)
        
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()  # Start timing this round
            
            logger.info(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (HiCS)-------------")
            
            # Store previous global model before sending parameters
            self.before_step(self.model)
            
            self.send_parameters()
            self.evaluate()

            # Get all client indices
            all_client_indices = list(range(len(self.users)))
            
            # HiCS client selection
            selected_indices = self.select_clients_hics(glob_iter, all_client_indices)
            self.selected_users = [self.users[i] for i in selected_indices]
            logger.info(f"Selected Clients: {selected_indices}")

            # Record selected client losses before training (for Effectiveness of Selection analysis)
            self.record_selected_client_losses(selected_indices)

            # Train selected users
            local_models = []
            for user_idx in selected_indices:
                user = self.users[user_idx]
                user.train(self.local_epochs)
                # Store local model for gradient calculation
                local_models.append(copy.deepcopy(user.model))

            # Update gradients after training
            current_loss = self.rs_train_loss[-1] if len(self.rs_train_loss) > 0 else 0
            current_acc = self.rs_glob_acc[-1] if len(self.rs_glob_acc) > 0 else 0
            self.after_step(selected_indices, local_models, self.model, current_loss, current_acc)

            # Aggregate parameters
            self.evaluate_personalized_model()
            self.persionalized_aggregate_parameters()

            # Record time for this round
            round_time = time.time() - round_start_time
            self.record_time(round_time)

        self.save_results()
        if save_model:
            self.save_model()
