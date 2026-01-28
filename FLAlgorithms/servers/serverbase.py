import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
from utils.path_utils import build_result_dir, build_result_filename
import copy
import time
from loguru import logger

class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, current_time, total_times=1):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        # model 是元组 (model_object, model_name)，如 (DNN(), "dnn")
        if isinstance(model, tuple):
            self.model = copy.deepcopy(model[0])
            self.model_name = model[1]
        else:
            self.model = copy.deepcopy(model)
            self.model_name = "unknown"
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.current_time = current_time  # 当前是第几次实验 (用于文件名)
        self.total_times = total_times    # 实验总次数 (用于目录名)
        
        # Time tracking for Accuracy vs. Wall-clock Time plots
        self.wall_clock_times = []  # Cumulative wall-clock time at each round
        self.round_times = []  # Time taken for each round
        self.total_elapsed_time = 0.0  # Total elapsed time
        
        # Selected client loss tracking for Effectiveness of Selection analysis
        self.selected_client_losses = []  # List of lists: each round contains losses of selected clients
        self.selected_client_indices = []  # List of lists: each round contains indices of selected clients
        
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            logger.info("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        # 使用统一的路径构建工具
        K = getattr(self, 'K', None)
        personal_lr = getattr(self, 'personal_learning_rate', None)
        
        # 目录名使用 total_times（实验总次数）
        result_dir = build_result_dir(
            self.model_name, self.dataset, self.num_users, self.num_glob_iters, self.total_times,
            self.learning_rate, self.beta, self.lamda, self.batch_size
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # 文件名使用 current_time（当前实验次数）
        alg = build_result_filename(self.algorithm, self.local_epochs, self.current_time, K, personal_lr, personalized=False)[:-3]  # 去掉.h5后缀
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File(f"{result_dir}/{alg}.h5", 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                # Save time data for Accuracy vs. Wall-clock Time plots
                if len(self.wall_clock_times) > 0:
                    hf.create_dataset('wall_clock_times', data=self.wall_clock_times)
                    hf.create_dataset('round_times', data=self.round_times)
                # Save selected client losses for Effectiveness of Selection analysis
                if len(self.selected_client_losses) > 0:
                    # Convert list of lists to numpy array (padding with NaN for variable lengths)
                    max_len = max(len(losses) for losses in self.selected_client_losses) if self.selected_client_losses else 0
                    if max_len > 0:
                        padded_losses = []
                        for losses in self.selected_client_losses:
                            padded = losses + [np.nan] * (max_len - len(losses))
                            padded_losses.append(padded)
                        hf.create_dataset('selected_client_losses', data=np.array(padded_losses))
                        # Also save indices (as strings, padded with empty strings)
                        padded_indices = []
                        for indices in self.selected_client_indices:
                            # Convert to strings and pad
                            str_indices = [str(idx) for idx in indices]
                            padded = str_indices + [''] * (max_len - len(str_indices))
                            padded_indices.append(padded)
                        # Save as variable-length string array
                        hf.create_dataset('selected_client_indices', data=np.array(padded_indices, dtype='S'))
                hf.close()
        
        # store persionalized value
        alg_p = build_result_filename(self.algorithm, self.local_epochs, self.current_time, K, personal_lr, personalized=True)[:-3]  # 去掉.h5后缀
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File(f"{result_dir}/{alg_p}.h5", 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                # Save time data for personalized model (same as global model)
                if len(self.wall_clock_times) > 0:
                    hf.create_dataset('wall_clock_times', data=self.wall_clock_times)
                    hf.create_dataset('round_times', data=self.round_times)
                # Save selected client losses (same as global model)
                if len(self.selected_client_losses) > 0:
                    max_len = max(len(losses) for losses in self.selected_client_losses) if self.selected_client_losses else 0
                    if max_len > 0:
                        padded_losses = []
                        for losses in self.selected_client_losses:
                            padded = losses + [np.nan] * (max_len - len(losses))
                            padded_losses.append(padded)
                        hf.create_dataset('selected_client_losses', data=np.array(padded_losses))
                        # Save indices as strings
                        padded_indices = []
                        for indices in self.selected_client_indices:
                            str_indices = [str(idx) for idx in indices]
                            padded = str_indices + [''] * (max_len - len(str_indices))
                            padded_indices.append(padded)
                        hf.create_dataset('selected_client_indices', data=np.array(padded_indices, dtype='S'))
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def record_time(self, round_time):
        """Record time for current round and update cumulative time"""
        self.round_times.append(round_time)
        self.total_elapsed_time += round_time
        self.wall_clock_times.append(self.total_elapsed_time)
    
    def record_selected_client_losses(self, selected_indices=None):
        """Record losses of selected clients for Effectiveness of Selection analysis
        
        Args:
            selected_indices: List of selected client indices (can be integer indices or user objects).
                             If None, uses self.selected_users
        """
        if selected_indices is None:
            # Get indices from selected_users
            selected_indices = [user.id for user in self.selected_users]
        
        round_losses = []
        round_indices = []
        
        for idx in selected_indices:
            # Find the user
            user = None
            
            # Handle different input types
            if hasattr(idx, 'id'):
                # idx is a user object
                user = idx
                idx = user.id
            elif isinstance(idx, (int, np.integer)):
                # idx is an integer index (position in self.users)
                if 0 <= idx < len(self.users):
                    user = self.users[idx]
            else:
                # idx might be user id (string or other type), find by id
                for u in self.users:
                    if str(u.id) == str(idx):
                        user = u
                        idx = u.id
                        break
            
            if user is not None:
                # Get loss for this user (before training, to show selection criteria)
                try:
                    _, loss, _ = user.train_error_and_loss()
                    if hasattr(loss, 'item'):
                        loss = loss.item()
                    elif hasattr(loss, 'detach'):
                        loss = loss.detach().item()
                    elif isinstance(loss, (int, float)):
                        loss = float(loss)
                    else:
                        loss = float(loss)
                    
                    round_losses.append(loss)
                    # Store index as string for consistency
                    round_indices.append(str(idx))
                except Exception as e:
                    print(f"Warning: Failed to get loss for user {idx}: {e}")
                    continue
        
        if len(round_losses) > 0:
            self.selected_client_losses.append(round_losses)
            self.selected_client_indices.append(round_indices)
    
    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).detach().item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #logger.info("stats_train[1]",stats_train[3][0])
        logger.info(f"Average Global Accurancy: {glob_acc}")
        logger.info(f"Average Global Trainning Accurancy: {train_acc}")
        logger.info(f"Average Global Trainning Loss: {train_loss}")

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).detach().item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #logger.info("stats_train[1]",stats_train[3][0])
        logger.info(f"Average Personal Accurancy: {glob_acc}")
        logger.info(f"Average Personal Trainning Accurancy: {train_acc}")
        logger.info(f"Average Personal Trainning Loss: {train_loss}")

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).detach().item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #logger.info("stats_train[1]",stats_train[3][0])
        logger.info(f"Average Personal Accurancy: {glob_acc}")
        logger.info(f"Average Personal Trainning Accurancy: {train_acc}")
        logger.info(f"Average Personal Trainning Loss: {train_loss}")
