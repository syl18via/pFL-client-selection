import torch
import os
import time

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users,times):
        super().__init__(device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer ,total_users , num_users)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self, save_model=False, current_time=0, total_times=1):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            round_start_time = time.time()  # Start timing this round
            
            print(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (PerAvg)-------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
                
            self.aggregate_parameters()

            # Record time for this round
            round_time = time.time() - round_start_time
            self.record_time(round_time)

        self.save_results()
        if save_model:
            self.save_model()
