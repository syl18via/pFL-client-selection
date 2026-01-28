import torch
import os
import time

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times):
        super().__init__(device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

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
            
            print(f"-------------[{current_time+1}/{total_times}] Round: {glob_iter+1}/{self.num_glob_iters} (pFedMe)-------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # choose several users to send back upated model to server (random selection)
            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # do update for selected users only (for fair comparison with client selection algorithms)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()

            # Record time for this round
            round_time = time.time() - round_start_time
            self.record_time(round_time)

        #print(loss)
        self.save_results()
        if save_model:
            self.save_model()
    
  
