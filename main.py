#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from loguru import logger
from FLAlgorithms.servers.serverMESA import MESA
from FLAlgorithms.servers.serverOort import Oort
from FLAlgorithms.servers.serverPoC import PoC
from FLAlgorithms.servers.serverHICS import HiCS
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(
    dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
    local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, save_model
):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    
    # 保存原始 model 名称，用于文件命名
    model_name = model

    for i in range(times):
        logger.info(f"---------------Running time:------------ {i}")
        # Generate model (返回元组: (model_object, model_name))
        if(model_name == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model_name
            else:
                model = Mclr_Logistic(60, 10).to(device), model_name
                
        if(model_name == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model_name
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model_name
            
        if(model_name == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model_name
            else: 
                model = DNN(60,20,10).to(device), model_name

        # select algorithm (current_time=i, total_times=times)
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, times)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, times)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, times)

        if(algorithm == "Oort"):
            server = Oort(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, times)
        
        if(algorithm == "PoC"):
            server = PoC(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, times)
        
        if(algorithm == "MESA"):
            server = MESA(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, times)
        
        if(algorithm == "HiCS"):
            server = HiCS(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, times)
        
        server.train(save_model=save_model, current_time=i, total_times=times)
        server.test()

    # Average data 
    if(algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if(algorithm == "pFedMe"):
        average_data(
            num_users=numusers,
            loc_ep1=local_epochs,
            Numb_Glob_Iters=num_glob_iters,
            lamb=lamda,
            learning_rate=learning_rate,
            beta=beta,
            algorithms="pFedMe_p",
            batch_size=batch_size,
            dataset=dataset,
            k=K,
            personal_learning_rate=personal_learning_rate,
            times=times,
            model_name=model_name
        )
    average_data(
        num_users=numusers,
        loc_ep1=local_epochs,
        Numb_Glob_Iters=num_glob_iters,
        lamb=lamda,
        learning_rate=learning_rate,
        beta=beta,
        algorithms=algorithm,
        batch_size=batch_size,
        dataset=dataset,
        k=K,
        personal_learning_rate=personal_learning_rate,
        times=times,
        model_name=model_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg", "Oort", "PoC", "MESA", "HiCS"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--save_model", action="store_true", help="Save model after training (default: False)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Summary of training process:")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learing rate       : {args.learning_rate}")
    logger.info(f"Average Moving       : {args.beta}")
    logger.info(f"Subset of users      : {args.numusers}")
    logger.info(f"Number of global rounds       : {args.num_global_iters}")
    logger.info(f"Number of local rounds       : {args.local_epochs}")
    logger.info(f"Dataset       : {args.dataset}")
    logger.info(f"Local Model       : {args.model}")
    logger.info("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta, 
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        gpu=args.gpu,
        save_model=args.save_model
    )
