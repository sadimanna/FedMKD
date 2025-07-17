import sys
import os
import time
import numpy as np
import random
import json
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.multiprocessing import Pool
import copy
import time
import scipy.sparse as sp
from torchvision import models
import copy
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
from torch.backends import cudnn
import parse
args = parse.args
# os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.gpu)

# 自动混合精度
from torch.cuda import amp

torch.backends.cudnn.enabled = False
torch.multiprocessing.set_sharing_strategy('file_system')

import easyfl
import utils

from model import get_model, BYOL
from dataset import get_semi_supervised_dataset
from easyfl.datasets.data import CIFAR100
from easyfl.coordinator import Coordinator
from client import FedSSLClient
from server import MyDistillServer
# from Fedmd_server import FedmdServer
from fedema_server import FedSSLServer
# from distillServer import distillServer
import logging
import torch.distributed as dist

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

model_dir = args.save_model_path
CIFAR100 = "cifar100"
logger = logging.getLogger(__name__)


# def setup(rank, world_size): 
#     """ 
#     Initialize the process group for distributed training. 
#     """ 
#     # Initialize the process group using NCCL backend 
#     dist.init_process_group("nccl")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     global_rank = int(os.environ["RANK"])
#     torch.cuda.set_device(local_rank)
#     return local_rank, global_rank

# def cleanup(): 
#     """ 
#     Destroy the process group after training is complete. 
#     """ 
#     dist.destroy_process_group() 

def ignore_resize_warning(message, category, filename, lineno, file=None, line=None):
    if "An output with one or more elements was resized" in str(message):
        return True
    return False


# 将警告过滤器应用到特定的警告消息
warnings.showwarning = ignore_resize_warning

def client_main(client_list, data_list, device, epoch, clients, train_data, test_data, config):
    # return client model parameters
    new_model_weights = []
    for idx in tqdm(range(len(client_list))):
        client = client_list[idx]
        client_name = data_list[idx]
        client_i = FedSSLClient(client_name, config, train_data, test_data, device, epoch)
        if config.framework == 'fedema':
            client_i.compressed_model = clients[idx]
            tmp_model_path = os.path.join(model_dir, 'saved_model/', task_id, '/local_model', client_name, '.pth')
            if os.path.exists(tmp_model_path):
                client_i.local_model = clients[idx]
                client_i.local_model.load_state_dict(torch.load(tmp_model_path))
            client_i.update_model()
        else:
            client_i.model = clients[idx]  # 直接重置了模型参数

        # Train
        client_i.train(config.client, device=device)
        # save model
        # if (epoch + 1) % args.test_every == 0:
        #     client_i.save_model()
        tmp_model = client_i.model.cpu()
        new_model_weights.append(tmp_model)
    return new_model_weights


def server_train(client_models, server_model, public_dataset, device, epoch):
    # multi-teacher distillation
    # input: clients model, server model, device, public dataset
    # output: server model
    if args.framework in ['ours','oursnoalign']:
        server = MyDistillServer(None, config, public_dataset, test_data, device, epoch)
    else:
        raise ValueError(f"framework type is wrong")
    server.model = server_model
    server.client_models = client_models
    start_time = time.time()
    # 调用训练过程
    # Train
    server.train(config.server, device=device)
    server.save_model()
    # Test
    if (epoch + 1) % args.test_every == 0:
        server.test()
    return server.model.cpu()


def fedAvg_agg(client_models, avg_weights, public_dataset, device, epoch):
    # server = MyDistillServer(None, config, public_dataset, test_data, device, epoch)
    server = FedSSLServer(config, public_dataset, test_data)
    start_time = time.time()
    # fedAvg_weight = FedAvg(client_models, avg_weights)
    # server.model = fedAvg_weight
    server.client_model = client_models
    server.weight = avg_weights
    server.model = client_models[0]
    server.aggregation()
    server.save_model()
    # Test
    if (epoch + 1) % args.test_every == 0:
        server.test(device)
    logger.info(f"---------Time cost of fedavg is {time.time() - start_time}s---------")
    return server.model


def client_align(ori_client, server_model, public_dataset, device, config):
    # update clients model
    # input: clients model, server model, device, public dataset
    # output: clients model
    new_model_weights = []
    start_time = time.time()
    for idx in tqdm(range(len(ori_client))):
        client_config = config
        client_config.client.momentum_update = False
        client_config.client.local_epoch = 1
        s_client = FedSSLClient(None, client_config, public_dataset, None, device, None)
        s_client.model = ori_client[idx]
        s_client.model.target_encoder = server_model.online_encoder
        # Train
        s_client.align_train(config.client, device=device)
        tmp_model = s_client.model.online_encoder.cpu()
        new_model_weights.append(tmp_model)
    logger.info(f"--------Time of align is {time.time() - start_time}s--------")
    return new_model_weights


if __name__ == '__main__':
    gpu_index = args.gpu
    num_processes = len(gpu_index)
    devices = [torch.device('cuda', int(index)) for index in gpu_index]

    client_list = range(args.num_of_clients)
    step = int(np.ceil(len(client_list) / num_processes))

    class_per_client = args.class_per_client

    if args.dataset == CIFAR100:
        class_per_client *= 10

    task_id = args.task_id
    if task_id == "":
        task_id = f"{args.dataset}_{args.framework}_{args.client_type}_{args.server_model}_{args.num_of_clients}_" \
                  f"{args.encoder_network}_{args.data_partition}_{args.dir_alpha}_" \
                  f"{args.local_epoch}_{args.server_epoch}_{args.rounds}_{args.batch_size}_{args.public}_{args.data_size}_{args.public_size}"

    # print(task_id)
    momentum_update = True
    image_size = 32
    config = {
        "task_id": task_id,
        "seed": args.seed,
        "framework": args.framework,
        "client_type": args.client_type,
        "data_number": args.data_size,
        "data": {
            "dataset": args.dataset,
            "num_of_clients": args.num_of_clients,
            "split_type": args.data_partition,
            "class_per_client": class_per_client,
            "data_amount": 1,
            "iid_fraction": 1,
            "min_size": 10,
            "alpha": args.dir_alpha,
            "public": args.public,
        },
        "client_model": args.client_model,
        "server_model": args.server_model,
        "test_mode": "test_in_server",
        "server": {
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "test_every": args.test_every,
            "save_model_every": args.save_model_every,
            "save_model_path": args.save_model_path,
            "clients_per_round": args.clients_per_round,
            "random_selection": args.random_selection,
            "save_predictor": args.save_predictor,
            "test_all": True,
            "model": args.server_model,
            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "drop_last": True,
            "server_epoch": args.server_epoch,
            "gaussian": False,
            "image_size": image_size,
            "aggregate_encoder": args.aggregate_encoder,
            "update_encoder": args.update_encoder,
            "update_predictor": args.update_predictor,
            "random_selection": args.random_selection,

            "encoder_weight": args.encoder_weight,
            "predictor_weight": args.predictor_weight,

            "momentum_update": momentum_update,
            "data_number": args.data_size,
        },
        "client": {
            "drop_last": True,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "data_number": args.data_size,
            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            # application specific
            "model": args.client_model,
            "rounds": args.rounds,
            "round_id": 0,
            "gaussian": False,
            "image_size": image_size,
            "save_model_path": args.save_model_path,

            "aggregate_encoder": args.aggregate_encoder,
            "update_encoder": args.update_encoder,
            "update_predictor": args.update_predictor,
            "random_selection": args.random_selection,
            "dapu_threshold": args.dapu_threshold,
            "weight_scaler": args.weight_scaler,
            "auto_scaler": args.auto_scaler,
            "auto_scaler_target": args.auto_scaler_target,

            "encoder_weight": args.encoder_weight,
            "predictor_weight": args.predictor_weight,
            "momentum_update": momentum_update,
        }
    }

    # split public data first
    if args.semi_supervised:
        print('true')
        train_data, test_data, _ = get_semi_supervised_dataset(args.dataset,
                                                               args.num_of_clients,
                                                               args.data_partition,
                                                               class_per_client,
                                                               args.label_ratio)
        print(train_data.dtype())

    # print("######################")
    # print("######################")
    small_client_network = get_model(args.client_model, "resnet18", args.predictor_network)
    # print("***********************")
    middel_client_network = get_model(args.client_model, "resnet34", args.predictor_network)
    # print("$$$$$$$$$$$$$$$$$$$$$$")
    vgg_client_network = get_model(args.client_model, "vgg", args.predictor_network)
    # print("@@@@@@@@@@@@@@@@@@@@@@")
    # print(vgg_client_network)

    if args.framework in ['ours', 'single', 'oursnoalign']:
        # define server model
        server_model = get_model(args.server_model, args.encoder_network, args.predictor_network) #BYOLServerModel is instantiated in get_model
        model_path = os.path.join(model_dir, 'saved_models', task_id, 'global_model.pth')
        print(model_path)
        if os.path.exists(model_path) and args.resume:
            load_model = torch.load(model_path)
            new_model = OrderedDict()
            for k, v in load_model.items():
                if k[:15] == 'online_encoder.':
                    name = k
                    new_model[name] = v
                elif k[:15] == 'target_encoder.':
                    pass
                else:
                    name = k
                    new_model[k]=v
            server_model.load_state_dict(new_model)
            print("loaded successfully")

    coord = Coordinator()
    # print(config)
    coord, config = easyfl.init(config, init_all=False)
    train_data = coord.train_data
    test_data = coord.test_data
    public_data = coord.public_data
    data_user = coord.train_data.users
    weight = []
    for user in data_user:
        weight.append(len(train_data.data[user]['y']))

    # define local model list
    if config.client_type == 'resnet18':
        clients = [small_client_network] * args.num_of_clients
    elif config.client_type == 'resnet34':
        clients = [middel_client_network] * args.num_of_clients
    elif config.client_type == 'vgg':
        clients = [vgg_client_network] * args.num_of_clients
    elif config.client_type == 'mix':
        choices = [small_client_network, middel_client_network]
        # clients = np.random.choice(choices, args.num_of_clients, replace=True)
        clients = [copy.deepcopy(small_client_network), 
                   copy.deepcopy(small_client_network), 
                   copy.deepcopy(vgg_client_network), 
                   copy.deepcopy(vgg_client_network),
                   copy.deepcopy(vgg_client_network)]
    idx = 0
    for user in data_user:
        model_path = os.path.join(model_dir, 'saved_models', task_id, 'local_model', user, '.pth')
        if os.path.exists(model_path):
            clients[idx].load_state_dict(torch.load(model_path))
        idx += 1

    # logger.info(f"{clients}")

    avg_weights = weight / np.sum(weight)
    print('---------Start training---------')

    for epoch in tqdm(range(args.rounds)):
        config.client.round_id += 1

        # multi-process
        time_start = time.time()
        ctx = torch.multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=num_processes)
        process_arr = []
        for i in range(num_processes):
            device = devices[i]
            process_arr.append(
                pool.apply_async(client_main, args=(
                    client_list[i * step:(i + 1) * step], data_user[i * step:(i + 1) * step], devices[i], epoch,
                    clients[i * step:(i + 1) * step], train_data, test_data, config)))
        pool.close()
        pool.join()
        print('---------Finishing training clients---------')

        # each process should get model parameters.
        usr_model_weights_t = []
        if num_processes > 1:
            usr_model_weights_t = process_arr[0].get()
            for process in process_arr[1:]:
                tmp_usr_model_weights_t = process.get()
                usr_model_weights_t += tmp_usr_model_weights_t
            usr_model_weights = usr_model_weights_t
        else:
            usr_model_weights = process_arr[0].get()

        # print("---------Time cost of train of epoch {:d} is {:.1f}s---------".format(epoch, time.time() - time_start))
        logger.info(f"---------Time cost of train of epoch {epoch} is {time.time() - time_start}s---------")
        # print('user model weights is :', len(usr_model_weights))

        if config.framework == 'ours':
            # train server model use distill, server only need the client online encoder
            start_time = time.time()
            server_model = server_train(usr_model_weights, server_model, public_data, devices[0], epoch)
            print("---------Finish distill---------")
            logger.info(f"---------Time cost of distill is {time.time() - start_time}s---------")
            # align local model in server, change target network
            # single process version
            # clients model parameter, data_user:cid
            # for i in range(len(client_list)):
            #     client = client_align(clients[i], server_model, public_data, devices[0])
            #     clients[i].target_encoder = client.online_encoder  # can considering EMA update
            #     print("End training of client {:d}".format(i))

            # multi-process version
            align_time = time.time()
            # ctx = torch.multiprocessing.get_context("spawn")
            # torch.multiprocessing.set_start_method(method='forkserver', force=True)
            pool = ctx.Pool(processes=num_processes)
            process_arr = []
            for i in range(num_processes):
                device = devices[i]
                torch.cuda.set_device(device)
                process_arr.append(
                    pool.apply_async(client_align, args=(
                        usr_model_weights[i * step:(i + 1) * step], server_model, public_data, device, config)))
            pool.close()
            pool.join()

            # each process should get model parameters.
            model_weights_t = []
            if num_processes > 1:
                model_weights_t = process_arr[0].get()
                for process in process_arr[1:]:
                    tmp_model_weights_t = process.get()
                    model_weights_t += tmp_model_weights_t
                model_weights = model_weights_t
            else:
                model_weights = process_arr[0].get()

            for i in range(len(clients)):
                clients[i].target_encoder = model_weights[i]
            logger.info(f"---------Time cost of align is {time.time() - align_time}s---------")

        elif config.framework == 'fedavg':
            fedAvg_weight = fedAvg_agg(usr_model_weights, avg_weights, public_data, devices[0], epoch)
            for idx in range(len(clients)):
                clients[idx] = fedAvg_weight
        elif config.framework == 'modfedmkd':
            raise NotImplementedError("modfedmkd is not implemented")
        else:
            raise ValueError(f"framework type is wrong")

        print("---------Time cost of train of epoch {:d} is {:.1f}s---------".format(epoch, time.time() - time_start))
