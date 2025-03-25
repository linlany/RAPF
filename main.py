
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pdb
import random
import hydra
import logging
from omegaconf import DictConfig

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import load_model, sample
from continual_clip.datasets import build_cl_scenarios
import numpy as np

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )

    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names
    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    for task_id, _ in enumerate(eval_dataset):
        logging.info(f"Train for task {task_id} has started.")
        model.adaptation(task_id, threshold=cfg.threshold)
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        # epoch
        model.train()
        optimizer = torch.optim.Adam(model.adapter.parameters(), lr=cfg.lr, weight_decay=0.0000)

        milestones = cfg.milestones
        epochs = cfg.epochs

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        for i_epoch in range(epochs):
            loss = torch.tensor(0.0).to(device)
            loss_c = torch.tensor(0.0).to(device)
            loss_hinge = torch.tensor(0.0).to(device)
            tqdm_loader = tqdm(train_loader)
            if task_id >0:
                random_class_order_list = list(range(cfg.initial_increment+(task_id-1)*cfg.increment))
                random.shuffle(random_class_order_list)
            batch_id = -1
            for inputs, targets, task_ids in tqdm_loader:
                batch_id += 1
                inputs, targets = inputs.to(device), targets.to(device)
                sg_inputs = None
                edge_sample = None
                ori_targets = targets.clone()
                if task_id > 0:
                    sg_inputs = []
                    sg_targets = []
                    # num of classes per batch. Ensure an epoch traverses all classes at least once. 
                    # For exemple, if there are 100 classes and 50 batches per epoch , there will be 2 classes per batch.
                    if cfg.dataset == "cifar100" and cfg.increment == 5:
                        list_for_one_batch = [random_class_order_list[batch_id*4%len(random_class_order_list)], random_class_order_list[(batch_id*4+1)%len(random_class_order_list)], random_class_order_list[(batch_id*4+2)%len(random_class_order_list)], random_class_order_list[(batch_id*4+3)%len(random_class_order_list)]]
                    elif cfg.dataset == "imagenet_R":
                        list_for_one_batch = [random_class_order_list[batch_id*5%len(random_class_order_list)], random_class_order_list[(batch_id*5+1)%len(random_class_order_list)], random_class_order_list[(batch_id*5+2)%len(random_class_order_list)], random_class_order_list[(batch_id*5+3)%len(random_class_order_list)], random_class_order_list[(batch_id*5+4)%len(random_class_order_list)]]
                    elif cfg.dataset == "cub200":
                        list_for_one_batch = [random_class_order_list[batch_id*10%len(random_class_order_list)], random_class_order_list[(batch_id*10+1)%len(random_class_order_list)], random_class_order_list[(batch_id*10+2)%len(random_class_order_list)], random_class_order_list[(batch_id*10+3)%len(random_class_order_list)], random_class_order_list[(batch_id*10+4)%len(random_class_order_list)], random_class_order_list[(batch_id*10+5)%len(random_class_order_list)], random_class_order_list[(batch_id*10+6)%len(random_class_order_list)], random_class_order_list[(batch_id*10+7)%len(random_class_order_list)], random_class_order_list[(batch_id*10+8)%len(random_class_order_list)], random_class_order_list[(batch_id*10+9)%len(random_class_order_list)]]
                    else:
                        list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
                    for i in list_for_one_batch:
                        sg_inputs.append(sample(model.class_mean_list[i], model.class_cov_list[i],int(10*cfg.beta), shrink=cfg.shrinkage))
                        sg_targets.append(torch.ones(int(10*cfg.beta), dtype=torch.long, device=device)*i)
                    sg_inputs = torch.cat(sg_inputs, dim=0)
                    sg_targets = torch.cat(sg_targets, dim=0)
                    targets = torch.cat([targets, sg_targets], dim=0)
                if model.hard_pairs is not None and model.hard_pairs.shape[0] > 0:
                    edge_sample = []
                    edge_p_target = []
                    edge_n_target = []
                    for hard_pair in model.hard_pairs:
                        edge_sample.append(sample(model.class_mean_list[hard_pair[0]], model.class_cov_list[hard_pair[0]],int(20*cfg.beta), shrink=cfg.shrinkage))
                        edge_p_target.append(torch.ones(int(20*cfg.beta), dtype=torch.long, device=device)*hard_pair[0])
                        edge_n_target.append(torch.ones(int(20*cfg.beta), dtype=torch.long, device=device)*hard_pair[1])
                    edge_sample = torch.cat(edge_sample, dim=0)
                    edge_p_target = torch.cat(edge_p_target, dim=0)
                    edge_n_target = torch.cat(edge_n_target, dim=0)
                if task_id > 0:
                    not_ini = True
                else:
                    not_ini = False
                outputs, _, __, edge_sample_features = model(inputs, memory_data=sg_inputs, not_ini=not_ini, edge_sample=edge_sample, prompt=False)



                if task_id>0:
                    if edge_sample is not None:
                        edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                        edge_target_features = model.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                        edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                        edge_nearest_class_features = model.class_name_features[edge_n_target].type(edge_sample_features.dtype)
                        edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(dim=-1, keepdim=True)
                        loss_hinge = torch.relu(- (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (edge_sample_features * edge_nearest_class_features.clone().detach()).sum(-1) + 0.1).mean()
                loss_c = torch.nn.functional.cross_entropy(outputs, targets.detach())
                if edge_sample is not None:
                    loss = loss_c + loss_hinge
                else:
                    loss = loss_c 

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                tqdm_loader.set_description(f"Epoch {i_epoch + 1}/{cfg.epochs} | Loss: {loss.item():.4f} | Loss_c: {loss_c.item():.4f}| loss_hinge: {loss_hinge.item():.4f} | lr: {scheduler.get_last_lr()[0]:.4f}")
            
            scheduler.step()
        sample_loader = DataLoader(train_dataset[task_id], batch_size=128, shuffle=False, num_workers=cfg.num_workers)
        sample_data = []
        sample_target = []
        sample_after_adapt_feature = []
        print('analyze')
        for input, target, task_ids in tqdm(sample_loader):
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature = model(input, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
            sample_after_adapt_feature.append(after_adapt_feature)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        sample_after_adapt_feature = torch.cat(sample_after_adapt_feature, dim=0)
        model.analyze_mean_cov(sample_data, sample_target)
        model.mix_matrix()
        model.eval()
        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs, _, __, ___ = model(inputs)
                torch.nn.functional.softmax(outputs, dim=-1)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')





@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    with open(cfg.log_path, 'w+') as f: 
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.scenario == "class":
        run_class_incremental(cfg, device)




    
        

















if __name__ == "__main__":
    continual_clip()