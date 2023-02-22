import os
import random
import numpy as np
import torch
from easydict import EasyDict as edict
import yaml
import torch.nn as nn
import itertools
from tqdm import tqdm
import time


def pmi(co_occurrence_matrix, positive=True):
    col_totals = co_occurrence_matrix.sum(axis=0)
    total = col_totals.sum()
    row_totals = co_occurrence_matrix.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    arr = co_occurrence_matrix / expected
    arr [np.isnan(arr)] = 0.0
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        pmi_matrix = np.log(arr)
    pmi_matrix[np.isinf(pmi_matrix)] = 0.0  # log(0) = 0
    if positive:
        pmi_matrix[pmi_matrix < 0] = 0.0
    return pmi_matrix.cpu().numpy()

def construct_pmi_matrix(features, dataset):
    dir='../datasets/'+ dataset +'/pmi'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    file_path = os.path.join(dir, "pmi_matrix.npy")

    if os.path.exists(file_path):
        pmi_matrix = np.load(file_path)

    else:
        dim = features.shape[1]
        co_occurrence_matrix = torch.zeros((dim, dim))
        for feat in features:
            idx = [i for i,value in enumerate(feat) if value!=0.]
            # print(idx)
            pairs = list(itertools.product(idx, repeat=2))
            idx1 = [x for x,_ in pairs]
            idx2 = [y for _,y in pairs]
            co_occurrence_matrix[[idx1],[idx2]] += 1
        co_occurrence_matrix = co_occurrence_matrix - torch.diag_embed(torch.diag(co_occurrence_matrix))
        pmi_matrix = pmi(co_occurrence_matrix)
        np.save(file_path, pmi_matrix)

    return pmi_matrix


def regularization(named_parameters, l1, l2, l1_params_list, l2_params_list):
    l1_norm = 0.
    l2_norm = 0.
    for pname, p in named_parameters:
        if any([l1_param in pname for l1_param in l1_params_list]):
            l1_norm += torch.norm(p, 1)
        if any([l2_param in pname for l2_param in l2_params_list]):
            l2_norm += torch.norm(p, 2)
    
    l1_norm = l1 * l1_norm
    l2_norm = l2 * l2_norm

    return l1_norm, l2_norm


def expRun(model, optimizer, data, epochs, early_stopping, need_regularizations=False, l1=0.0, l2=0.0, l1_params_list=[], l2_params_list=[]):
    train_acc = val_acc = test_acc = 0
    val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(epochs):
        # train
        if need_regularizations:
            params = model.named_parameters()
            l1_norm, l2_norm = regularization(params, l1, l2, l1_params_list, l2_params_list)
            t_st=time.time()
            train(model, optimizer, data.x, data.y, data.train_mask, l1_norm, l2_norm)
        else:
            t_st=time.time()
            train(model, optimizer, data.x, data.y, data.train_mask)
            
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        # val
        losses = []   # train, val, test
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            if need_regularizations:
                params = model.named_parameters()
                l1_norm, l2_norm = regularization(params, l1, l2, l1_params_list, l2_params_list)
                acc, loss = evaluate(model, data.x, data.y, mask, l1_norm, l2_norm)
            else:
                acc, loss = evaluate(model, data.x, data.y, mask)
            accs.append(acc)
            losses.append(loss)

        print(
            f'epoch : {epoch} \t train_loss : {losses[0]:.4f} \t train_acc : {accs[0]:.4f} \t val_acc : {accs[1]:.4f} \t test_acc : {accs[2]:.4f}')
        
        if losses[1] < val_loss:
            val_loss = losses[1]
            train_acc = accs[0]
            val_acc = accs[1]
            test_acc = accs[2]

        val_loss_history.append(losses[1])
        val_acc_history.append(accs[1])

        if early_stopping > 0 and epoch > early_stopping:
            tmp = torch.Tensor(
                val_loss_history[-(early_stopping+1):-1])
            if losses[1] > tmp.mean().item():
                print('The sum of epochs:', epoch+1)
                break
    print(
        f'best_val_loss : {val_loss:.4f} \t train_acc : {train_acc:.4f} \t val_acc : {val_acc:.4f} \t test_acc : {test_acc:.4f}')
    
    return val_acc, test_acc, time_run


def train(model, optimizer, x, y, mask, l1norm=0.0, l2norm=0.0):
    model.train()
    optimizer.zero_grad()
    pre = model(x)[mask]
    loss = nn.functional.cross_entropy(pre, y[mask]) + l1norm + l2norm
    loss.backward()
    optimizer.step()
    return pre


def evaluate(model, x, y, mask, l1norm=0.0, l2norm=0.0):
    model.eval()
    logits = model(x)[mask]
    pred = logits.max(1)[1]
    acc = pred.eq(y[mask]).sum().item()/mask.sum().item()
    loss = (nn.functional.cross_entropy(logits, y[mask]) + l1norm + l2norm).item()
    return acc, loss


def init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_config(model):
    config = edict(
        yaml.load(open(('../config/'+model+'.yaml'), 'r'), Loader=yaml.FullLoader))
    return config


def group_param(model, **kwargs):
    groups = {}

    for pname, p in model.named_parameters():
        idx = [i for i, gname in enumerate(kwargs.values()) if gname in pname]
        if len(idx):
            group_name = list(kwargs.values())[idx[0]]
            if group_name not in groups:
                groups[group_name] = [p]
            else:
                groups[group_name].append(p)

    params = model.parameters()
    params_id = []
    for group_item in groups.values():
        params_id += list(map(id, group_item))

    other_params = list(filter(lambda p: id(p) not in params_id, params))
    groups["other_params"] = other_params

    return groups


def RunTimes(seeds, RunOnce, data, config, device):
    
    results = []
    time_results = []

    for RP in tqdm(range(len(seeds))):
        seed=seeds[RP]
        val_acc, test_acc, time_run = RunOnce(seed, data, config, device)
        results.append([test_acc, val_acc])
        time_results.append(time_run)

    test_acc_mean, val_acc_mean = np.mean(results, axis=0)
    test_acc_std, val_acc_std = np.std(results, axis=0)

    print(f'{config.model.name} on dataset {config.dataset}, in {len(seeds)} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {test_acc_std:.4f}  \t val acc mean = {val_acc_mean:.4f} ± {val_acc_std:.4f}')

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum = run_sum + sum(i) - i[0] - i[1] - i[2]
        epochsss = epochsss + len(i) - 3

    print("each epoch avg_time:", 1000 * run_sum / epochsss,"ms")
