import os
import copy
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, distributed
import torch.distributed as dist

from data import MultiViewDataset
from models import ProposedModel


def train(gpu_rank, model, dataset: Dataset, valid: Union[float, int, Dataset] = 0.1, valid_interval: int = None,
          epochs=20, batch_size=256, lr=0.01, weight_decay=1e-6, save_weights_to=None):
    distributable = (gpu_rank is not None)
    # Initialize process and model
    if distributable:
        dist.init_process_group(backend='nccl',
                                init_method='tcp://localhost:23456',
                                world_size=torch.cuda.device_count(),
                                rank=gpu_rank)
        # torch.cuda.set_device(gpu_rank)  # Ensuring that each process exclusively works on a single GPU
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank])
    else:
        model = model.cuda()

    # Initialize data
    if isinstance(valid, Dataset):
        train_data, valid_data = dataset, valid
    else:
        num_valid = int(valid) if valid >= 1 else int(len(dataset) * valid)
        train_data, valid_data = random_split(dataset, [len(dataset) - num_valid, num_valid])
    if distributable:
        train_sampler = distributed.DistributedSampler(train_data)
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size * 4,)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if valid_interval is None:
        valid_interval = (len(train_data) + batch_size - 1) // batch_size

    # Start to train
    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    batch_counter = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch:2d}")
        if distributable:
            train_sampler.set_epoch(epoch)  # set shuffling seed for every epoch
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, y = batch['x'], batch['y']
            for k in x.keys():
                x[k] = x[k].cuda()
            y = y.cuda()
            ret = model(x, y)
            optimizer.zero_grad()
            ret['loss'].mean().backward()
            optimizer.step()
            train_loss += ret['loss'].mean().item() * len(batch['y'])
            correct += torch.sum(ret['prob'].cpu().argmax(dim=-1).eq(batch['y'])).item()
            num_samples += len(batch['y'])
            batch_counter += 1
            if batch_counter % valid_interval == 0:  # Validate
                valid_acc = validate(model, valid_loader)
                print(f"  {batch_counter:3d} batches passed and validating accuracy is {valid_acc:.4f}")
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        train_loss = train_loss / num_samples
        train_acc = correct / num_samples
        print(f'  train loss {train_loss:.4f}, train acc {train_acc:.4f}')

    dist.destroy_process_group()

    if save_weights_to is not None:
        os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
        torch.save(best_model_wts, save_weights_to)
    model.load_state_dict(best_model_wts)
    return model.module if distributable else model


def mp_train(multi_proc=True, *args, **kargs):
    if multi_proc:
        torch.multiprocessing.spawn(train, args=(*args, kargs), nprocs=torch.cuda.device_count(), join=True)
    else:
        train(None, *args, **kargs)


def validate(model, loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        correct, num_samples = 0, 0
        for batch in loader:
            x = batch['x']
            for k in x.keys():
                x[k] = x[k].to(device)
            ret = model(x)
            correct += torch.sum(ret['y'].cpu().eq(batch['y'])).item()
            num_samples += len(batch['y'])
    acc = correct / num_samples
    return acc


def predict(model, dataset, batch_size=1024, device='cuda'):
    loader = DataLoader(dataset, batch_size, shuffle=False)
    pred = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x']
            for k in x.keys():
                x[k] = x[k].to(device)
            ret = model(x)
            pred.append(ret['y'].cpu().numpy())
    pred = np.concatenate(pred)
    return pred


if __name__ == '__main__':
    train_data = MultiViewDataset(train=True)
    valid_data = MultiViewDataset(train=False)
    pdmf = ProposedModel([s.shape for s in train_data[0]['x'].values()], num_classes=10)
    pdmf = mp_train(pdmf, train_data, valid_data)
    pred = predict(pdmf, valid_data)
    print('prediction:', pred)
