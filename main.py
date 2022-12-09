import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MultiViewDataset
from models import ProposedModel


def train(model, train_loader, valid_loader, pre_epochs=20, epochs=50, save_weights_to=None, device='cuda'):
    model = model.to(device)

    # Pre-training
    optimizer = torch.optim.SGD([
        {'params': (p for n, p in model.named_parameters() if 'weight' in n), 'weight_decay': 1e-4},
        {'params': (p for n, p in model.named_parameters() if 'weight' not in n)}
    ], lr=0.0001)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
    model.train()
    for epoch in range(pre_epochs):
        train_loss, num_samples = 0, 0
        for batch in train_loader:
            x, y, index = batch['x'], batch['y'], batch['index']
            for k in x.keys():
                x[k] = x[k].to(device)
            y = y.to(device)
            index = index.to(device)
            ret = model.pre_training(x, y, index)
            optimizer.zero_grad()
            ret['loss'].mean().backward()
            optimizer.step()
            train_loss += ret['loss'].mean().item() * len(y)
            num_samples += len(y)
        step_lr.step()
        print(f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, pre-training loss {train_loss/num_samples:.6f}')

    # Fine-tuning
    model.W.requires_grad_(False)  # W drop out of fine-tuning
    optimizer = torch.optim.Adam([
        {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n), 'weight_decay': 1e-2},
        {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
    ], lr=0.01)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=23, gamma=0.1)
    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, y, index = batch['x'], batch['y'], batch['index']
            for k in x.keys():
                x[k] = x[k].to(device)
            y = y.to(device)
            index = index.to(device)
            ret = model(x, y, index, epoch)
            optimizer.zero_grad()
            ret['loss'].mean().backward()
            optimizer.step()
            train_loss += ret['loss'].mean().item() * len(y)
            num_samples += len(y)
            correct += torch.sum(ret['prob'].argmax(dim=-1).eq(y)).item()
        train_loss = train_loss / num_samples
        train_acc = correct / num_samples
        pred, valid_acc = validate(model, valid_loader)
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        step_lr.step()
        print(f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, train loss {train_loss:.4f}, train acc {train_acc:.4f}, valid acc {valid_acc:.4f}')

    if save_weights_to is not None:
        os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
        torch.save(best_model_wts, save_weights_to)
    model.load_state_dict(best_model_wts)
    return model


def validate(model, loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        pred = []
        correct, num_samples = 0, 0
        for batch in loader:
            x = batch['x']
            for k in x.keys():
                x[k] = x[k].to(device)
            ret = model(x)
            pred.append(ret['y'].cpu().numpy())
            correct += torch.sum(ret['y'].cpu().eq(batch['y'])).item()
            num_samples += len(batch['y'])
    pred = np.concatenate(pred)
    acc = correct / num_samples
    return pred, acc


if __name__ == '__main__':
    train_data = MultiViewDataset(data_path='dataset/handwritten_6views_train.pkl')
    valid_data = MultiViewDataset(data_path='dataset/handwritten_6views_test.pkl')
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1024)
    pdmf = ProposedModel([s.shape for s in train_data[0]['x'].values()],
                         num_train=len(train_data),
                         num_classes=len(set(train_data.y)),
                         fc_hidden=512,
                         clustering_dim=128,
                         clustering_label=train_data.y,
                         alpha_1=0.01,  # L_n_c
                         alpha_2=0.00001,  # Loss W by 1-inf norm
                         sigma=0.01,  # Loss bn with pretrained W
                         lambda_1=0.1,  # Loss of classify by cross entropy
                         lambda_2=0.01  # Loss bn by L1-norm
                         )
    print('---------------------------- Experiment ------------------------------')
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of classes:', len(set(train_data.y)))
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')
    for n, p in pdmf.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    pdmf = train(pdmf, train_loader, valid_loader)
    pred, acc = validate(pdmf, valid_loader)
    print('predicting accuracy is', acc)
