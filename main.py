import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MultiViewDataset
from models import ProposedModel


def train(model, train_loader, valid_loader, pre_epochs=50, epochs=50, save_weights_to=None, device='cuda'):
    model = model.to(device)

    # Pre-training
    optimizer = torch.optim.SGD([
        {'params': (p for n, p in model.named_parameters() if 'weight' in n), 'weight_decay': 1e-6},
        {'params': (p for n, p in model.named_parameters() if 'weight' not in n)}
    ], lr=0.1)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.1)
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
            for i in range(len(x)):
                model.W[i].data.clamp_(min=0)
            model.clustering.weight.data.clamp_(0, 1)
            train_loss += ret['loss'].mean().item() * len(y)
            num_samples += len(y)
        step_lr.step()
        print(f'Epoch {epoch:3d}: lr {step_lr.get_last_lr()[0]:.6f}, pre-training loss {train_loss/num_samples:.6f}')

    debug_pre_W = [w.clone().detach().data for w in model.W]
    debug_pre_W_pooling = torch.stack([w.clone().detach().data.sum(dim=-1) for w in model.W], dim=1)
    debug_clustering_weight = model.clustering.weight[:50, :]
    # debug_clustering_weight = torch.cat([debug_clustering_weight.cpu(), torch.Tensor(train_loader.dataset[:50]['y']).unsqueeze(0)], dim=0)  # 显示出类别

    # Fine-tuning
    model.W.requires_grad_(False)  # W不参与fine-tuning
    optimizer = torch.optim.Adam([
        {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' in n), 'weight_decay': 1e-3},
        {'params': (p for n, p in model.named_parameters() if p.requires_grad and 'weight' not in n)},
    ], lr=0.001)
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
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


def experiment(data_path, hidden=512, clu_dim=128, alpha=0.01, beta=0.05, gamma=0.1):
    train_data = MultiViewDataset(data_path=data_path, train=True)
    valid_data = MultiViewDataset(data_path=data_path, train=False)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1024)
    pdmf = ProposedModel([s.shape for s in train_data[0]['x'].values()],
                         num_train=len(train_data),
                         num_classes=len(set(train_data.y)),
                         fc_hidden=hidden,
                         clustering_dim=clu_dim,
                         alpha=alpha,  # L-∞ norm for W
                         beta=beta,  # L-1 norm for Z
                         gamma=gamma,  # graph norm for Z
                         )
    print('---------------------------- Experiment ------------------------------')
    print('Dataset:', data_path)
    print('Number of views:', len(train_data.x), ' views with dims:', [v.shape[-1] for v in train_data.x.values()])
    print('Number of classes:', len(set(train_data.y)))
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')
    for n, p in pdmf.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')
    pdmf = train(pdmf, train_loader, valid_loader, pre_epochs=50, epochs=50)
    pred, acc = validate(pdmf, valid_loader)
    print('predicting accuracy is', acc)


if __name__ == '__main__':
    # experiment(data_path="dataset/handwritten_6views_train_test.mat")
    # experiment(data_path="dataset/CUB_c10_2views_train_test.mat")
    # experiment(data_path="dataset/PIE_train_test.mat")
    # experiment(data_path="dataset/2view-caltech101-8677sample_train_test.mat")
    # experiment(data_path="dataset/scene15_mtv_train_test.mat")
    # experiment(data_path="dataset/HMDB51_HOG_MBH_train_test.mat")
    experiment(data_path="toy-example", hidden=12, clu_dim=9, alpha=0.01, beta=0.05, gamma=0.1)
