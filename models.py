from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelModule(nn.Module):
    def __init__(self, modules: Tuple[nn.Module]):
        super().__init__()
        self.parallel_modules = nn.ModuleList(modules)

    def forward(self, x: dict):
        y = dict()
        for k, v in x.items():
            y[k] = self.parallel_modules[k](v)
        return y


class ProposedModel(nn.Module):

    def __init__(self, sample_shapes: list, num_classes, num_train, fc_hidden, clustering_dim, alpha, beta, gamma):
        super().__init__()
        self.sample_shapes = sample_shapes
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 解码器用于pre-training
        self.decoder = ParallelModule([
            nn.Sequential(
                nn.Linear(fc_hidden, shape[-1]),
                nn.ReLU()
            ) for shape in sample_shapes
        ])
        # 编码器用于fine-tuning
        self.encoder = ParallelModule([
            nn.Sequential(
                nn.Linear(shape[-1], fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, clustering_dim),
                nn.ReLU()
            )for shape in sample_shapes
        ])
        self.W = nn.ParameterList([nn.Parameter(torch.rand(clustering_dim, fc_hidden)) for i in sample_shapes])
        self.clustering = nn.Embedding(num_train, clustering_dim)
        nn.init.normal_(self.clustering.weight.data)
        self.classifier = nn.Linear(clustering_dim, num_classes)
        self.bn = ParallelModule([nn.BatchNorm1d(clustering_dim) for s in sample_shapes])

    def pre_training(self, x, y, index):
        # Decoding
        z_ = self.clustering(index)
        views = {i: z_ @ self.W[i] for i in range(len(x))}
        views = self.decoder(views)

        # `loss_x` is for reconstruction loss
        # `loss_w` is for group constraint
        loss_x, loss_w = 0, 0
        for i in range(len(views)):
            loss_x += torch.sum((views[i] - x[i])**2, dim=-1)
            loss_w += torch.max(self.W[i], dim=-1)[0].sum()

        # `loss_z` is for z_
        loss_z = z_.abs().sum()

        # `loss_affinity` and `loss_away` is for Graph constraint
        A = torch.zeros(len(y), len(y), device=y.device)
        for c in range(self.num_classes):
            active_class = y.eq(c).float().unsqueeze(dim=0)  # 筛选出当前类别c
            A = A + (active_class.T @ active_class)  # 将类别c的邻接矩阵加入到总邻接矩阵
        L_intern = A.sum(dim=-1).diag_embed() - A
        loss_intern = torch.trace(z_.T @ L_intern @ z_)
        L_extern = (1 - A).sum(dim=-1).diag_embed() - (1 - A)
        loss_extern = - torch.trace(z_.T @ L_extern @ z_) * 0

        loss = loss_x.mean() + self.alpha * loss_w + self.beta * loss_z + self.gamma * (loss_intern + loss_extern)
        return {'loss': loss}

    def forward(self, x, y=None, index=None, epoch=100):
        views = self.encoder(x)
        for i in range(len(self.sample_shapes)):
            views[i] = views[i] * self.W[i].sum(dim=-1).softmax(dim=-1)
        views = self.bn(views)
        fusion = sum(views.values()) / len(views)
        evidence = self.classifier(fusion)
        ret = {
            'fusion': fusion,
            'evidence': evidence,
            'prob': evidence.softmax(dim=-1),
            'y': evidence.argmax(dim=-1),
        }
        if y is not None:
            z_ = self.clustering(index)
            delta = max(0, 1 - epoch / 10)  # 逐渐减小的系数
            ret['loss'] = F.cross_entropy(evidence, y)
            + delta * torch.norm(fusion - z_)**2
            # + delta * torch.norm(torch.zeros(1))**2  # todo bn's gamma
        return ret
