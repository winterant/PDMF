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


class EleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(dim))
        self.bias = nn.Parameter(torch.rand(dim))

    def forward(self, input):
        return input * self.weight + self.bias


class ProposedModel(nn.Module):

    def __init__(self, sample_shapes: list, num_classes, num_train, fc_hidden, clustering_dim, clustering_label, alpha_1, alpha_2, sigma, lambda_1, lambda_2):
        super().__init__()
        self.sample_shapes = sample_shapes  # each view's dim
        self.num_views = len(sample_shapes)
        self.num_classes = num_classes
        self.clustering_dim = clustering_dim
        self.clustering_label = clustering_label
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.sigma = sigma
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        # W and z_n
        self.W = nn.ParameterList([nn.Parameter(torch.rand(clustering_dim, fc_hidden)) for i in sample_shapes])
        self.clustering = nn.Embedding(num_train, clustering_dim)
        nn.init.normal_(self.clustering.weight.data)
        # decoding for pre-training
        self.decoder = ParallelModule([
            nn.Sequential(
                nn.Linear(fc_hidden, shape[-1]),
                nn.ReLU()
            ) for shape in sample_shapes
        ])
        # encoding for fine-tuning
        self.encoder = ParallelModule([
            nn.Sequential(
                nn.Linear(shape[-1], fc_hidden),
                nn.BatchNorm1d(fc_hidden),
                nn.ReLU(),
                nn.Linear(fc_hidden, clustering_dim),
                nn.BatchNorm1d(clustering_dim),
                nn.ReLU()
            )for shape in sample_shapes
        ])
        self.bn_trans = ParallelModule([
            nn.Sequential(
                EleLinear(clustering_dim),
                nn.Sigmoid()
            ) for i in sample_shapes
        ])
        self.classifier = nn.Linear(clustering_dim, num_classes)

    def pre_training(self, x, y, index):
        # Decoding
        z_ = self.clustering(index)
        views = {i: z_ @ self.W[i].abs() for i in range(len(x))}
        views = self.decoder(views)

        # `loss_n_r` is for reconstruction loss
        # `loss_w` is for structured sparseness regularizer
        loss_n_r, loss_w = torch.zeros_like(y).float(), torch.tensor(0.0, device=y.device)
        for i in range(len(views)):
            loss_n_r += torch.mean((views[i] - x[i])**2, dim=-1)
            loss_w += torch.max(self.W[i].abs(), dim=-1)[0].sum()

        # `loss_n_c` is for supervision information
        inner_prod = z_ @ self.clustering.weight.T
        y_pred = torch.cat(
            [inner_prod[:, self.clustering_label == i].mean(dim=-1, keepdim=True) for i in range(self.num_classes)],
            dim=-1
        ).argmax(dim=-1)
        distance = torch.zeros_like(y).float()
        for i in range(len(y)):
            distance[i] += inner_prod[i, self.clustering_label == y_pred[i].item()].mean().item()
            distance[i] -= inner_prod[i, self.clustering_label == y[i].item()].mean().item()
        loss_n_c = F.relu(torch.not_equal(y, y_pred).float() + distance)

        # Total loss
        loss = torch.mean(loss_n_r + self.alpha_1 * loss_n_c) + self.alpha_2 * loss_w
        return {'loss': loss}

    def forward(self, x, y=None, index=None, epoch=50):
        views = self.encoder(x)
        fusion = sum(views.values()) / len(views)
        evidence = self.classifier(fusion)
        ret = {
            'fusion': fusion,
            'evidence': evidence,
            'prob': evidence.softmax(dim=-1),
            'y': evidence.argmax(dim=-1),
        }
        if y is not None:
            w_c_v = {k: v.abs().mean(dim=-1) for k, v in enumerate(self.W)}  # shape of each view: (clustering_dim)
            bn_transform = self.bn_trans(w_c_v)
            loss_bn_with_w = 0
            loss_bn_l1norm = 0
            for k in range(self.num_views):  # for each view
                last_bn = None
                for net in self.encoder.parallel_modules[k]:  # nets of current view
                    if (isinstance(net, nn.BatchNorm1d)):
                        last_bn = net  # Find the last bn layer
                loss_bn_with_w += torch.mean((last_bn.weight - bn_transform[k])**2)
                loss_bn_l1norm += last_bn.weight.mean()
            loss_bn_with_w /= self.num_views

            loss_n_a = torch.sum((fusion - self.clustering(index))**2, dim=-1) + self.sigma * loss_bn_with_w

            ret['loss'] = torch.mean(
                self.lambda_1 * F.cross_entropy(evidence, y)
                + max(0, 1 - epoch / 10) * loss_n_a
            )
            + self.lambda_2 * loss_bn_l1norm  # todo: loss of L1-norm for bn's parameter
        return ret
