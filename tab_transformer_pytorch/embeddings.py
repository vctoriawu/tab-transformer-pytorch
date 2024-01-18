import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class PLE(nn.Module):
    def __init__(self, n_bins=10):
        super(PLE, self).__init__()
        self.n_bins = n_bins

    def adapt(self, data, y=None, task='classification', tree_params={}):
        if y is not None:
            if task == 'classification':
                dt = DecisionTreeClassifier(max_leaf_nodes=self.n_bins, **tree_params)
            elif task == 'regression':
                dt = DecisionTreeRegressor(max_leaf_nodes=self.n_bins, **tree_params)
            else:
                raise ValueError("This task is not supported")
            dt.fit(data, y)
            bins = torch.sort(torch.tensor(np.unique(dt.tree_.threshold))).values.float()
        else:
            interval = 1 / self.n_bins
            bins = torch.unique(torch.tensor([np.quantile(data, q) for q in np.arange(0.0, 1 + interval, interval)])).float()

        self.n_bins = len(bins)
        self.lookup_table = {i: bin_val for i, bin_val in enumerate(bins)}

    def forward(self, x):
        ple_encoding_one = torch.ones((x.size(0), self.n_bins), device=x.device)
        ple_encoding_zero = torch.zeros((x.size(0), self.n_bins), device=x.device)

        left_masks = []
        right_masks = []
        other_case = []

        for i in range(1, self.n_bins + 1):
            x_float = x.float().unsqueeze(1)
            left_mask = (x_float < self.lookup_table.get((i - 1), -1)) & (i > 1)
            right_mask = (x_float >= self.lookup_table.get(i, -1)) & (i < self.n_bins)
            v = (x_float - self.lookup_table.get((i - 1), -1)) / (self.lookup_table.get(i, -1) - self.lookup_table.get((i - 1), -1))
            left_masks.append(left_mask)
            right_masks.append(right_mask)
            other_case.append(v)

        left_masks = torch.stack(left_masks, dim=1).squeeze()
        right_masks = torch.stack(right_masks, dim=1).squeeze()
        other_case = torch.stack(other_case, dim=1).squeeze()

        other_mask = (right_masks == left_masks).logical_not_()
        enc = torch.where(left_masks, ple_encoding_zero, ple_encoding_one)
        enc = torch.where(other_mask, other_case, enc).view(-1, 1, self.n_bins)

        return enc

class Periodic(nn.Module):
    def __init__(self, emb_dim, n_bins=50, sigma=5):
        super(Periodic, self).__init__()
        self.n_bins = n_bins
        self.emb_dim = emb_dim
        self.sigma = sigma
        self.p = nn.Parameter(torch.randn(emb_dim, n_bins))
        self.l = nn.Parameter(torch.randn(emb_dim, n_bins * 2, emb_dim))

    def forward(self, inputs):
        v = 2 * math.pi * self.p[None] * inputs[..., None]
        emb = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        emb = torch.einsum('fne, bfn -> bfe', self.l, emb)
        emb = F.relu(emb)

        return emb

class NEmbedding(nn.Module):
    MASK_VALUE = -9999.0
    def __init__(
        self,
        num_features,
        X,
        y=None,
        task=None,
        emb_dim=64,
        emb_type='linear',
        n_bins=128,
        sigma=1,
        tree_params={},
        mask_prob=0.15
    ):
        super(NEmbedding, self).__init__()

        if emb_type not in ['linear', 'ple', 'periodic']:
            raise ValueError("This emb_type is not supported")
        
        self.num_features = num_features
        self.emb_type = emb_type
        self.emb_dim = emb_dim

        self.mask_prob = mask_prob
        
        # Initialise embedding layers
        if emb_type == 'ple':
            self.embedding_layers = nn.ModuleDict()
            self.linear_layers = nn.ModuleDict()
            for i in range(self.num_features):
                emb_l = PLE(n_bins)
                if y is None:
                    emb_l.adapt(X[:, i], tree_params=tree_params)
                else:
                    emb_l.adapt(X[:, i].view(-1, 1), y, task=task, tree_params=tree_params)
                
                lin_l = nn.Linear(emb_l.n_bins, emb_dim)
                relu_activation = nn.ReLU()
                
                self.embedding_layers[str(i)] = emb_l
                self.linear_layers[str(i)] = nn.Sequential(lin_l, relu_activation) #lin_l

        elif emb_type == 'periodic':
            # There's just 1 periodic layer
            self.embedding_layer = Periodic(n_bins=n_bins, emb_dim=emb_dim, sigma=sigma)
        else:
            # Initialise linear layer
            self.linear_w = nn.Parameter(torch.randn(self.num_features, 1, self.emb_dim))
            self.linear_b = nn.Parameter(torch.randn(self.num_features, 1))
    
    def mask_inputs(self, inputs, mask_prob):
        column_mask = torch.rand((inputs.size(1,)), device=inputs.device) < mask_prob
        column_mask = column_mask.unsqueeze(0).expand(inputs.size(0), -1)
        masked_inputs = torch.where(column_mask, self.MASK_VALUE, inputs)

        self.mask = column_mask
        self.masked_inputs = masked_inputs

    @property
    def get_mask(self):
        return self.masked_inputs, self.mask
    
    def embed_column(self, f, data):
        emb = self.linear_layers[f](self.embedding_layers[f](data))
        return emb
   
    def forward(self, x, mask=False):
        if (mask == True):
            self.mask_inputs(x, self.mask_prob)
            x =  self.masked_inputs
        else:
            x = x

        if self.emb_type == 'ple':
            emb_columns = [self.embed_column(str(i), x[:, i]) for i in range(self.num_features)]
            embs = torch.cat(emb_columns, dim=1)
            
        elif self.emb_type == 'periodic':
            embs = self.embedding_layer(x)
        else:
            embs = torch.einsum('f n e, b f -> bfe', self.linear_w, x)
            embs = F.relu(embs + self.linear_b)
            
        return embs