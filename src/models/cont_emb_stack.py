import pytorch_lightning as pl
from typing import List, Tuple, Callable, Dict
from torch import nn, optim
from torch.nn import functional as F
import torch
from collections import OrderedDict
import numpy as np

def mae(yhat, y, weight=None):
    error = torch.abs(yhat-y)
    if weight is None:
        return error.mean()
    weight /= weight.sum()
    return (error * weight).sum()


def get_emb_size(cardinality: int, maximum: int = 20):
    emb_szs = int(np.ceil(cardinality**(1/2)))
    return (cardinality, min(emb_szs, maximum))


class EmbeddingLayer(nn.Module):
    def __init__(self, categories: Dict[str, int],
                 dropout: float = 0.,
                 max_emb_sz: int = 20,
                 stack_fn: Callable = torch.cat):
        super().__init__()

        emb_szs = OrderedDict({name: get_emb_size(size, max_emb_sz)
                              for name, size in categories.items()})
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.ModuleDict(OrderedDict({name: nn.Embedding(size, hidden_dim)
                                             for name, (size, hidden_dim) in emb_szs.items()}))
        self.stack_fn = stack_fn
        self.out_features = sum([hidden_dim for (_, hidden_dim) in emb_szs.values()])
    
    def forward(self, x):
        output = self.stack_fn([emb(x[:, e]) for e, emb in enumerate(self.emb.values())], dim=-1)
        output = self.dropout(output)
        return output


def build_feed_forward(layers: List[int],
                       use_bn: bool = False,
                       input_bn: bool = False,
                       act_fn: Callable = nn.ReLU,
                       dropout: List[int] = None):
    model = []

    if input_bn:
        model.append(nn.BatchNorm1d(layers[0]))

    if isinstance(dropout, float):
        dropout = [dropout] * len(layers)

    for L in range(len(layers)-1):
        input_dim, output_dim = layers[L], layers[L+1]
        linear = nn.Linear(input_dim, output_dim)
        model.append(linear)
        if L+1 < len(layers) -1:
            model.append(act_fn())
            if use_bn:
                model.append(nn.BatchNorm1d(output_dim))
            if dropout is not None:
                model.append(nn.Dropout(dropout[L]))

    return nn.Sequential(*model)


def compute_factor_layers(input_dim,
                          depth: int = 1,
                          factor: float = 1.,
                          dtype=np.int64):
    layers_sizes = np.array([input_dim] * (depth-1))
    layers_pct = np.array([[factor] * (depth-1)])
    layers_pct = np.cumprod(layers_pct)
    layers_sizes = list((layers_sizes * layers_pct).astype(dtype))
    return layers_sizes


def scale_to_100(x):
    return torch.sigmoid(x) * 100


class EmbModelModule(pl.LightningModule):
    def __init__(self, 
                 cont_features: int,
                 categories: Dict[str, int],
                 encoder_dim: int,
                 depth: int,
                 decrease_factor: float = 1.,
                 drop_decrease_factor: float = 1.,
                 emb_dropout: float = 0.,
                 max_emb_sz: int = 20,
                 dropout: float = 0.,
                 lr: float = 0.01,
                 wd: float = 0.,
                 out_features: int = 1,
                 output_act_fn: Callable = None):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.output_act_fn = output_act_fn
        
        self.emb = EmbeddingLayer(categories, dropout=emb_dropout, max_emb_sz=max_emb_sz)
        
        layers_sizes = compute_factor_layers(encoder_dim, depth=depth, factor=decrease_factor)
        drop_sizes = compute_factor_layers(dropout, depth=depth,
                                           factor=drop_decrease_factor, dtype=float)
        daily_features = self.emb.out_features + cont_features
        dropout_layers = [dropout] + drop_sizes
        daily_decoder_layers = [daily_features, encoder_dim] + layers_sizes + [out_features]
        self.output_layer = build_feed_forward(daily_decoder_layers, use_bn=False,
                                                dropout=dropout_layers,
                                                act_fn=self._act_fn)

    def _act_fn(self):
        return nn.LeakyReLU(0.2)

    def forward(self, features,
                categories,
                target=None,
                weight=None):

        categories = self.emb(categories)

        features = torch.cat((features, categories), dim=-1)
        
        prediction = self.output_layer(features)
        if self.output_act_fn is not None:
            prediction = self.output_act_fn(prediction)
        return prediction


class RegressionEmbModel(EmbModelModule):
    def __init__(self, 
                 cont_features: int,
                 categories: Dict[str, int],
                 encoder_dim: int,
                 depth: int,
                 decrease_factor: float = 1.,
                 drop_decrease_factor: float = 1.,
                 emb_dropout: float = 0.,
                 max_emb_sz: int = 20,
                 dropout: float = 0.,
                 lr: float = 0.01,
                 wd: float = 0.,
                 out_features: int = 1,
                 scale_output: bool = True):

        output_act_fn = (scale_to_100 if scale_output else None)
        super().__init__(cont_features=cont_features, categories=categories,
                         max_emb_sz=max_emb_sz, encoder_dim=encoder_dim, out_features=out_features,
                         dropout=dropout, emb_dropout=emb_dropout, depth=depth, lr=lr,
                         decrease_factor=decrease_factor, drop_decrease_factor=drop_decrease_factor,
                         wd=wd, output_act_fn=output_act_fn)
        
    def training_step(self, batch, batch_idx):
        y_hat = self(**batch)
        weight = batch['weight'] if 'weight' in batch else None
        loss = mae(y_hat, batch['target'], weight)
        self.log('train_loss', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss = mae(y_hat, batch['target'])
        self.log('valid_loss', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer