import pytorch_lightning as pl
from typing import List, Tuple, Callable, Dict
from torch import nn, optim
from torch.nn import functional as F
import torch
from collections import OrderedDict

def mae(yhat, y):
    return torch.abs(yhat-y).mean()

def get_emb_size(cardinality: int, maximum: int = 20):
    return (cardinality+1, min(int(cardinality**(1/2)), maximum))


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


class LstmModel(pl.LightningModule):
    def __init__(self, 
                 static_features: int,
                 time_features: int,
                 categories: Dict[str, int],
                 max_emb_sz: int,
                 hidden_dim: int,
                 encoder_dim: int,
                 emb_dropout: float = 0.,
                 dropout: float = 0.,
                 lr: float = 0.01,
                 wd: float = 0.,
                 out_features: int = 1,
                 n_layers: int = 2):
        super().__init__()
        
        self.lr = lr
        self.wd = wd
        
        self.emb = EmbeddingLayer(categories, dropout=emb_dropout, max_emb_sz=max_emb_sz)
        self.hidden_dim = hidden_dim
        # lstm
        self.net = nn.LSTM(time_features, hidden_dim, batch_first=True)
        # decoder
        input_decoder = static_features + hidden_dim + self.emb.out_features
        self.output_layer = nn.Sequential(nn.Linear(input_decoder, encoder_dim),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(encoder_dim),
                                          nn.Dropout(dropout),
                                          nn.Linear(encoder_dim, out_features))

    def _init_hidden(self, bs):
        return next(self.parameters()).new(1, bs, self.hidden_dim).zero_()
        
    def init_hidden_state(self, bs):
        return (self._init_hidden(bs), self._init_hidden(bs))
    
    def forward(self, features,
                timeft,
                categories,
                target=None):        
        bs, sq, ft = timeft.size()
        hidden_state = self.init_hidden_state(bs)

        _, (final_state, _) = self.net(timeft, hidden_state)
        
        final_state.squeeze_(dim=0)
        categories = self.emb(categories)
        final_state = torch.cat((features, categories, final_state), dim=1)
        prediction = self.output_layer(final_state)
        # scaled prediction to 0 to 100
        prediction = torch.sigmoid(prediction) * 100

        return prediction
    
    def training_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss = mae(y_hat, batch['target'])
        self.log('train_mae', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(**batch)
        loss = mae(y_hat, batch['target'])
        self.log('valid_mae', loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer