
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import gc
from models.cont_emb_stack import RegressionEmbModel ###no
from data.dataset import Dataset ###no
from train.core import ModelOutput ###no


def predict_dl(model: nn.Module, valid_dl: DataLoader):
    model.eval()
    with torch.no_grad():
        
        prediction = torch.cat([model(**batch)
                               for batch in valid_dl])
    return prediction.numpy()


def load_best_state(model, checkpoint_callback):
    print('loading model to best score')
    print(f'best score = {checkpoint_callback.best_model_score}')
    best_model_parameters = torch.load(checkpoint_callback.best_model_path)['state_dict']
    model.load_state_dict(best_model_parameters)
    return model
    

def run_emb_model_fn(config: Dict[str, Any],
                     train_data: pd.DataFrame,
                     valid_data: pd.DataFrame,
                     device: torch.device = torch.device('cpu')):
    
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # hyperparameters
    hp = config.hp
    map_dtypes = {'features': np.float32,
                  'categories': np.int64}
    
    # create dls
    train_ds = Dataset.from_dataframe(train_data,
                                      features=config.features,
                                      categories=config.categories,
                                      target=config.target_cols,
                                      weight=config.weight,
                                      dtypes_mapping=map_dtypes,
                                      device=device)

    train_dl = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                          num_workers=4)

    valid_ds = Dataset.from_dataframe(valid_data, 
                                      features=config.features,
                                      categories=config.categories,
                                      target=config.target_cols,
                                      dtypes_mapping=map_dtypes,
                                      device=device)

    valid_dl = DataLoader(valid_ds, batch_size=hp.batch_size,
                          shuffle=False, num_workers=4)

    categories = train_data.loc[:, config.categories].nunique().to_dict()

    model = RegressionEmbModel(cont_features=len(config.features),
                               categories=categories,
                               max_emb_sz=hp.max_emb_sz,
                               encoder_dim=hp.encoder_dim,
                               out_features=len(config.target_cols),
                               dropout=hp.dropout,
                               emb_dropout=hp.emb_dropout,
                               depth=hp.depth,
                               lr=hp.lr,
                               decrease_factor=hp.decrease_factor,
                               drop_decrease_factor=hp.drop_decrease_factor,
                               scale_output=hp.scale_output,
                               wd=hp.wd)
    print(model)

    patience = hp.early_stop_patience if hp.early_stop_patience is not None else 3
    early_stopping = EarlyStopping('valid_loss', patience=patience)
    checkpoint_callback  = ModelCheckpoint(monitor='valid_loss',
                                           save_top_k=3,
                                           save_weights_only=True)
    trainer = pl.Trainer(max_epochs=hp.epochs,
                         callbacks=[early_stopping, checkpoint_callback])

    trainer.fit(model, train_dl, valid_dl)

    # loading best model so far
    model = load_best_state(model, checkpoint_callback)
    
    def predict_fn(test_features: pd.DataFrame):
        test_ds = Dataset.from_dataframe(test_features,
                                         features=config.features,
                                         categories=config.categories,
                                         dtypes_mapping=map_dtypes,
                                         device=device)
        test_dl = DataLoader(test_ds, batch_size=hp.batch_size, shuffle=False)
        model.to(device=device)
        prediction = predict_dl(model, test_dl)
        del test_dl, test_ds
        return prediction

    valid_prediction = predict_dl(model, valid_dl)
    gc.collect()
    return ModelOutput(model, predict_fn, valid_prediction)