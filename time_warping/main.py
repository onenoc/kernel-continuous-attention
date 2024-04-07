import lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from data import DATASET
from matplotlib import pyplot as plt
from model import MODEL
import pdb
import mlflow

if __name__=="__main__":
    #1e-5 seems to work
    #encoder hidden size
    hidden_size = 128
    #encoder output size
    output_size = 128
    params = {
        'N':10000,
        'T':95,
        'bs':25,
        'nb_basis':64,
        'heads':2,
        'lr':1e-5,
        'hidden_size':128,
        'output_size':128,
        'inducing_points':128,
        'attention_mechanism':'kernel_sparsemax',
        'scheduler':'ReduceLROnPlateau',
        'optimizer':'RAdam'
    }
    #initialize dataset
    dataset = DATASET(params['N'],params['T'],params['nb_basis'])
    #split dataset into 0.75, 0.15, 0.1 using random split
    train, val, test = random_split(dataset, [0.75,0.15,0.1])
    #create train/val/test dataloaders, each with batch size bs
    train_loader = DataLoader(train, batch_size=params['bs'])
    val_loader = DataLoader(val, batch_size=params['bs'])
    test_loader = DataLoader(test, batch_size=params['bs'])
    #get a single example from the train_loader
    x,B,y = next(iter(train_loader))
    #kernel sparsemax isn't working well
    model = MODEL(params['T'], params['hidden_size'], params['output_size'], params['heads'], params['nb_basis'], params['inducing_points'], params['attention_mechanism'], params['heads'], params['optimizer'], params['lr'],params['scheduler'])
    #initialize trainer
    trainer = pl.Trainer(max_epochs=10,gradient_clip_val=0.1,precision=32)
    mlflow.pytorch.autolog()
    with mlflow.start_run():
        #mlflow log params
        mlflow.log_params(params)
        #train model
        trainer.fit(model, train_loader, val_loader)
        #test model
        trainer.test(model, test_loader)
