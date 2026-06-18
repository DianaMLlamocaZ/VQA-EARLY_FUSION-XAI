from utils.utils import load_data_loaders
from src.load_dataset import DatasetVQA

from torch.utils.data import DataLoader

def load_dls(batch_train:int,batch_valid:int,batch_test:int):

    train_data,valid_data,test_data,vocab=load_data_loaders()
    
    train_ds=DatasetVQA(data=train_data,vocab_train=vocab,path="./data/train")
    valid_ds=DatasetVQA(data=valid_data,vocab_train=vocab,path="./data/train")
    test_ds=DatasetVQA(data=test_data,vocab_train=vocab,path="./data/test")

    train_dl=DataLoader(train_ds,batch_size=batch_train,shuffle=True)
    valid_dl=DataLoader(valid_ds,batch_size=batch_valid,shuffle=False)
    test_dl=DataLoader(test_ds,batch_size=batch_test,shuffle=False)
    
    return train_dl,valid_dl,test_dl,vocab
