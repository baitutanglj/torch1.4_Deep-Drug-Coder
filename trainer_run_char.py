import torch.nn as nn
from char_rnn.model import AutoEncoder
from char_rnn.trainer import Trainer
from char_rnn.config import args
from char_rnn.utils import CharVocab


def train_fit(train_data_name,val_data_name):
    with open(train_data_name, "r") as f:
        train_data = f.read().splitlines()
    with open(val_data_name, "r") as f:
        val_data = f.read().splitlines()[:130000]
    vocabulary = CharVocab.from_data(train_data)
    model = AutoEncoder(vocabulary,args).cuda(0)
    trainer = Trainer(args)
    model_out = trainer.fit(model = model,train_data = train_data[:], val_data=val_data)
    return model_out

if __name__ == '__main__':
    train_data_name = 'datasets/ChEMBL_training_set'
    val_data_name = 'datasets/ChEMBL_validation_set'
    train_fit(train_data_name,val_data_name)


# import h5py
# with h5py.File('torch_datasets2/char_enc_output_001.hdf5', "r") as f:
#     data = f["vec"][:]
# data.shape
# data[:1000]