import torch.nn as nn
from my_ddc4.model import AutoEncoder
from my_ddc4.trainer import Trainer
from my_ddc4.config import args
from my_ddc4.utils import CharVocab


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
# with h5py.File('torch_datasets/test_enc_output_gen.hdf5', "r") as f:
#     data = f["vec"][:]
# data.shape
# data[:2]