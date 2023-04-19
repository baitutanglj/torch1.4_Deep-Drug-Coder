'''
%load_ext autoreload
%autoreload 2
# Occupy a GPU for the model to be loaded
%env CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU ID, if occupied change to an available GPU ID listed under !nvidia-smi
%env CUDA_VISIBLE_DEVICES=0
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import rdkit
from rdkit import Chem
import h5py, ast, pickle
from ddc_pub import ddc_v3 as ddc

#import tensorflow as tf

# Load dataset
dataset_filename = "datasets/CHEMBL25_TEST_MOLS.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols = f["mols"][:]
#%%
Chem.Mol(binmols[0])
Chem.MolToSmiles(Chem.Mol(binmols[0]))
# All apriori known characters of the SMILES in the dataset
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
# Apriori known max length of the SMILES in the dataset
maxlen = 128
# Name of the dataset
name = "DRD2_TEST"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}
# mols = Chem.Mol(binmols[1])
# sm = Chem.MolToSmiles(mols)#'COc1ccccc1CNN1c2ccc(Cl)cc2N=C(N2CCN(C)CC2)c2ccccc21'
# #'COc1cc(N)c(Cl)cc1C(=O)NCCN1CCN(c2ccccc2OC)CC1'
# len(sm)
#%%

# Initialize a model
model = ddc.DDC(x              = binmols,      # input
                y              = binmols,      # output
                dataset_info   = dataset_info, # dataset information
                noise_std      = 0.1,          # std of the noise layer
                lstm_dim       = 128,          # breadth of LSTM layers
                dec_layers     = 3,            # number of decoding layers
                codelayer_dim  = 128,          # dimensionality of latent space
                batch_size     = 128)          # batch size for training

# model = ddc.DDC(x=binmols,
#                 y=binmols,
#                 #scaling=True,
#                 #pca=True,
#                 dataset_info=dataset_info,
#                 noise_std=0.1,
#                 lstm_dim=256,
#                 dec_layers=3,
#                 #td_dense_dim=0,
#                 batch_size=128,
#                 codelayer_dim=128)

#%%

model.fit(epochs              = 100,                            # number of epochs
          lr                  = 0.001,                          # initial learning rate for Adam, recommended
          model_name          = "new_heteroencoder_model",      # base name to append the checkpoints with
          checkpoint_dir      = "",                            # save checkpoints in the notebook's directory
          mini_epochs         = 10,                             # number of sub-epochs within an epoch to trigger lr decay
          save_period         = 50,                             # checkpoint frequency (in mini_epochs)
          lr_decay            = True,                           # whether to use exponential lr decay or not
          sch_epoch_to_start  = 500,                            # mini-epoch to start lr decay (bypassed if lr_decay=False)
          sch_lr_init         = 1e-3,                           # initial lr, should be equal to lr (bypassed if lr_decay=False)
          sch_lr_final        = 1e-6,                           # final lr before finishing training (bypassed if lr_decay=False)
          patience            = 25)                             # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

#%%

# Save the final model
model.save("new_heteroencoder_model")


