import numpy as np
import h5py
import torch
from my_ddc4.config import args
from my_ddc4.utils import CharVocab
from my_ddc4.model import AutoEncoder

with open('datasets/ChEMBL_training_set', "r") as f:
    data = f.read().splitlines()

vocabulary = CharVocab.from_data(data)
model = AutoEncoder(vocabulary, args).cuda(3)
model.load_state_dict(torch.load('torch_models/_train_060.pt'))
model = model.eval()
smile_list = model.forward(92248,300)
print(smile_list)
for i in range(len(smile_list)):
    with open('torch_datasets/smile_sample_my_ddc3', "a") as f:
        f.write(smile_list[i])
        f.write('\n')

for i in range(1000):
    smile_list = model.forward(1, 300)
    print(smile_list)
    with open('torch_datasets/smile_sample_one', "a") as f:
        f.write(smile_list[0])
        f.write('\n')
