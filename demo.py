from my_ddc3.model import AAE
from my_ddc3.trainer import AAETrainer
from my_ddc3 import config
from moses.utils import CharVocab, Logger
configp = config.get_config()
def train_fit(train_data_name,val_data_name):
    with open(train_data_name, "r") as f:
        train_data = f.read().splitlines()
    with open(val_data_name, "r") as f:
        val_data = f.read().splitlines()
    vocabulary = CharVocab.from_data(train_data)
    model = AAE(vocabulary,configp).cuda()
    aaetrainer = AAETrainer(configp)
    model_out = aaetrainer.fit(model = model,train_data = train_data, val_data=val_data)
    return model_out

if __name__ == '__main__':
    train_data_name = 'datasets/ChEMBL_training_set'
    val_data_name = 'datasets/ChEMBL_validation_set'
    train_fit(train_data_name,val_data_name)







import  torch
# model = torch.load('model.pkl')
model = AAE(vocabulary,configp)
model.load_state_dict(torch.load('models/_040.pt'))
model = model.cuda()
model = model.eval()

lens = torch.tensor([len(t) - 1 for t in train_data],
                    dtype=torch.long)
# string_list = [CharVocab.from_data(train_data).string2ids(c, add_bos=False, add_eos=False) for c in train_data]
vocabulary = CharVocab.from_data(train_data)
def string2tensor(string, device='model'):
    ids = vocabulary.string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(ids, dtype=torch.long)
    return tensor

string_list = [string2tensor(i) for i in train_data]
# len(vocabulary)#22
# ids = torch.tensor(string_list, dtype=torch.long)
# tensor = torch.tensor(ids, dtype=torch.long)
embedding_layer = torch.nn.Embedding(22,22,padding_idx=vocabulary.pad)
import torch.nn.utils.rnn as rnn_utils
def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length
train_data, data_length = collate_fn(string_list)
import numpy as np
train_data = np.array(train_data)
train_data = torch.tensor(train_data,dtype=torch.long)
em = embedding_layer(train_data)
em.shape#torch.Size([10, 72, 22])
em[0].shape#torch.Size([72, 22])
x, lengths, hiddens = model(train_data,data_length)

import torch.nn.functional as F
def tensor2string(tensor):
    ids = tensor.tolist()
    string = vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

    return string
n_batch = 10#多少个句子
max_length = 100#每个句子长为100
def sample(n_batch, max_length=100):
    with torch.no_grad():
        starts = [torch.tensor([vocabulary.bos],
                               dtype=torch.long)
                  for _ in range(n_batch)]#[tensor([18]), tensor([18])]

        starts = torch.tensor(starts, dtype=torch.long).unsqueeze(1)#tensor([[18],[18]])

        new_smiles_list = [
            torch.tensor(vocabulary.pad, dtype=torch.long,).repeat(max_length + 2)
            for _ in range(n_batch)]#[tensor([100个20]),tensor([100个20])]

        for i in range(n_batch):
            new_smiles_list[i][0] = vocabulary.bos#第一个为18

        len_smiles_list = [1 for _ in range(n_batch)]# [1, 1]
        lens = torch.tensor([1 for _ in range(n_batch)],
                            dtype=torch.long)#tensor([1, 1])
        end_smiles_list = [False for _ in range(n_batch)]#[False, False]

        hiddens = None
        for i in range(1, max_length + 1):
            output, _, hiddens = model(train_data,data_length)
            #output.shape#torch.Size([10, 72, 22])

            # probabilities
            probs = [F.softmax(o, dim=-1) for o in output]
            #probs#([10, 72, 22])
            # import numpy as np
            # np.array(probs)[0].shape#torch.Size([72, 22])

            # sample from probabilities
            ind_tops = [torch.multinomial(p, 1) for p in probs]
            # print(len(ind_tops))#10
            # print(ind_tops[0].shape)#torch.Size([72, 1])

            for j, top in enumerate(ind_tops):
                if not end_smiles_list[j]:
                    top_elem = top[0].item()#首字母首字母的编号
                    if top_elem == vocabulary.eos:
                        end_smiles_list[j] = True

                    new_smiles_list[j][i] = top_elem
                    len_smiles_list[j] = len_smiles_list[j] + 1

            starts = torch.tensor([t.numpy() for t in ind_tops], dtype=torch.long,).unsqueeze(1)
            # starts = torch.tensor(ind_tops, dtype=torch.long, ).unsqueeze(1)

        new_smiles_list = [new_smiles_list[i][:l]
                           for i, l in enumerate(len_smiles_list)]
        return [tensor2string(t) for t in new_smiles_list]

########################################################
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from my_ddc.utils import set_torch_seed_to_all_gens
from my_ddc.utils import CharVocab, Logger

with open('datasets/ChEMBL_training_set', "r") as f:
    data = f.read().splitlines()[:20]
vocabulary = CharVocab.from_data(data)
def string2tensor(string):
    ids = vocabulary.string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(
        ids, dtype=torch.long)
    return tensor

def collate(data):
    data.sort(key=lambda x: len(x), reverse=True)

    tensors = [string2tensor(string)
               for string in data]
    lengths = torch.tensor([len(t) for t in tensors],
                           dtype=torch.long,
                          )

    encoder_inputs = pad_sequence(tensors,
                                  batch_first=True,
                                  padding_value=vocabulary.pad)
    encoder_input_lengths = lengths - 2

    decoder_inputs = pad_sequence([t[:-1] for t in tensors],
                                  batch_first=True,
                                  padding_value=vocabulary.pad)
    decoder_input_lengths = lengths - 1

    decoder_targets = pad_sequence([t[1:] for t in tensors],
                                   batch_first=True,
                                   padding_value=vocabulary.pad)
    decoder_target_lengths = lengths - 1
    print('encoder_inputs',encoder_inputs.shape)#torch.Size([20, 68])
    print('decoder_inputs', decoder_inputs.shape)#torch.Size([20, 67])
    print('decoder_targets', decoder_targets.shape)#torch.Size([20, 67])

    return (encoder_inputs, encoder_input_lengths), \
           (decoder_inputs, decoder_input_lengths), \
           (decoder_targets, decoder_target_lengths)




















import  torch
with open('datasets/ChEMBL_training_set', "r") as f:
    train_data = f.read().splitlines()
vocabulary = CharVocab.from_data(train_data)
model = AAE(vocabulary,configp)
model.load_state_dict(torch.load('torch_models/_train_040.pt'))
model = model.cuda()
model = model.eval()
a = model.sample(n_batch=10,max_len=100)