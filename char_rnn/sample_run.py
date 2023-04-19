import numpy as np
import h5py
import torch
from char_rnn.config import args
from char_rnn.utils import CharVocab
from char_rnn.model import AutoEncoder

def Sampel_run(n,model_path,smiles_out_path):
    '''vocabulary'''
    with open('../datasets/ChEMBL_training_set', "r") as f:
        data = f.read().splitlines()
    vocabulary = CharVocab.from_data(data)

    '''load model'''
    model = AutoEncoder(vocabulary, args).cuda(0)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # '''sample smile and save'''
    # smile_list = model.forward(n,300)
    # print(smile_list)
    # for i in range(len(smile_list)):
    #     with open(smiles_out_path, "a") as f:
    #         f.write(smile_list[i])
    #         f.write('\n')

    '''sample repeat and save'''
    for i in range(100):
        smile_list = model.forward(n, 300)
        print(smile_list)
        for i in range(len(smile_list)):
            with open(smiles_out_path, "a") as f:
                f.write(smile_list[i])
                f.write('\n')
    print('save finish!')
    return smile_list

if __name__ == '__main__':
    Sampel_run(n=1,
               model_path='../torch_models2/charrnn_020.pt',
               smiles_out_path='../torch_datasets2/smile_sample_charrnn_20one')