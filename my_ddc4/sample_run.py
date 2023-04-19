import numpy as np
import h5py
import torch
from my_ddc4.config import args
from my_ddc4.utils import CharVocab
from my_ddc4.model import AutoEncoder

def Sampel_run(n,model_path,smiles_out_path):
    '''vocabulary'''
    with open('../datasets/ChEMBL_training_set', "r") as f:
        data = f.read().splitlines()
    vocabulary = CharVocab.from_data(data)

    '''load model'''
    model = AutoEncoder(vocabulary, args).cuda(0)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    '''sample smiles and save'''
    smile_list = model.forward(n,300)#92248
    print('sample smlies finish!')
    for i in range(len(smile_list)):
        with open(smiles_out_path, "a") as f:
            f.write(smile_list[i])
            f.write('\n')
    print('save smiles finish!')

    '''sample repeat and save'''
    # for i in range(1000):
    #     smile_list = model.forward(n, 300)
    #     print(smile_list)
    #     for i in range(len(smile_list)):
    #         with open(smiles_out_path, "a") as f:
    #             f.write(smile_list[i])
    #             f.write('\n')
    # print('save smiles finish!')


    return smile_list

if __name__ == '__main__':
    Sampel_run(n=100,
               model_path='../torch_models_ddc4/_train_010.pt',
               smiles_out_path='../torch_datasets_ddc4/point_smiles_sample010')
