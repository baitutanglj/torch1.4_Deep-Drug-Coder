import numpy as np
import h5py
import torch
from my_ddc4.config import args
from my_ddc4.utils import CharVocab
from my_ddc4.model import AutoEncoder

def point_smiles():
    with h5py.File('../torch_datasets/test_enc_output_gen.hdf5', "r") as f:
        data = f["vec"][:]
    x1 = data[0]
    x2 = data[2000]
    t = np.linspace(2,202,num=100)
    point = []
    for i in t:
        xi = x1 + (x2-x1)*i
        point.append(xi)

    '''save point gen'''
    point_gen_name = '../torch_datasets/point_gen.hdf5'
    with h5py.File(point_gen_name, 'w') as f:
        dset = f.create_dataset("vec", data=np.array(point))
    print("save point_gen finish")


    '''sample smiles'''
    with open('../datasets/ChEMBL_training_set', "r") as f:
        data = f.read().splitlines()
    vocabulary = CharVocab.from_data(data)
    model = AutoEncoder(vocabulary, args).cuda(3)
    model.load_state_dict(torch.load('../torch_models/_train_060.pt'))
    model = model.eval()
    smile_list = model.forward(100,300)
    print(smile_list)
    for i in range(len(smile_list)):
        with open('../torch_datasets/point_smiles_sample', "a") as f:
            f.write(smile_list[i])
            f.write('\n')

if __name__ == '__main__':
    point_smiles()
