import numpy as np
import pandas as pd
import h5py
import torch
from my_ddc4.config import args
from my_ddc4.utils import CharVocab
from my_ddc4.model import AutoEncoder

def point_smiles(enc_path,point_gen_path,model_path,smiles_out_path):
    # with h5py.File('../torch_datasets/test_enc_output_gen.hdf5', "r") as f:
    #     data = f["vec"][:]
    data = pd.read_csv(enc_path, header=None)
    data = np.array(data[:], dtype=np.float32)

    x1 = data[3]
    x2 = data[2]
    t = np.linspace(-10,1,num=100)
    point = []
    for i in t:
        xi = x1 + (x2-x1)*round(i,2)
        point.append(xi)

    '''save point gen'''
    # point_gen_name = '../torch_datasets/point_gen.hdf5'
    # with h5py.File(point_gen_name, 'w') as f:
    #     dset = f.create_dataset("vec", data=np.array(point))
    df = pd.DataFrame(data=point)
    df.to_csv(point_gen_path, index=False, header=False)
    print("save point_gen finish")


    '''sample smiles'''
    with open('../datasets/ChEMBL_training_set', "r") as f:
        data = f.read().splitlines()
    vocabulary = CharVocab.from_data(data)
    model = AutoEncoder(vocabulary, args).cuda(1)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    smile_list = model.forward(100,300)
    print(smile_list)
    for i in range(len(smile_list)):
        with open(smiles_out_path, "a") as f:
            f.write(smile_list[i])
            f.write('\n')
    print('save smiles out finish!')

if __name__ == '__main__':
    point_smiles(enc_path='../torch_datasets_ddc4/test_enc_output_gen_010_130000',
                 point_gen_path='../torch_datasets_ddc4/point_gen_010',
                 model_path='../torch_models_ddc4/_train_010.pt',
                 smiles_out_path='../torch_datasets_ddc4/point_smiles_sample010'
                 )

