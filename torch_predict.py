import torch
import  torch.nn as nn
from data_gen import Data_generate
from torch.utils.data import DataLoader
from parser import args
from vectorizers import SmilesVectorizer
import numpy as np
import pandas as pd
import h5py
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def test_predict():
    testdata = Data_generate("datasets/DRD2_TRAIN_MOLS.h5")
    testloader = DataLoader(dataset=testdata,batch_size=args.batch_size,shuffle=False,num_workers=8,drop_last=False)#139
    # model = Model_mol_all(args)
    # model_dict=model.load_state_dict(torch.load("model_params.pkl"))
    #加载整个模型
    # model = torch.load('model.pkl')
    # print(model_dict)
    # model.cuda()
    # model.eval()
    encoder = torch.load('torch_models/encoder.pkl').cuda()
    decoder = torch.load('torch_models/decoder.pkl').cuda()
    encoder = encoder.eval()
    decoder = decoder.eval()
    criterion = nn.CrossEntropyLoss()
    maxlen = 133
    charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
    smilesvec1 = SmilesVectorizer(
        canonical=False,
        augment=True,
        maxlength=maxlen,
        charset=charset,
        binary=False,
    )
    with torch.no_grad():
        test_losses = []
        enc_output_list = []
        pred_list = []
        y_pre_list = []
        correct = 0
        for batch,(enc_input, dec_input, dec_output) in enumerate(testloader):
            enc_x, dec_x, dec_y = enc_input.cuda(), dec_input.cuda(), dec_output.cuda()
            enc_output = encoder(enc_x)  # torch.Size([64, 35, 132])
            y_pre = decoder(dec_x,enc_output)
            pred_gpu = torch.argmax(y_pre, dim=1)#torch.Size([64, 132])
            enc_output = enc_output.cpu().numpy()
            enc_output_list.extend(enc_output)
            y_pre_list.extend(y_pre.cpu().numpy())
            pred = pred_gpu.cpu().numpy()
            pred_list.extend(pred)
            test_loss = criterion(y_pre, dec_y)
            test_losses.append(test_loss.item())
            # print('pred',pred.shape)
            # print(pred.shape)#torch.Size([64, 132])
            correct += pred_gpu.eq(dec_y).sum().item()

        print('\nAccuracy:({:.6f})\nval mean loss:{:.4f}\n'.format(
            correct / (len(testdata)*pred_gpu.shape[-1]) , np.mean(test_losses)))
        print("test predict finish!")
    ##############################save encoder output#############################
    # with h5py.File('torch_datasets/ENC_OUTPUT.hdf5', 'w') as f:
    #     dset = f.create_dataset("vec", data=enc_output_list)
    # print("save encoder output finish")
    ##############################save decoder output#############################
    # with h5py.File('torch_datasets/DEC_OUTPUT.hdf5', 'w') as f:
    #     dset1 = f.create_dataset("vec", data=y_pre_list)
    # print("svae decoder output finish")
    ##############################vec to smiles###################################
    smiles_list = []
    for j in pred_list:
        smiles = "".join(smilesvec1._int_to_char[i] for i in j if i != 33)
        smiles_list.append(smiles)
        smiles_ar = np.array(smiles_list)
    print("transform to smiles finish!\n")
    dt = h5py.special_dtype(vlen=str)
    with h5py.File('torch_datasets/output_smiles_list.hdf5', 'w') as f:
        ds = f.create_dataset('smiles', smiles_ar.shape, dtype=dt)
        ds[:] = smiles_ar
    print("save output_smiles_list!\n")
    return enc_output,smiles_list


    # return enc_output,pred


if __name__ == '__main__':
    enc_output,smiles_list = test_predict()
    # print('\nshow smiles:')
    # for i in range(5):
    #     print(smiles_list[i])

from rdkit import Chem
import h5py
with h5py.File('torch_datasets/train_enc_output_100.hdf5', 'r') as f:
   smiles_out = f['vec'][()]
len(smiles_out)
smiles_out[-10:]

smiles_out = ['O=C(Cc1ccncc1)NCCN1CCC(Cc2ccc(F)cc2)CC1',
 'Cc1cc(NN=CC=Cc2ccccc2)nc(NCc2ccc([N+](=O)[O-])cc2)n1',
 'NC1CC(NC(=O)c2ccc([N+](=O)[O-])cc2)C2(CF)C(CCc3ccccc3)C1(CO)C2O',
 'Cc1ccc2nc(NS(=O)(=O)c3ccc(Cl)cc3)c3cccc(Cl)c3c2c1',
 'N=C(N)NN=C1CCc2cc(S(=O)(=O)N3CC(CO)C(CO)C3C(=O)NC3(c4ccccc4)CCC3)ccc21',
 'COc1ccc2ccccc2c1C=NN=C(N)NOC(=O)c1ccco1',
 'CC(C)(C)Cc1c[nH]c(-c2ccoc2C)n1',
 'NC(=O)Cn1cc(CC(=O)c2ccc(Br)cc2)c2ccccc21',
 'O=C(O)C(=O)Cc1nc2ccccc2o1',
 'COc1ccc(-c2c(C(=O)NS(=O)(=O)c3cccc(Cl)c3)nc3ccc(Cl)cn23)nn1',
 'NC(=O)c1cccc(NC(=O)c2ccc(CNC(=O)Cc3c[nH]cn3)cc2)n1',
 'O=C1CCCc2nc(-c3cc(-c4nc5ccc(Cl)cc5o4)ccc3N3CC3)nc21',
 'CN(C)N=Nc1nc(Nc2ccc(Cl)c(Cl)c2)nc2ccsc12',
 'c1cn(C2CCN(CC3CC3)C(O)C2)cn1',
 'CCOC(=O)OC(C)C=CC=CC=CC=CC1OC(C2=CC(C(C)(C)O)OC2=O)CCO1',
 'C#Cc1cc(C(=O)N2CCC(CCC(=O)NCCN3Cc4ccccc4CC3C)CC2)ccn1',
 'CCc1cccc(NC(=O)CN(Cc2ccco2)C(=O)COc2ccc(Cl)cc2)n1',
 'CCc1nn(Cc2ccc(NC(=O)c3ccc(F)cc3)cc2)c(=O)c2nn(-c3ccccc3)c(C)c12',
 'Cc1cc(NS(=O)(=O)c2ccc(NC(=S)NOC(=O)c3cc(-c4ccccc4)[nH]n3)cc2)cs1',
 'CC(Cn1nc(C)cc1C)NC(=O)c1cccs1']
for idx, smiles in enumerate(smiles_out):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        smiles_out[idx] = Chem.MolToSmiles(mol, canonical=True)
    else:
        smiles_out[idx] = "INVALID"
