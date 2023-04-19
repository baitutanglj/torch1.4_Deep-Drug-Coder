# %load_ext autoreload
# %autoreload 2

import numpy as np
import rdkit
from rdkit import Chem

import h5py, ast, pickle
from ddc_pub import ddc_v3 as ddc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from ddc_pub import ddc_v3 as ddc


# Import existing (trained) model
# Ignore any warning(s) about training configuration or non-seriazable keyword arguments
model_name = "models/heteroencoder_model"
model = ddc.DDC(model_name=model_name)

######################################################################
# Input SMILES to auto-encode
# Input SMILES to auto-encode
# with open('datasets/ChEMBL_validation_set', "r") as f:
#     smiles_in = f.read().splitlines()

smiles_in = ['Cc1cccn2c(CN(C)C3CCCc4ccccc43)c(C(=O)N3CCOCC3)nc12',
             'COC(=O)NN=C(c1ccc(O)cc1)C1C(=O)N(C)C(=O)N(C)C1=O',
             'CCc1cc(CC)nc(OCCCn2c3c(c4cc(-c5nc(C)no5)ccc42)CC(F)(F)CC3)n1',
             'Cc1ccc2c(C(=O)Nc3ccccc3)c(SSc3c(C(=O)Nc4ccccc4)c4ccc(C)cc4n3C)n(C)c2c1',
             'Cc1cccc(-c2ccccc2)c1Oc1nc(O)nc(NCc2ccc3occc3c2)n1',
             'Cn1nnnc1SCC(=O)NN=Cc1ccc(Cl)cc1',
             'COc1cccc(NS(=O)(=O)c2ccc(OC)c(OC)c2)c1',
             'COc1ccc(OC)c(S(=O)(=O)n2nc(C)cc2C)c1',
             'NCCCn1cc(C2=C(c3ccncc3)C(=O)NC2=O)c2ccccc21',
             'CN(C)C(=O)N1CCN(C(c2ccc(Cl)cc2)c2cccnc2)CC1']

# MUST convert SMILES to binary mols for the model to accept them (it re-converts them to SMILES internally)
mols_in = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles)) for smiles in smiles_in]
# Encode the binary mols into their latent representations
latent = model.transform(model.vectorize(mols_in))


# with h5py.File('datasets/output_smiles_list.hdf5', 'r') as f:
#    data = f['smiles'][()]
# len(data)
# latent = data[:10]
# data.shape
# Convert back to SMILES
smiles_out = []
for lat in latent:
    smiles, _ = model.predict(lat, temp=0)
    smiles_out.append(smiles)



for idx, smiles in enumerate(smiles_out):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        smiles_out[idx] = Chem.MolToSmiles(mol, canonical=True)
    else:
        smiles_out[idx] = "INVALID"