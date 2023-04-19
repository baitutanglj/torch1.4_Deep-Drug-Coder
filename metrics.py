import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi
import h5py


# configp = config.get_config()
########################

# # load original data
# with open("datasets/DRD2_TRAIN_MOLS.h5", "r") as f:
#     data = f.readlines()
#     data = [i.split("\n")[0] for i in data]
# f.close()
#
# # load gen data
# f2 =  h5py.File("torch_datasets/output_smiles_list.hdf5", "r")
# gens = f2["smiles"][:]
# f2.close()


with open('datasets/ChEMBL_validation_set', "r") as f:
    data = f.read().splitlines()[:13000]
# with open('torch_datasets2/smile_sample_charrnn', "r") as f:
#     data = f.read().splitlines()
with open('torch_datasets2/smile_sample_one', "r") as f:
    gens = f.read().splitlines()
# with h5py.File('torch_datasets2/char_gens_smiles_100.hdf5', "r") as f:
#     gens = f["smiles"][:]
# with h5py.File('torch_datasets/test_smile_sample_060_2.hdf5', "r") as f:
#     gens2 = f["smiles"][:]
# gens = np.concatenate((gens , gens2))
len(gens)
len(data)
keys = []
v = 0
r = 0

for i in range(len(gens)):

    o_mol = Chem.MolFromSmiles(data[i])
    print('o_mol',o_mol)
    o_key = inchi.MolToInchiKey(o_mol)
    print('o_key', o_key)
    try:
        g_mol = Chem.MolFromSmiles(gens[i])
        print('g_mol', g_mol)
        g_key = inchi.MolToInchiKey(g_mol)
        print('g_key',g_key)
        keys.append(g_key)
        print('keys',keys)

        v = v + 1
        if g_key == o_key:
            r += 1
    except:
        continue

vaildy = v * 1.00 / len(gens)
recovery = r * 1.00 / len(gens)
unique = len(list(set(keys))) / len(gens)

print(vaildy)
print(recovery)
print(unique)

########################
import h5py
from rdkit import Chem
# with h5py.File('torch_datasets/AutoEncoder_gens_smiles_000.hdf5', "r") as f:
#     gens = f["smiles"][:]
with open('torch_datasets2/smile_sample_one', "r") as f:
    gens = f.read().splitlines()
validy_count = 0
for idx, smiles in enumerate(gens):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        gens[idx] = Chem.MolToSmiles(mol, canonical=True)
        validy_count += 1
    else:
        gens[idx] = "INVALID"
    vaildy = validy_count / len(gens)
print('len(gens):{}, validy_count:{}'.format(len(gens),validy_count))
print('vaildy:',vaildy)
########################
