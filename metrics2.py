import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi
import h5py




with open('datasets/ChEMBL_validation_set', "r") as f:
    data = f.read().splitlines()[-1:]
with open('torch_datasets_ddc4/point_smiles_sample010', "r") as f:
    gens = f.read().splitlines()
# with h5py.File('torch_datasets2/char_gens_smiles_060.hdf5', "r") as f:
#     gens = f["smiles"][:]

len(gens)
# len(data)
o_keys = []
keys = []
v = 0
r = 0

# with open('torch_datasets/test_keys', "r") as f:
#     o_keys = f.read().splitlines()[:]
# len(o_keys)


for i in range(len(data)):
    o_mol = Chem.MolFromSmiles(data[i])
    o_key = inchi.MolToInchiKey(o_mol)
    o_keys.append(o_key)
for i in range(len(gens)):
    try:
        g_mol = Chem.MolFromSmiles(gens[i])
        g_key = inchi.MolToInchiKey(g_mol)
        keys.append(g_key)

        v = v + 1

        if g_key in o_keys:
            r += 1
    except:
        continue

vaildy = v * 1.00 / len(gens)
recovery = r * 1.00 / len(gens)
unique = len(list(set(keys))) / v

print(vaildy)
print(recovery)
print(unique)
print(r)
########################
import h5py
from rdkit import Chem
# with h5py.File('torch_datasets/AutoEncoder_gens_smiles_000.hdf5', "r") as f:
#     gens = f["smiles"][:]
with open('torch_datasets_ddc4/point_smiles_sample010', "r") as f:
    gens = f.read().splitlines()
vaildy_count = 0
vaildy_smiles = []
for idx, smiles in enumerate(gens):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        gens[idx] = Chem.MolToSmiles(mol, canonical=True)
        vaildy_count += 1
        print(idx)
        with open(smiles_out_path, "a") as f:
            f.write(gens[idx])
            f.write('\n')
        print('save smiles finish!')
print(vaildy_count)