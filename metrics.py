import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi
import h5py

def count_v(idx,smiles_in_path,smiles_out_path):

    with open('datasets/ChEMBL_validation_set', "r") as f:
        data = f.read().splitlines()[idx:]
    with open('torch_datasets_ddc4/smile_sample_010one1000', "r") as f:
        gens = f.read().splitlines()
    # with h5py.File('torch_datasets2/char_gens_smiles_060.hdf5', "r") as f:
    #     gens = f["smiles"][:]

    len(gens)
    # len(data)
    o_keys = []
    keys = []
    validy_smiles = []
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

            v += 1

            if g_key in o_keys:
                r += 1


            gens[i] = Chem.MolToSmiles(g_mol, canonical=True)
            validy_smiles.append(gens[i])

        except:
            continue

    '''save vaildy smiles'''
    validy_smiles = set(validy_smiles)
    for smiles in validy_smiles:
        with open(smiles_out_path, "a") as f:
            f.write(smiles)
            f.write('\n')
    print('save smiles finish!')

    validy = v * 1.00 / len(gens)
    recovery = r * 1.00 / len(gens)
    unique = len(list(set(keys))) / v

    print('validy',validy)
    print('recovery',recovery)
    print('unique',unique)
    print('r count',r)
    print('unique count',len(list(set(keys))))

    return validy,recovery,unique

if __name__ == '__main__':
    count_v(idx=-1,
            smiles_in_path='torch_datasets_ddc4/smile_sample_010one1000',
            smiles_out_path='torch_datasets_ddc4/smile_sample_010one1000_vaildy'
           )

# with open('torch_datasets_ddc4/smile_sample_010one1000_vaildy', "r") as f:
#     data = f.read().splitlines()[:]
# len(data)