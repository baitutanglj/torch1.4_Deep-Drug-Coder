import h5py
from rdkit import Chem
def Validy(smiles_in_path,smiles_out_path):
    # with h5py.File('torch_datasets/AutoEncoder_gens_smiles_000.hdf5', "r") as f:
    #     gens = f["smiles"][:]
    with open(smiles_in_path, "r") as f:
        gens = f.read().splitlines()
    validy_count = 0
    validy_smiles = []
    for idx, smiles in enumerate(gens):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            gens[idx] = Chem.MolToSmiles(mol, canonical=True)
            validy_count += 1
            print(idx)
            with open(smiles_out_path, "a") as f:
                f.write(gens[idx])
                f.write('\n')
            print('save smiles finish!')
    v = validy_count / len(gens)
    print('validy_count',validy_count)
    print('validy_count / len(gens)',v)

    return validy_count,v

if __name__ == '__main__':
    Validy(smiles_in_path='torch_datasets_ddc4/point_smiles_sample010',
           smiles_out_path='torch_datasets_ddc4/point_smiles_sample010_vaildy'
           )

