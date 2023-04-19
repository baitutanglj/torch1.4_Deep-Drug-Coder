import h5py
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from vectorizers import SmilesVectorizer

# Load dataset
class Data_generate(Dataset):
    def __init__(self,dataset_filename):
        self.dataset_filename = dataset_filename
        with h5py.File(dataset_filename, "r") as f:
            self.data = f["mols"][:]
        # with open(dataset_filename, "r") as f:
        #     self.data = f.read().splitlines()
        # self.data = [Chem.MolFromSmiles(i) for i in self.data[:]]
        maxlen = 133
        charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
        self.smilesvec1 = SmilesVectorizer(
                        canonical=False,
                        augment=True,
                        maxlength=maxlen,
                        charset=charset,
                        binary=True,
                    )
        self.smilesvec2 = SmilesVectorizer(
                        canonical=False,
                        augment=True,
                        maxlength=maxlen,
                        charset=charset,
                        binary=True,
                        leftpad=False,
                    )
        # print("Default Charset %s" % self.smilesvec1.charset)
        # print("Default Maximum allowed SMILES length %s" % self.smilesvec1.maxlength)
        #
        # self.smilesvec1.fit(self.data, extra_chars=["\\"])
        # print()
        # print("After fitting")
        # print("Charset after fit %s" % self.smilesvec1.charset)
        # print("Maximum allowed SMILES length %s" % self.smilesvec1.maxlength)
        self.enc_input = self.smilesvec1.transform(self.data)#(17817, 133, 35)
        self.dec_in_out = self.smilesvec2.transform(self.data)#(17817, 133, 35)
        self.dec_input = self.dec_in_out[:, 0:-1, :]  # Including start_char#(17817, 132, 35)
        self.dec_output = self.dec_in_out[:, 1:, :]  # No start_char#(17817, 132, 35)
        self.enc_input = torch.FloatTensor(self.enc_input)
        self.dec_input = torch.FloatTensor(self.dec_input)
        self.dec_output = torch.FloatTensor(self.dec_output)
        # self.enc_input = torch.LongTensor(self.enc_input)
        # self.dec_input = torch.LongTensor(self.dec_input)
        # self.dec_output = torch.LongTensor(self.dec_output)

        # self.dec_input =torch.tensor(self.dec_input,dtype = torch.float,requires_grad=True)
        self.dec_output = torch.argmax(self.dec_output, dim=-1)

    def __len__(self):
        return len(self.enc_input)

    def __getitem__(self, idx):
        enc_input_i = self.enc_input[idx]
        dec_input_i = self.dec_input[idx]
        dec_output_i = self.dec_output[idx]
        return enc_input_i, dec_input_i, dec_output_i


