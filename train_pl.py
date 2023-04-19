import h5py
import torch
from vectorizers import SmilesVectorizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'


# Load dataset
dataset_filename = "datasets/DRD2_TEST_MOLS.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols = f["mols"][:]
maxlen = 128
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
batch_size=256
input_type = "mols"

smilesvec1 = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=maxlen,
                charset=charset,
                binary=True,
            )
#smilesvec1.fit(binmols,extra_chars=["\\"])
smilesvec2 = SmilesVectorizer(
    canonical=False,
    augment=True,
    maxlength=maxlen,
    charset=charset,
    binary=True,
    leftpad=False,
)
enc_input = smilesvec1.transform(binmols)#(17817, 133, 35)
dec_in_out = smilesvec2.transform(binmols)#(17817, 133, 35)
dec_input = dec_in_out[:, 0:-1, :]  # Including start_char#(17817, 132, 35)
dec_output = dec_in_out[:, 1:, :]  # No start_char#(17817, 132, 35)
enc_input = torch.FloatTensor(enc_input)
dec_input = torch.FloatTensor(dec_input)
dec_output = torch.FloatTensor(dec_output)
#dec_output = torch.argmax(dec_output, dim=-1)
#dec_output.shape#torch.Size([17817, 132, 35])
# data = TorchGen(enc_input,dec_input,dec_output)
# train_size = int(0.9 * len(data))
# val_size = len(data) - train_size
# train_data, val_data = random_split(data, [train_size, val_size])
# len(dec_input)
# trainloader = DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=8)
# valloader = DataLoader(dataset=val_data,batch_size=32,shuffle=True,num_workers=8)
############################


lstm_dim = 128
bn_momentum = 0.9
bn = True
noise_std = 0.01
codelayer_dim = 128
dec_layers = 2
td_dense_dim = 0
input_shape = smilesvec1.dims
dec_dims = list(smilesvec1.dims)
dec_dims[0] = dec_dims[0] - 1
dec_input_shape = dec_dims
output_len = smilesvec1.dims[0] - 1
output_dims = smilesvec1.dims[-1]
lr = 0.001
epochs = 10


from pytorch_lightning import Trainer
from ddc_pub.torch_model_pl import Build_model_mols

if __name__ == '__main__':
    model2 = Build_model_mols(input_shape, lstm_dim, bn_momentum, bn, noise_std, codelayer_dim,
                              dec_layers, dec_input_shape, td_dense_dim, output_dims,
                              enc_input, dec_input, dec_output)
    trainer = Trainer(max_epochs=10, gpus=2)
    trainer.fit(model2)
