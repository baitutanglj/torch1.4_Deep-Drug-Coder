import numpy as np
import torch
import h5py
from my_ddc3.model import AutoEncoder
from my_ddc3.trainer import Trainer
from my_ddc3.utils import CharVocab
from my_ddc3.config import args


device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def Enc_output_gen():
    with open('../datasets/ChEMBL_training_set', "r") as f:
        vocabulary_data = f.read().splitlines()[:]

    with open('../datasets/ChEMBL_validation_set', "r") as f:
        test_data = f.read().splitlines()[130000:]

    '''load model'''
    vocabulary = CharVocab.from_data(vocabulary_data)
    model = AutoEncoder(vocabulary, args).to(device)
    model.load_state_dict(torch.load('../torch_models/_train_060.pt'))
    model = model.eval()

    '''load test data'''
    trainer = Trainer(args)
    testloader = trainer.get_dataloader(model,test_data,shuffle=False)
    print(len(testloader))

    with torch.no_grad():
        gens_sm = []
        enc_output_list = []

        for i, (encoder_inputs,
                decoder_inputs,
                decoder_targets) in enumerate(testloader):
            encoder_inputs = (data.to(device)
                              for data in encoder_inputs)
            decoder_inputs = (data.to(device)
                              for data in decoder_inputs)
            decoder_targets = (data.to(device)
                               for data in decoder_targets)
            latent_codes_batch = model.encoder_forward(*encoder_inputs)
            enc_output_list.append(latent_codes_batch.cpu().detach())


    enc_output_name = '../torch_datasets/test_enc_output_gen.hdf5'
    with h5py.File(enc_output_name, 'w') as f:
        dset = f.create_dataset("vec", data=torch.cat(enc_output_list, dim=0).numpy())
    print("save test encoder output finish")

if __name__ == '__main__':
    Enc_output_gen()


#
import h5py
with h5py.File('torch_datasets/test_enc_output_gen.hdf5', "r") as f:
    data = f["vec"][:]
data[:]
#len(data)#92248



