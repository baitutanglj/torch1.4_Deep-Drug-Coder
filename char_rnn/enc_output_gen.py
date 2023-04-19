import numpy as np
import pandas as pd
import torch
import h5py
from char_rnn.model import AutoEncoder
from char_rnn.trainer import Trainer
from char_rnn.utils import CharVocab
from char_rnn.config import args


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Enc_output_gen(model_path,gen_path):
    with open('../datasets/ChEMBL_training_set', "r") as f:
        vocabulary_data = f.read().splitlines()[:]

    with open('../datasets/ChEMBL_validation_set', "r") as f:
        test_data = f.read().splitlines()[-1:]

    '''load model'''
    vocabulary = CharVocab.from_data(vocabulary_data)
    model = AutoEncoder(vocabulary, args).to(device)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    '''load test data'''
    trainer = Trainer(args)
    testloader = trainer.get_dataloader(model,test_data,shuffle=False)


    with torch.no_grad():
        gens_sm = []
        enc_output_list = []
        dec_output_list = []
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

            decoder_outputs, _, _  = model.decoder_forward(*decoder_inputs,
                                                           latent_codes_batch,
                                                           i==0)

            dec_output_list.append(decoder_outputs.cpu().detach())
            print('dec finish!')

    enc_output_name = gen_path
    with h5py.File(enc_output_name, 'w') as f:
        dset = f.create_dataset("vec", data=torch.cat(enc_output_list, dim=0).numpy())

        # enc_output_name = '../torch_datasets2/test_enc_output_gen_ten20'
        # df = pd.DataFrame(data=enc_output_list[0].numpy())
        # df.to_csv(enc_output_name, index=False, header=False)

        # dec_output_name = '../torch_datasets2/test_dec_output_gen_ten'
        # df = pd.DataFrame(data=dec_output_list[0][11].numpy())
        # df.to_csv(dec_output_name, index=False, header=False)

    print("save test encoder output finish")

if __name__ == '__main__':
    Enc_output_gen(model_path='../torch_models2/charrnn_020.pt',
                   gen_path='../torch_datasets2/test_enc_output_gen_20one.hdf5')


#
# import h5py
# with h5py.File('torch_datasets2/test_enc_output_gen_20one.hdf5', "r") as f:
#     data = f["vec"][:]
# data[:]
# len(data)#92248



