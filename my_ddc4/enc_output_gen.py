import pandas as pd
import numpy as np
import torch
import h5py
from my_ddc4.model import AutoEncoder
from my_ddc4.trainer import Trainer
from my_ddc4.utils import CharVocab
from my_ddc4.config import args


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
            df = pd.DataFrame(data=enc_output_list[i].numpy())
            df.to_csv(gen_path, mode='a', index=False, header=False)

    # with h5py.File(gen_path, 'w') as f:
    #     dset = f.create_dataset("vec", data=torch.cat(enc_output_list, dim=0).numpy())
    print("save test encoder output finish")

if __name__ == '__main__':
    Enc_output_gen(model_path='../torch_models_ddc4/_train_010.pt',
                   gen_path='../torch_datasets_ddc4/test_enc_output_gen_010one')


# #
# import h5py
# with h5py.File('torch_datasets_ddc4/test_enc_output_gen_010tenten.hdf5', "r") as f:
#     data = f["vec"][:]
# data[:]
# len(data)#92248
# import pandas as pd
# import numpy as np
# data = pd.read_csv('torch_datasets_ddc4/test_enc_output_gen_010_130000', header=None)
# data = np.array(data[:],dtype=np.float32)
# len(data)#92248


