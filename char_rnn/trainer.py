import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from char_rnn.interfaces import MosesTrainer
from char_rnn.utils import CharVocab, Logger


class Trainer(MosesTrainer):

    def __init__(self, config):
        self.config = config

    def train_epoch(self, model, tqdm_data, criterion, optimizer=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        latent_codes_list = []
        gens_sm = []
        postfix = {'loss': 0,
                   'running_loss': 0}

        for i, (encoder_inputs,
                decoder_inputs,
                decoder_targets) in enumerate(tqdm_data):
            encoder_inputs = (data.to(model.device)
                              for data in encoder_inputs)
            decoder_inputs = (data.to(model.device)
                              for data in decoder_inputs)
            decoder_targets = (data.to(model.device)
                               for data in decoder_targets)

            latent_codes = model.encoder_forward(*encoder_inputs)
            decoder_outputs, decoder_output_lengths, _ = model.decoder_forward(
                *decoder_inputs, latent_codes, is_latent_states=True)

            '''transform to smlies '''
            # logits = torch.softmax(decoder_outputs, 2)
            # currents = torch.distributions.Categorical(logits).sample()
            # sm = [model.tensor2string(i).split('<eos>')[0] for i in currents]
            # gens_sm.extend(sm)
            #####################################
            logits = torch.softmax(decoder_outputs, 2)
            currents = torch.argmax(logits, dim=-1)
            sm = [model.tensor2string(i).split('<eos>')[0] for i in currents]
            gens_sm.extend(sm)


            decoder_outputs = torch.cat(
                [t[:l] for t, l in zip(decoder_outputs,
                                       decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat(
                [t[:l] for t, l in zip(*decoder_targets)], dim=0)

            loss = criterion(decoder_outputs, decoder_targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                for parameter in model.parameters():
                    parameter.grad.clamp_(-5, 5)
                optimizer.step()
            else:
                ######save val enc_output######
                latent_codes_list.append(latent_codes.cpu().detach())

            postfix['loss'] = loss.item()
            postfix['running_loss'] += (loss.item() -
                                        postfix['running_loss']) / (i + 1)
            tqdm_data.set_postfix(postfix)

        postfix['mode'] = 'Eval' if optimizer is None else 'Train'
        if optimizer is not None:
            return postfix,latent_codes
        else:
            return postfix, torch.cat(latent_codes_list, dim=0),np.array(gens_sm)

    def train(self, model, train_loader, val_loader=None, logger=None):
        def get_params():
            return (p for p in model.parameters() if p.requires_grad)

        device = model.device
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(get_params(), lr=self.config.lr)
        optimizer = torch.optim.Adam(
                list(model.encoder.parameters()) +
                list(model.decoder.parameters()), lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              self.config.step_size,
                                              self.config.gamma)

        model.zero_grad()
        for epoch in range(self.config.train_epochs):
            # scheduler.step()

            tqdm_data = tqdm(train_loader,
                             desc='Training (epoch #{})'.format(epoch))
            postfix,latent_codes = self.train_epoch(model, tqdm_data, criterion, optimizer)
            if logger is not None:
                logger.append(postfix)
                logger.save(self.config.log_file)

            if val_loader is not None:
                tqdm_data = tqdm(val_loader,
                                 desc='Validation (epoch #{})'.format(epoch))
                postfix,latent_codes_val,gens_sm_val = self.train_epoch(model, tqdm_data, criterion)
                if logger is not None:
                    logger.append(postfix)
                    logger.save(self.config.log_file)
            scheduler.step()

            if epoch % self.config.save_frequency == 0:
                model = model.to('cpu')
                torch.save(model.state_dict(),
                           'torch_models2/charrnn_{0:03d}.pt'.format(epoch))
                model = model.to(device)
                print('save model finish!')

                if val_loader is not None:
                    enc_output_name = 'torch_datasets2/char_enc_output_{0:03d}.hdf5'.format(epoch)
                    enc_output = latent_codes_val.numpy()
                    with h5py.File(enc_output_name, 'w') as f:
                        dset = f.create_dataset("vec", data=enc_output)
                    print("save encoder output finish")

                    gens_sm_name = 'torch_datasets2/char_gens_smiles_{0:03d}.hdf5'.format(epoch)
                    dt = h5py.special_dtype(vlen=str)
                    with h5py.File(gens_sm_name, 'w') as f:
                        ds = f.create_dataset("smiles", gens_sm_val.shape, dtype=dt)
                        ds[:] = gens_sm_val
                    print("save generate smiles finish")

            scheduler.step()


    def get_vocabulary(self, data):
        return CharVocab.from_data(data)

    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            data.sort(key=lambda x: len(x), reverse=True)

            tensors = [model.string2tensor(string, device=device)
                       for string in data]
            lengths = torch.tensor([len(t) for t in tensors],
                                   dtype=torch.long,
                                   device=device)

            encoder_inputs = pad_sequence(tensors,
                                          batch_first=True,
                                          padding_value=model.vocabulary.pad)
            encoder_input_lengths = lengths - 2

            decoder_inputs = pad_sequence([t[:-1] for t in tensors],
                                          batch_first=True,
                                          padding_value=model.vocabulary.pad)
            decoder_input_lengths = lengths - 1

            decoder_targets = pad_sequence([t[1:] for t in tensors],
                                           batch_first=True,
                                           padding_value=model.vocabulary.pad)
            decoder_target_lengths = lengths - 1

            return (encoder_inputs, encoder_input_lengths), \
                   (decoder_inputs, decoder_input_lengths), \
                   (decoder_targets, decoder_target_lengths)

        return collate

    def fit(self, model, train_data, val_data=None):
        logger = Logger() if self.config.log_file is not None else None

        train_loader = self.get_dataloader(model, train_data, shuffle=True)
        val_loader = (None if val_data is None
                      else self.get_dataloader(model, val_data, shuffle=False))

        self.train(model, train_loader, val_loader, logger)
        return model


