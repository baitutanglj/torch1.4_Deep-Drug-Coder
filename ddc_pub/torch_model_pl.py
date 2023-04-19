import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import torch
import  torch.nn as nn
from torch.nn import LSTM,BatchNorm1d,Linear, Sequential
from torch.optim import optimizer,Adam
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.nn.functional as F

from torchsummary import summary

import pytorch_lightning as pl
from ddc_pub.TimeDistributed import TimeDistributed
from ddc_pub.torch_gen2 import TorchGen
#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class Mol_to_latent_model(pl.LightningModule):
    def __init__(self,input_shape,lstm_dim,bn_momentum,codelayer_dim,bn,noise_std):
        super(Mol_to_latent_model, self).__init__()
        self.input_shape = input_shape
        self.lstm_dim = lstm_dim
        self.bn_momentum = bn_momentum
        self.codelayer_dim = codelayer_dim
        self.bn = bn
        self.noise_std = noise_std
        self.encoder = LSTM(input_size=self.input_shape[-1],
                            hidden_size=self.lstm_dim // 2,
                            batch_first=True,
                            bidirectional=True)
        self.BN_1 = BatchNorm1d(num_features=self.lstm_dim,
                                  momentum=self.bn_momentum)
        self.encoder2 = LSTM(input_size=self.lstm_dim,
                             hidden_size=self.lstm_dim // 2,
                             batch_first=True,
                             bidirectional=True)
        self.BN_2 = BatchNorm1d(num_features=self.lstm_dim*4,
                                  momentum=self.bn_momentum)
        self.neck_relu = Sequential(
            Linear(self.lstm_dim * 4, self.codelayer_dim),
            nn.ReLU()
        )
        self.BN_Codelayer = BatchNorm1d(num_features=codelayer_dim, momentum=self.bn_momentum)

    def forward(self, x):
        #self.bn = bn
        x, (state_h_r, state_c_r) = self.encoder(x)
        state_h, state_h_reverse = state_h_r.chunk(2, dim=0)
        state_c, state_c_reverse = state_c_r.chunk(2, dim=0)
        if self.bn:
            x = x.permute(0,2,1)
            x = self.BN_1(x)
            x = x.permute(0, 2, 1)
        _, (state_h2_r, state_c2_r) = self.encoder2(x)
        state_h2, state_h2_reverse = state_h2_r.chunk(2, dim=0)
        state_c2, state_c2_reverse = state_c2_r.chunk(2, dim=0)
        states = torch.cat((state_h, state_c, state_h2, state_c2, state_h_reverse,
                            state_c_reverse, state_h2_reverse, state_c2_reverse), dim=-1)
        states = states.squeeze(0)
        if self.bn:
            states = self.BN_2(states)
        neck_outputs = self.neck_relu(states)
        if self.bn:
            neck_outputs = self.BN_Codelayer(neck_outputs)
            #neck_outputs = neck_outputs + torch.normal(0, self.noise_std, size=neck_outputs.shape)
        return neck_outputs
# #######################################################################33
def linear_block(in_f, out_f,bn,bn_momentum):
    if bn:
        return nn.Sequential(
            Linear(in_f, out_f),
            BatchNorm1d(out_f, momentum=bn_momentum),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            Linear(in_f, out_f),
            nn.ReLU()
        )

class Latent_to_states_model(pl.LightningModule):
    def __init__(self,codelayer_dim,dec_layers,lstm_dim,bn_momentum,bn):
        super(Latent_to_states_model, self).__init__()
        self.decoder_state_list = []
        self.codelayer_dim = codelayer_dim
        self.dec_layers = dec_layers
        self.lstm_dim = lstm_dim
        self.bn_momentum = bn_momentum
        self.bn = bn
        self.h_layer_block = linear_block(self.codelayer_dim,self.lstm_dim,self.bn,self.bn_momentum)
        self.c_layer_block = linear_block(self.codelayer_dim, self.lstm_dim, self.bn,self.bn_momentum)


    def forward(self, latent_input):
        for dec_layer in range(self.dec_layers):
            h = self.h_layer_block(latent_input)
            c = self.c_layer_block(latent_input)
            h = torch.unsqueeze(h, dim=0)
            c = torch.unsqueeze(c, dim=0)
            self.decoder_state_list.append(h)
            self.decoder_state_list.append(c)
        return self.decoder_state_list

# ####################################################################################
class Batch_model(pl.LightningModule):
    def __init__(self,dec_layers,dec_input_shape,lstm_dim,bn,bn_momentum,td_dense_dim,output_dims):
        super(Batch_model, self).__init__()
        self.dec_layers = dec_layers
        self.dec_input_shape = dec_input_shape
        self.lstm_dim = lstm_dim
        self.bn = bn
        self.bn_momentum = bn_momentum
        self.td_dense_dim = td_dense_dim
        self.output_dims = output_dims
        self.inputs = []
        self.decoder_lstm_1 = LSTM(input_size=self.dec_input_shape[-1],
                                                hidden_size=self.lstm_dim,
                                                batch_first=True)
        self.BN_Decoder_1 = BatchNorm1d(num_features=self.lstm_dim,
                                      momentum=self.bn_momentum)

        if self.td_dense_dim > 0:
            self.Time_Distributed_1 = TimeDistributed(Linear(self.lstm_dim, self.td_dense_dim), True)
            self.Dense_Decoder_1 = Sequential(
                Linear(self.td_dense_dim, self.output_dims),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.Dense_Decoder_1 = Sequential(
                Linear(self.lstm_dim, self.output_dims),
                nn.LogSoftmax(dim=-1)
            )



    def forward(self,x,decoder_state_list):
        x, (State_h_1, State_c_1) = self.decoder_lstm_1(x,(decoder_state_list[0],decoder_state_list[1]))
        if self.bn:
            x = x.permute(0, 2, 1)
            x = self.BN_Decoder_1(x)
            x = x.permute(0, 2, 1)
        if self.td_dense_dim > 0:
            x = self.Time_Distributed_1(x)
            x = self.Dense_Decoder_1(x)
        else:
            x = self.Dense_Decoder_1(x)
        ############dec_laye2#############

        return x

# ##########################################################################################
class Build_model_mols(pl.LightningModule):
    def __init__(self,input_shape,lstm_dim,bn_momentum,bn,noise_std,
                 codelayer_dim,dec_layers,dec_input_shape,td_dense_dim,output_dims,enc_input,dec_input,dec_output):
        super(Build_model_mols,self).__init__()
        self.input_shape = input_shape
        self.lstm_dim = lstm_dim
        self.bn_momentum = bn_momentum
        self.bn = bn
        self.noise_std = noise_std
        self.codelayer_dim = codelayer_dim
        self.dec_layers = dec_layers
        self.dec_input_shape = dec_input_shape
        self.td_dense_dim = td_dense_dim
        self.output_dims = output_dims
        self.enc_input = enc_input
        self.dec_input = dec_input
        self.dec_output = dec_output


        self.mol_to_latent_model =  Mol_to_latent_model(self.input_shape,
                                              self.lstm_dim,
                                              self.bn_momentum,
                                              self.codelayer_dim,
                                              self.bn,
                                              self.noise_std)
        self.latent_to_states_model = Latent_to_states_model(self.codelayer_dim,
                                                self.dec_layers,
                                                self.lstm_dim,
                                                self.bn_momentum,
                                                self.bn)
        self.batch_model = Batch_model(self.dec_layers,
                                            self.dec_input_shape,
                                            self.lstm_dim,
                                            self.bn,
                                            self.bn_momentum,
                                            self.td_dense_dim,
                                            self.output_dims)
    def mol_to_latent(self):
        self.encoder = LSTM(input_size=self.input_shape[-1],
                            hidden_size=self.lstm_dim // 2,
                            batch_first=True,
                            bidirectional=True)
        self.BN_1 = BatchNorm1d(num_features=self.lstm_dim,
                                  momentum=self.bn_momentum)
        self.encoder2 = LSTM(input_size=self.lstm_dim,
                             hidden_size=self.lstm_dim // 2,
                             batch_first=True,
                             bidirectional=True)
        self.BN_2 = BatchNorm1d(num_features=self.lstm_dim*4,
                                  momentum=self.bn_momentum)
        self.neck_relu = Sequential(
            Linear(self.lstm_dim * 4, self.codelayer_dim),
            nn.ReLU()
        )
        self.BN_Codelayer = BatchNorm1d(num_features=self.codelayer_dim, momentum=self.bn_momentum)

    def forward(self,encoder_input,decoder_input):
        x = self.mol_to_latent_model(encoder_input)
        latent_state_list = self.latent_to_states_model(x)
        #x = [decoder_inputs] + x
        x = self.batch_model(decoder_input,latent_state_list)
        return x
    def training_step(self, batch, batch_idx):
        enc_x,dec_x,dec_y = batch[0],batch[1],batch[2]
        torch.autograd.set_detect_anomaly(True)
        dec_y = torch.argmax(dec_y, dim=-1)
        y_hat = self(enc_x,dec_x)
        y_hat = y_hat.permute(0,2,1)
        loss = F.nll_loss(y_hat,dec_y)
        #loss.backward(retain_graph=True)
        return {'loss': loss}
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    def prepare_data(self):
        #self.traindataset = TorchGen(self.enc_input,self.dec_input,self.dec_output)
        self.traindataset = Data.TensorDataset(self.enc_input,self.dec_input,self.dec_output)
    def train_dataloader(self):
        loader = DataLoader(self.traindataset, batch_size=32, num_workers=4)
        return loader







