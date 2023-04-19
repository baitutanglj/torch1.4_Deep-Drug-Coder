#import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import  torch.nn as nn
from torch.nn import LSTM,BatchNorm1d,Linear, Sequential
from torch.autograd import Variable #
# from torch.optim import optimizer,Adam
# from torch.nn.utils import clip_grad_value_
# from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
# from torch.nn.parallel import DistributedDataParallel
# import torch.nn.functional as F
# from torchsummary import summary
# from collections import OrderedDict
# import pytorch_lightning as pl
torch.cuda.is_available()
from ddc_pub.TimeDistributed import TimeDistributed
#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class Mol_to_latent_model(nn.Module):
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
        self.BN_Codelayer = BatchNorm1d(num_features=self.codelayer_dim, momentum=self.bn_momentum)

    # def init_hidden(self, batch_size):
    #     return (torch.zeros(self.num_layers, batch_size, self.lstm_dim).cuda(),
    #             torch.zeros(self.num_layers, batch_size, self.lstm_dim).cuda())

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
            #neck_outputs = neck_outputs + torch.normal(0, self.noise_std, size=neck_outputs.shape).to(device)
        return neck_outputs


# input_shape = (138, 35)
# lstm_dim=128
# bn_momentum=0.9
# codelayer_dim=128
# bn = True
# noise_std = 0.01
# x = torch.randn(200, 138, 35).cuda(1)
# y = torch.randn(200, 138, 35).cuda(1)
# model= Mol_to_latent_model(input_shape,lstm_dim,bn_momentum,codelayer_dim,bn,noise_std).cuda(1)
# a = model(x)
# #summary(model,input_size=input_shape[1],batch_size=200)
#
# def mol_to():
#     # model = Mol_to_latent_model(self.input_shape,self.lstm_dim, self.bn_momentum,
#     #                             self.codelayer_dim, self.bn, self.noise_std).cuda(1)
#     model = Mol_to_latent_model(input_shape=(138, 35),lstm_dim=128, bn_momentum=0.9,
#                                 codelayer_dim=128, bn=True, noise_std=0.01).cuda(1)
#     #output = model(input)
#     return model
# x = torch.randn(200, 138, 35).cuda(1)
# b = mol_to()
# c = b(x)
#
#
#
#
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

class Latent_to_states_model(nn.Module):
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

# input_shape = 128
# codelayer_dim = 128
# lstm_dim = 128
# dec_layers = 2
# bn = True
# bn_momentum = 0.9
# m = Latent_to_states_model(128,2,128,0.9,True)
# print(m)
# x = torch.randn(5,128)
# x.shape
# a = m(x)#list
# import numpy as np
# np.array(a).shape#(4,)
# b = np.array(a)
# type(b[1])#torch.Tensor
# c = b[1].detach().numpy()
# c.shape#(5,128)

# ####################################################################################
class Batch_model(nn.Module):
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
        x, (State_h_1, State_c_1) = self.decoder_lstm_1(x, (decoder_state_list[0],decoder_state_list[1]))
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
#
input_shape = (138,35)
dec_dims = list(input_shape)#[138, 35]
dec_dims[0] = dec_dims[0] - 1
dec_input_shape = dec_dims#[137, 35]
output_len = input_shape[0] - 1#137
output_dims = input_shape[-1]#35
dec_layers = 2
lstm_dim = 128
td_dense_dim = 0
bn = True
bn_momentum = 0.9
x = torch.randn(5,138,35)
h1 = torch.randn(1,5,128)
c1 = torch.randn(1,5,128)
h2 = torch.randn(1,5,128)
c2= torch.randn(1,5,128)
statehc = []
statehc.append(h1)
statehc.append(c1)
statehc.append(h2)
statehc.append(c2)
a = Batch_model(dec_layers,dec_input_shape,lstm_dim,bn,bn_momentum,td_dense_dim,output_dims)
print(a)
b = a(x,statehc)
b.size()#torch.Size([5, 138, 35])
b[0]#xiangliang
import numpy as np
type(b[0][0])
np.array(b).shape
a.state_dict()
c = a.named_parameters()

# ##########################################################################################
class Build_model_mols(nn.Module):
    def __init__(self,input_shape,lstm_dim,bn_momentum,bn,noise_std,
                 codelayer_dim,dec_layers,dec_input_shape,td_dense_dim,output_dims):
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

    def forward(self,encoder_inputs,decoder_inputs):
        x = self.mol_to_latent_model(encoder_inputs)
        latent_state_list = self.latent_to_states_model(x)
        #x = [decoder_inputs] + x
        x = self.batch_model(decoder_inputs,latent_state_list)
        return x

class Build_model_not_mols(nn.Module):
    def __init__(self,input_shape,lstm_dim,bn_momentum,bn,noise_std,
                 codelayer_dim,dec_layers,dec_input_shape,td_dense_dim,output_dims):
        super(Build_model_not_mols,self).__init__()
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

    def forward(self,latent_input,decoder_inputs):
        latent_state_list = self.latent_to_states_model(latent_input)
        #x = [decoder_inputs] + latent_state_list
        x = self.batch_model(decoder_inputs,latent_state_list)
        return x


# input_shape = (138,35)
# dec_dims = list(input_shape)#[138, 35]
# dec_dims[0] = dec_dims[0] - 1
# dec_input_shape = dec_dims#[137, 35]
# output_len = input_shape[0] - 1#137
# output_dims = input_shape[-1]#35
# dec_layers = 2
# lstm_dim = 128
# td_dense_dim = 2
# bn = True
# bn_momentum = 0.9
# noise_std = 0.01
# codelayer_dim = 128
# x = torch.randn(5,138,35)
# dx = torch.randn(5,138,35)
# model_all = Build_model_mols(input_shape,lstm_dim,bn_momentum,bn,noise_std,codelayer_dim,dec_layers,dec_input_shape,td_dense_dim,output_dims)
# print(model_all)
# a = model_all(x,dx)
# import numpy as np
# np.array(a).shape
# a[0][0][0]


#####################################################################












