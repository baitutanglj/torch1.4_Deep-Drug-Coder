import torch
import  torch.nn as nn
from torch.nn import LSTM,BatchNorm1d,Linear, Sequential
from ddc_pub.TimeDistributed import TimeDistributed
from parser import args

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.input_shape = args.input_shape
        self.lstm_dim = args.lstm_dim
        self.bn_momentum = args.bn_momentum
        self.codelayer_dim = args.codelayer_dim
        self.bn = args.bn
        self.noise_std = args.noise_std
        self.encoder = LSTM(input_size=self.input_shape[-1],
                            hidden_size=self.lstm_dim // 2,
                            batch_first=True,
                            bidirectional=True)
        self.BN_enc_1 = BatchNorm1d(num_features=self.lstm_dim,
                                  momentum=self.bn_momentum)
        self.encoder2 = LSTM(input_size=self.lstm_dim,
                             hidden_size=self.lstm_dim // 2,
                             batch_first=True,
                             bidirectional=True)
        self.BN_enc_2 = BatchNorm1d(num_features=self.lstm_dim*4,
                                  momentum=self.bn_momentum)
        self.neck_relu = Sequential(
            Linear(self.lstm_dim * 4, self.codelayer_dim),
            nn.ReLU(inplace=True)
        )
        self.BN_Codelayer = BatchNorm1d(num_features=self.codelayer_dim, momentum=self.bn_momentum)

    # def init_hidden(self, batch_size):
    #     return (torch.zeros(2, batch_size, self.lstm_dim//2).cuda(),
    #             torch.zeros(2, batch_size, self.lstm_dim//2).cuda())

    def forward(self, x):
        # hidden1 = self.init_hidden(args.batch_size)
        x, (state_h_r, state_c_r) = self.encoder(x)
        if self.bn:
            x = x.permute(0,2,1).contiguous()
            x = self.BN_enc_1(x)
            x = x.permute(0, 2, 1).contiguous()
        _, (state_h2_r, state_c2_r) = self.encoder2(x)
        states = torch.cat((state_h_r[0], state_c_r[0], state_h2_r[0], state_c2_r[0],
                                  state_h_r[1], state_c_r[1], state_h2_r[1], state_c2_r[1]), dim=-1)
        if self.bn:
            states = self.BN_enc_2(states)
        neck_outputs = self.neck_relu(states)
        if self.bn:
            neck_outputs = self.BN_Codelayer(neck_outputs)
            neck_outputs = neck_outputs + torch.normal(0, self.noise_std, size=neck_outputs.shape).cuda()
        return neck_outputs

# def linear_block(in_f, out_f,bn,bn_momentum):
#     if bn:
#         return nn.Sequential(
#             Linear(in_f, out_f),
#             BatchNorm1d(out_f, momentum=bn_momentum),
#             nn.ReLU(inplace=True)
#         )
#     else:
#         return nn.Sequential(
#             Linear(in_f, out_f),
#             nn.ReLU(inplace=True)
#         )

class Latent_states(nn.Module):
    def __init__(self,args):
        super(Latent_states, self).__init__()
        self.codelayer_dim = args.codelayer_dim
        self.num_layer = args.num_layer
        self.lstm_dim = args.lstm_dim
        self.bn_momentum = args.bn_momentum
        self.bn = args.bn
        if self.bn:
            self.h_state = nn.Sequential(
                Linear(self.codelayer_dim,self.lstm_dim),
                BatchNorm1d(self.lstm_dim, momentum=self.bn_momentum),
                nn.ReLU()
            )
            self.c_state = nn.Sequential(
                Linear(self.codelayer_dim, self.lstm_dim),
                BatchNorm1d(self.lstm_dim, momentum=self.bn_momentum),
                nn.ReLU()
            )
        else:
            self.h_state = nn.Sequential(Linear(self.codelayer_dim,self.lstm_dim),nn.ReLU())
            self.c_state = nn.Sequential(Linear(self.codelayer_dim, self.lstm_dim), nn.ReLU())
        # self.h_layer_block = linear_block(self.codelayer_dim,self.lstm_dim,self.bn,self.bn_momentum)
        # self.c_layer_block = linear_block(self.codelayer_dim, self.lstm_dim, self.bn,self.bn_momentum)

    def forward(self, latent_input):
        dec_h = self.h_state(latent_input)
        dec_c = self.c_state(latent_input)
        dec_h = torch.unsqueeze(dec_h, dim=0).contiguous()
        dec_c = torch.unsqueeze(dec_c, dim=0).contiguous()
        return dec_h,dec_c

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.num_layer = args.num_layer
        self.dec_input_shape = args.dec_input_shape
        self.lstm_dim = args.lstm_dim
        self.bn = args.bn
        self.bn_momentum = args.bn_momentum
        self.td_dense_dim = args.td_dense_dim
        self.output_dims = args.output_dims
        self.decoder_lstm_1 = LSTM(input_size=self.dec_input_shape[-1],
                                                hidden_size=self.lstm_dim,
                                                batch_first=True)
        self.BN_Decoder_1 = BatchNorm1d(num_features=self.lstm_dim,
                                      momentum=self.bn_momentum)

        # if self.td_dense_dim > 0:
        #     self.Time_Distributed_1 = TimeDistributed(Linear(self.lstm_dim, self.td_dense_dim), True)

        self.decoder_lstm_2 = LSTM(input_size=self.lstm_dim,
                                   hidden_size=self.lstm_dim,
                                   batch_first=True)
        self.BN_Decoder_2 = BatchNorm1d(num_features=self.lstm_dim,
                                        momentum=self.bn_momentum)
        if self.td_dense_dim > 0:
            self.Time_Distributed_2 = TimeDistributed(Linear(self.lstm_dim, self.td_dense_dim), True)
            self.Dense_Decoder_2 = Sequential(
                Linear(self.td_dense_dim, self.output_dims),
                nn.LogSoftmax(dim=-1)
            )
        else:
            self.Dense_Decoder_2 = Sequential(
                Linear(self.lstm_dim, self.output_dims),
                nn.LogSoftmax(dim=-1)
            )

    # def forward(self,dec,latent_list):
    def forward(self,dec,dec_h,dec_c):
        x, (State_h_1, State_c_1) = self.decoder_lstm_1(dec,(dec_h,dec_c))
        if self.bn:
            x = x.permute(0, 2, 1).contiguous()
            x = self.BN_Decoder_1(x)
            x = x.permute(0, 2, 1).contiguous()
        # if self.td_dense_dim > 0:
        #     x = self.Time_Distributed_1(x)
        x, (State_h_2, State_c_2) = self.decoder_lstm_2(x, (dec_h,dec_c))
        if self.bn:
            x = x.permute(0, 2, 1).contiguous()
            x = self.BN_Decoder_2(x)
            x = x.permute(0, 2, 1).contiguous()
        if self.td_dense_dim > 0:
            x = self.Time_Distributed_2(x)
            x = self.Dense_Decoder_2(x)
        else:
            # x = x.view(-1,self.output_dims).contiguous()
            x = self.Dense_Decoder_2(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
    
class Model_mol_all(nn.Module):
    """Full model that constitutes the complete pipeline."""
    # IFF input is not encoded, stack the encoder (mol_to_latent_model)
    def __init__(self,args):
        super(Model_mol_all, self).__init__()
        # self.input_shape = args.input_shape
        # self.lstm_dim = args.lstm_dim
        # self.bn_momentum = args.bn_momentum
        # self.codelayer_dim = args.codelayer_dim
        # self.bn = args.bn
        # self.noise_std = args.noise_std
        # self.num_layer = args.num_layer
        # self.dec_input_shape = args.dec_input_shape
        # self.td_dense_dim = args.td_dense_dim
        # self.output_dims = args.output_dims
        self.encoder = Encoder(args)
        self.latent_states = Latent_states(args)
        self.decoder = Decoder(args)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_output = self.encoder(encoder_inputs)
        h,c = self.latent_states(encoder_output)
        # x = [decoder_inputs] + x
        decoder_output = self.decoder(decoder_inputs,h,c)
        return encoder_output,decoder_output


class Model_NoEncoder_all(nn.Module):
    def __init__(self,args):
        super(Model_NoEncoder_all, self).__init__()
        self.latent_states = Latent_states(args)
        self.decoder = Decoder(args)
    def forward(self, latent_input, decoder_inputs):
        h,c = self.latent_states(latent_input)
        decoder_output = self.decoder(decoder_inputs,h,c)
        return decoder_output
