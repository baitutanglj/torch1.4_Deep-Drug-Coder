import torch
import  torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,random_split
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ddc_pub.custom_callbacks import LearningRateSchedule
from data_gen import Data_generate
from parser import args
from torch_model3 import Encoder,Latent_states,Decoder,Model_mol_all

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
##########################load dataset######################################
data = Data_generate("datasets/DRD2_TRAIN_MOLS.h5")
print('generate dataset ok!')
train_size = int(0.9 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])
trainloader = DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=True,num_workers=8,drop_last=True)
valloader = DataLoader(dataset=val_data,batch_size=args.batch_size,shuffle=True,num_workers=8)
#########################weight init############################################
 # 1. 根据网络层的不同定义不同的初始化方式
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # elif isinstance(m, nn.LSTM):
    #     nn.init.xavier_normal_(m.weight.data)
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
#########################train model########################################
def train():
    encoder = Encoder(args).cuda()#return neck_outputs
    latent_states = Latent_states(args).cuda()#return dec_h,dec_c
    decoder = Decoder(args).cuda()#return x
    encoder_optimizer = Adam(encoder.parameters(),lr=args.lr)
    latent_optimizer = Adam(latent_states.parameters(),lr=args.lr)
    decoder_optimizer = Adam(decoder.parameters(),lr=args.lr)
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_losses = []
        encoder.train()
        latent_states.train()
        decoder.train()

        for batch_idx, (enc_x, dec_x, dec_y) in enumerate(trainloader):
            enc_x,dec_x,dec_y = enc_x.cuda(),dec_x.cuda(),dec_y.cuda()
            #print('dec_y',dec_y.shape)#[64, 132]
            encoder_optimizer.zero_grad()
            latent_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            enc_output = encoder(enc_x)#[64, 35, 132]
            latent_h,latent_c = latent_states(enc_output)
            y_pre = decoder(dec_x,latent_h,latent_c)
            loss = criterion(y_pre, dec_y)
            train_losses.append(loss.item())
            # Backward and optimizer
            loss.backward()
            ## 按值裁剪
            ### 指定clip_value之后，裁剪的范围就是[-clip_value, clip_value]
            #clip_grad_value_(model.parameters(), clip_value = args.clip_value)
            decoder_optimizer.step()
            latent_optimizer.step()
            encoder_optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(trainloader),
                    100. * batch_idx / len(trainloader), loss.item()
                ))

        ####validate####
        encoder.eval()
        latent_states.eval()
        decoder.eval()
        val_total_loss = 0
        correct = 0
        with torch.no_grad():
            val_losses = []
            for enc_input, dec_input, dec_output in valloader:
                enc_x,dec_x,dec_y = enc_input.cuda(), dec_input.cuda(), dec_output.cuda()
                enc_output = encoder(enc_x)  # [64, 35, 132]
                latent_h, latent_c = latent_states(enc_output)
                y_pre = decoder(dec_x, latent_h, latent_c)
                #print('y_pre_v:',y_pre.shape)#torch.Size([64, 35, 132])
                val_loss = criterion(y_pre, dec_y)
                val_losses.append(val_loss.item())

                pred = torch.argmax(y_pre, dim=1)
                #print('pred',pred.shape)
                #print(pred.shape)#torch.Size([64, 132])
                correct += pred.eq(dec_y).sum().item()
        with open("datasets/mean_train_loss", "a") as f:
            f.write(str(np.mean(train_losses)))
            f.write('\n')
        with open("datasets/mean_val_loss", "a") as f:
            f.write(str(np.mean(val_losses)))
            f.write('\n')
        print('\ntrain mean loss:{:.6f}\nval mean loss:{:.4f}\n'.format(np.mean(train_losses),np.mean(val_losses)))
        print('val accuracy:({:.6f})\n'.format(correct / (val_size*pred.shape[-1])))

        # scheduler.step(val_loss)


    torch.save(encoder.state_dict(), 'torch_models/encoder_params.pkl')
    torch.save(latent_states.state_dict(), 'torch_models/latent_params.pkl')
    torch.save(decoder.state_dict(), 'torch_models/decoder_params.pkl')
    # 保存和加载整个模型
    torch.save(encoder, 'torch_models/encoder.pkl')
    torch.save(latent_states, 'torch_models/latent_states.pkl')
    torch.save(decoder, 'torch_models/decoder.pkl')
    print('save model finish!')
    #model = torch.load('model.pkl')

if __name__ == '__main__':
    train()

