import torch
import  torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,random_split
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ddc_pub.custom_callbacks import LearningRateSchedule
from data_gen import Data_generate
from parser import args
from torch_model3 import Model_mol_all
# from torch_modelaae import Model_mol_all
import numpy as np
import h5py
from vectorizers import SmilesVectorizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
##########################load dataset######################################
data = Data_generate("datasets/DRD2_TRAIN_MOLS.h5")
print('generate dataset ok!')
train_size = int(0.9 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])
trainloader = DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=False,num_workers=8,drop_last=True)
valloader = DataLoader(dataset=val_data,batch_size=args.batch_size,shuffle=False,num_workers=8)
len(trainloader)
len(valloader)

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
#################################adjust_lr#####################################
# model = Model_mol_all(args)
# #将weight_init应用在子模块上
# model.apply(weight_init)
# model.cuda()
# optimizer = Adam(model.parameters(), lr=args.lr)
# def adjust_lr(optimizer, epoch, epoch_to_start=500, last_epoch=999,
#               lr_init=1e-3, lr_final=1e-6,lr=args.lr):
#     decay_duration = last_epoch - epoch_to_start
#     if epoch < epoch_to_start:
#         cur_lr = lr
#     else:
#         # Slope of the decay
#         k = -(1 / decay_duration) * np.log(lr_final / lr_init)
#
#         ad_lr = lr_init * np.exp(-k * (epoch - epoch_to_start))
#         cur_lr = ad_lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = cur_lr
# for epoch in args.epochs:
#     adjust_lr(optimizer, epoch)
#########################train model########################################
def train():
    smilesvec1 = SmilesVectorizer(
        canonical=False,
        augment=True,
        maxlength=args.maxlen,
        charset=args.charset,
        binary=False,
    )
    model = Model_mol_all(args)
    #将weight_init应用在子模块上
    model.apply(weight_init)
    model.cuda()
    # model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience,
                      verbose=True,min_lr=1e-6)
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        train_losses = []
        model.train()
        pred_list = []

        for batch_idx, (enc_x, dec_x, dec_y) in enumerate(trainloader):
            train_correct = 0

            enc_x,dec_x,dec_y = enc_x.cuda(),dec_x.cuda(),dec_y.cuda()
            #print('dec_y',dec_y.shape)#[64, 132]

            optimizer.zero_grad()
            enc_output,y_pre = model(enc_x, dec_x)#[64, 35, 132]
            print(enc_output)
            print(enc_output.shape)
            #print('y_pre',y_pre.shape)
            loss = criterion(y_pre, dec_y)
            train_losses.append(loss.item())
            # Backward and optimizer
            loss.backward()

            pred_gpu = torch.argmax(y_pre, dim=1)
            pred = pred_gpu.cpu().numpy()
            pred_list.extend(pred)

            train_correct += pred_gpu.eq(dec_y).sum().item()
            print('pred_gpu',pred_gpu[0])
            print('dec_y',dec_y[0])

            ## 按值裁剪
            ### 指定clip_value之后，裁剪的范围就是[-clip_value, clip_value]
            # clip_grad_value_(model.parameters(), clip_value = args.clip_value)
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain_accuracy: {:.6f}'.format(
                    epoch, batch_idx, len(trainloader),
                    100. * batch_idx / len(trainloader), loss.item(),
                    train_correct / (args.batch_size*pred.shape[-1])
                ))

        ############################save train smiles#####################################
        if epoch == 9:
            smiles_list = []
            for j in pred_list:
                smiles = "".join(smilesvec1._int_to_char[i] for i in j if i != 33)
                smiles_list.append(smiles)
                smiles_ar = np.array(smiles_list)
            print("transform to smiles finish!\n")
            dt = h5py.special_dtype(vlen=str)
            with h5py.File('torch_datasets/train_smiles_list.hdf5', 'w') as f:
                ds = f.create_dataset('smiles', smiles_ar.shape, dtype=dt)
                ds[:] = smiles_ar
            print("save train_smiles_list!\n")

        ####validate####
        model.eval()
        val_total_loss = 0
        correct = 0
        pred_list_v = []
        with torch.no_grad():
            val_losses = []
            for enc_input_v, dec_input_v, dec_output_v in valloader:
                enc_x_v,dec_x_v,dec_y_v = enc_input_v.cuda(), dec_input_v.cuda(), dec_output_v.cuda()
                #print(enc_x.shape)#torch.Size([64, 133, 35])
                enc_output,y_pre_v = model(enc_x_v, dec_x_v)#torch.Size([64, 35, 132])
                #print('y_pre_v:',y_pre.shape)#torch.Size([64, 35, 132])
                val_loss = criterion(y_pre_v, dec_y_v)
                val_losses.append(val_loss.item())

                pred_gpu = torch.argmax(y_pre_v, dim=1)
                #print('pred',pred.shape)
                #print(pred.shape)#torch.Size([64, 132])
                pred = pred_gpu.cpu().numpy()
                pred_list_v.extend(pred)

                correct += pred_gpu.eq(dec_y_v).sum().item()

        with open("datasets/mean_train_loss", "a") as f:
            f.write(str(np.mean(train_losses)))
            f.write('\n')
        with open("datasets/mean_val_loss", "a") as f:
            f.write(str(np.mean(val_losses)))
            f.write('\n')
        print('\ntrain mean loss:{:.6f}\nval mean loss:{:.4f}\n'.format(np.mean(train_losses),np.mean(val_losses)))
        print('val accuracy:({:.6f})\n'.format(correct / (val_size*pred.shape[-1])))

        scheduler.step(val_loss)
        ######################save val smiles#############################
        if epoch == 30:
            smiles_list_v = []
            for j in pred_list_v:
                smiles = "".join(smilesvec1._int_to_char[i] for i in j if i != 33)
                smiles_list_v.append(smiles)
                smiles_ar = np.array(smiles_list_v)
            print("transform to smiles finish!\n")
            dt = h5py.special_dtype(vlen=str)
            with h5py.File('torch_datasets/val_smiles_list.hdf5', 'w') as f:
                ds = f.create_dataset('smiles', smiles_ar.shape, dtype=dt)
                ds[:] = smiles_ar
            print("save val_smiles_list!\n")


    torch.save(model.state_dict(), 'torch_models/model_params.pkl')
    # 保存和加载整个模型
    torch.save(model, 'torch_models/model.pkl')
    print('save model finish!')
    #model = torch.load('model.pkl')

if __name__ == '__main__':
    train()

