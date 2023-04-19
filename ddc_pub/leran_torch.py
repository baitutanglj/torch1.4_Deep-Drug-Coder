import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import torch.nn as nn
from torch.nn import  LSTM
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
torch.cuda.is_available()



x = torch.randn(5, 4, 3)
h0 = torch.randn(2, 4, 2)
c0 = torch.randn(2, 4, 2)
output, (hn, cn)=LSTM(x)
output.shape
output.contiguous().view(-1, 2*2).shape
hn.shape
cn.shape
#定义一个num_layers=3的双向LSTM，h_n第一个维度的大小就等于 6 （2*3），
# h_n[0]表示第一层前向传播最后一个time
'''
lstm = nn.LSTM(input_size=3,hidden_size=1,num_layers=1,bidirectional=False)
output, (hn, cn)=lstm(x)
hn.shape
Out[42]: torch.Size([1, 4, 1])
output.shape
Out[43]: torch.Size([5, 4, 1])
lstm = nn.LSTM(input_size=3,hidden_size=1,num_layers=1,bidirectional=True)
output, (hn, cn)=lstm(x)
output.shape
Out[46]: torch.Size([5, 4, 2])
hn.shape
Out[47]: torch.Size([2, 4, 1])
'''


##############
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#或torch.cuda.set_device(0)
import torch
import torch.nn as nn
from torch.nn import  LSTM
import numpy as np
torch.cuda.is_available()

inputs = np.random.random([32, 10, 8]).astype(np.float32)#10:每句要遍历的单词数
inputs = torch.from_numpy(inputs).cuda()
inputs = inputs.permute(1,0,2)
inputs.shape#[10,32,8]
inputs.get_device()

# lstm = nn.LSTM(input_size=8,hidden_size=4,num_layers=1,bidirectional=False)
# output, (hn, cn)=lstm(inputs)
# output.shape#[10,32,4]
# hn.shape#[1,32,4]
# cn.shape#[1,32,4]
# hn = hn.view(32,4)
# hn.shape

##################################################################################################
input_shape = (138,35)
x = torch.randn(200,138,35)
#x = x.permute(1,0,2)#[138, 200, 35]
lstm_dim = 128
lstm = nn.LSTM(input_size=input_shape[1],hidden_size=lstm_dim // 2,batch_first=True,bidirectional=True)
encoder = nn.LSTM(input_size=input_shape[-1],hidden_size=lstm_dim // 2,num_layers=2,batch_first=True,bidirectional=True)
decoder_lstm_1 = LSTM(input_size=35,hidden_size=128,num_layers=2,batch_first=True)
x_d,(_,_) = decoder_lstm_1(x)
x_d.shape
linear = nn.Linear(128, 35)
a = linear(x_d)
a.shape
x_e, (hn,cn)=lstm(x)
x_e, (hn,cn)=encoder(x)
#x = x.permute(1,0,2)
x_e.shape#[200, 138, 128]
hn.shape#[2, 200, 64]##torch.Size([4, 200, 64])
cn.shape#[2, 200, 64]
hn[0].shape#torch.Size([200, 64])
# h = hn.permute(1, 2, 0).contiguous().view(x.shape[0], -1)
# h.shape#torch.Size([200, 256])
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# x = torch.randn(200,138)
# embedding_layer = nn.Embedding(200,35)
# x = x.long()
# x = abs(torch.LongTensor(x))
# x = embedding_layer(x)
# x.shape
# x = pack_padded_sequence(x, torch.tensor(138,), batch_first=True)
# x, h,c = encoder(x)
# x, lengths = pad_packed_sequence(x, batch_first=True)
#
# x, lengths = pad_packed_sequence(x, batch_first=True)
# x, lengths = pad_packed_sequence(x, batch_first=True)

h = torch.cat((hn[0],hn[1],hn[2],hn[3]),dim=-1)
h.shape
hs = h.unsqueeze(0).repeat(2, 1, 1)
hs.shape
state_h, state_h_reverse = hn.chunk(2, dim=0)  # 所给的是拆分的个数，即拆分成多少个
state_c, state_c_reverse = cn.chunk(2, dim=0)
state_h.shape#[1, 200, 64]
state_h_reverse.shape#[1, 200, 64]
a = torch.cat((state_h,state_h_reverse),dim=-1)
a.shape#torch.Size([1, 200, 128])
b = torch.cat((hn[0],hn[1]),dim=-1)
b.shape#torch.Size([200, 128])

x = x.permute(0,2,1)#[200, 128, 138]
bn_momentum = 0.9
x = nn.BatchNorm1d(num_features=lstm_dim,momentum=bn_momentum)(x)
x.shape#[200, 128, 138]
x = x.permute(0,2,1)
x.shape#[200, 138, 128]

encoder2 = LSTM(input_size=lstm_dim,hidden_size=lstm_dim // 2,batch_first=True,bidirectional=True)
_, (state_h2_r, state_c2_r) = encoder2(x)
_.shape#[200, 138, 128]
state_h2_r.shape#[2, 200, 64]
state_h2, state_h2_reverse = state_h2_r.chunk(2, dim=0)
state_c2, state_c2_reverse = state_c2_r.chunk(2, dim=0)
state_h2.shape#[1, 200, 64]

states = torch.cat((state_h, state_c, state_h2,state_c2,state_h_reverse,
                    state_c_reverse,state_h2_reverse,state_c2_reverse), dim=-1)
states.shape#[1, 200, 512]
states = states.squeeze(0)
states.shape#[200, 512]

states = nn.BatchNorm1d(num_features=lstm_dim*4,momentum=bn_momentum)(states)
states.shape#[200,512]
states = torch.randn(200,512)
a = states.reshape((states.shape[0], 1, states.shape[1]))
a.shape
codelayer_dim = 128
#h_activation = "relu"
neck_relu = nn.Sequential(
    nn.Linear(lstm_dim*4, codelayer_dim),nn.ReLU()
)
neck_outputs = neck_relu(states)
neck_outputs.shape#[200, 128]

neck_outputs = nn.BatchNorm1d(num_features=neck_outputs.shape[-1],momentum=bn_momentum)(neck_outputs)
neck_outputs.shape#[200, 128]

noise_std = 0.01
neck_outputs = neck_outputs + torch.normal(0,noise_std, size=neck_outputs.shape)
neck_outputs.shape#[200, 128]
###############################################################################
input_shape = (138,35)
x = torch.randn(200,138,35)
x = x.permute(1,0,2)#[138, 200, 35]
lstm_dim = 128
lstm = nn.LSTM(input_size=input_shape[1],hidden_size=lstm_dim // 2,bidirectional=True)
x, (hn, cn)=lstm(x)
#x = x.permute(1,0,2)
x.shape#[138, 200, 128]
hn.shape#[2, 200, 64]
cn.shape#[2, 200, 64]

state_h, state_h_reverse = hn.chunk(2, dim=0)  # 所给的是拆分的个数，即拆分成多少个
state_c, state_c_reverse = cn.chunk(2, dim=0)
state_h.shape#[1, 200, 64]
state_h_reverse.shape#[1, 200, 64]


x = x.permute(0,2,1)#[200, 128, 138]
bn_momentum = 0.9
x = nn.BatchNorm1d(num_features=lstm_dim // 2,momentum=bn_momentum)(x)
x.shape#[200, 128, 138]

x = x.permute(0,2,1)
x.shape#[200, 138, 128]
encoder2 = LSTM(input_size=x.shape[-1],hidden_size=lstm_dim // 2,batch_first=True,bidirectional=True)
_, (state_h2_r, state_c2_r) = encoder2(x)
_.shape#[200, 138, 128]
state_h2_r.shape#[2, 200, 64]
state_h2, state_h2_reverse = state_h2_r.chunk(2, dim=0)
state_c2, state_c2_reverse = state_c2_r.chunk(2, dim=0)
state_h2.shape#[1, 200, 64]

states = torch.cat((state_h, state_c, state_h2,state_c2,state_h_reverse,
                    state_c_reverse,state_h2_reverse,state_c2_reverse), dim=-1)
states.shape#[1, 200, 512]
states = states.squeeze(0)
states.shape#[200, 512]

states = nn.BatchNorm1d(num_features=states.shape[-1],momentum=bn_momentum)(states)
states.shape#[200,512]


###
lstm_layer = nn.LSTM(32,512, 2,batch_first=True, bidirectional=True)
x = torch.randn(200,138,32)
_, (h, c) = lstm_layer (x)
h.shape#torch.Size([4, 200, 512])
h[0].shape
h = h.permute(1, 2, 0).contiguous().view(200, -1)
h.shape#torch.Size([200, 2048])
linear_layer = nn.Linear((int(True) + 1) * 2 * 512,128)
h = linear_layer(h)#torch.Size([200, 128])
h.shape
###

codelayer_dim = 128
#h_activation = "relu"
neck_relu = nn.Sequential(
    nn.Linear(states.shape[-1], codelayer_dim),nn.ReLU()
)
neck_outputs = neck_relu(states)
neck_outputs.shape#[200, 128]

neck_outputs = nn.BatchNorm1d(num_features=neck_outputs.shape[-1],momentum=bn_momentum)(neck_outputs)
neck_outputs.shape#[200, 128]

noise_std = 0.01
neck_outputs = neck_outputs + torch.normal(0,noise_std, size=neck_outputs.shape)
neck_outputs.shape#[200, 128]



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
#torch.cuda.set_device(0)
print('number of GPUs available:{}'.format(torch.cuda.device_count()))
print('device name:{}'.format(torch.cuda.get_device_name(0)))
# 1) 初始化
torch.distributed.init_process_group(backend="gloo",init_method= "env://")
# 2） 配置每个进程的gpu
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)




import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net1 = Net(1, 10, 2)
print(net1)

############################
import torch
import torch.nn.functional as F

net2 = torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(net2)


x = torch.randn(5,1)

net1(x)
net2(x)

##########################################################################################################################


import torch.nn as nn
import torch
embedding = nn.Embedding(1000000,1)
x = torch.randn(200,133,35)
x = torch.randn(5,2,3)
x[:,-1,:].shape
x.view(-1,35).shape
x.view(-1)
x = x.long()
x = abs(torch.LongTensor(x))
x = embedding(x)
x.shape

a = embedding(input)
a.shape

# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# 每批取两组，每组四个单词
input = torch.LongTensor([[0,1,0,0,0],[0,0,0,1,0],[0,0,0,1,0]])
a = embedding(input) # 输出2*5*3
a.shape#torch.Size([2, 5, 3])
a[0],a[1]
####################################################################################
from torch_modelaae import Decoder
from parser import args
decoder = Decoder(args)
decoder.load_state_dict(torch.load(decoder_params.pkl))


import torch
import numpy as np
latent_size = 128
n_batch = 200
max_len=100
lengths = torch.zeros(n_batch, dtype=torch.long)#torch.Size([200])
states = torch.randn(n_batch, latent_size)#torch.Size([200, 128])
prevs = torch.empty(n_batch, 1, dtype=torch.long)#.fill_(vocabulary.bos)#torch.Size([200, 1])
one_lens = torch.ones(n_batch, dtype=torch.long)#torch.Size([200])
is_end = torch.zeros(n_batch, dtype=torch.uint8)#torch.Size([200])
logits =np.array([[[0.6,0.5,0.4,0.3,0.2,0.1],[0.7,0.6,0.5,0.4,0.3,0.2],[0.8,0.7,0.6,0.5,0.4,0.3]]])
logits.shape#(1, 3, 6)
logits = torch.FloatTensor(logits)
logits = torch.softmax(logits, 2)
for i in range(max_len):
    logits, _, states = decoder(prevs, one_lens,
                                     states, i == 0)
    logits = torch.softmax(logits, 2)
    shape = logits.shape[:-1]
    logits = logits.contiguous().view(-1, logits.shape[-1])
    currents = torch.distributions.Categorical(logits).sample()
    currents = currents.view(shape)
# m = torch.distributions.Categorical(torch.tensor([0.1, 0.2, 0.4, 0.3]))
# m.sample()  # equal probability of 0, 1, 2, 3
    is_end[currents.view(-1) == vocabulary.eos] = 1
    if is_end.sum() == max_len:
        break

    currents[is_end, :] = self.vocabulary.pad
    samples.append(currents.cpu())
    lengths[~is_end] += 1

    prevs = currents

import h5py
import torch
from rdkit import  Chem
from vectorizers import SmilesVectorizer
dataset_filename = "datasets/DRD2_TRAIN_MOLS.h5"
with h5py.File(dataset_filename, "r") as f:
    data = f["mols"][:10]
# with open(dataset_filename, "r") as f:
#     self.data = f.read().splitlines()
# self.data = [Chem.MolFromSmiles(i) for i in self.data[:]]
maxlen = 133
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
smilesvec1 = SmilesVectorizer(
    canonical=False,
    augment=True,
    maxlength=maxlen,
    charset=charset,
    binary=True,
)
smilesvec2 = SmilesVectorizer(
    canonical=False,
    augment=True,
    maxlength=maxlen,
    charset=charset,
    binary=True,
    leftpad=False,
)
smilesvec1._char_to_int('(')
smilesvec1._char_to_int.get('(')
smilesvec1._char_to_int.get(')')
# enc_input = smilesvec1.transform(data)  # (17817, 133, 35)
data = [Chem.Mol(i) for i in data[:]]
sm = [Chem.MolToSmiles(i) for i in data]
print("Default Charset %s"%smilesvec1.charset)
print("Default Maximum allowed SMILES length %s"%smilesvec1.maxlength)

smilesvec1.fit(data, extra_chars=["\\"])
print()
print("After fitting")
print("Charset after fit %s"%smilesvec1.charset)
print("Maximum allowed SMILES length %s"%smilesvec1.maxlength)
enc_input = smilesvec1.transform(data)
enc_input2 = torch.FloatTensor(enc_input)
enc_input2.shape
a = torch.argmax(enc_input2, dim=-1)
a = a.numpy()
a = list(a)
smiles_list = []
for j in a:
    smiles = "".join(smilesvec1._int_to_char[i] for i in j)
    smiles_list.append(smiles)

enc_input_smile = smilesvec1.reverse_transform(enc_input,strip=False)
enc_input_smile[0]
dec_in_out = smilesvec2.transform(data)  # (17817, 133, 35)


smiles = ["CCC(=O)O[C@@]1(CC[NH+](C[C@H]1CC=C)C)c2ccccc2",
          "CCC[S@@](=O)c1ccc2c(c1)[nH]/c(=N/C(=O)OC)/[nH]2"] * 10
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
sm_en = SmilesVectorizer(canonical=True, augment=False)
print("Default Charset %s"%sm_en.charset)
print("Default Maximum allowed SMILES length %s"%sm_en.maxlength)

sm_en.fit(mols, extra_chars=["\\"])
print()
print("After fitting")
print("Charset after fit %s"%sm_en.charset)
print("Maximum allowed SMILES length %s"%sm_en.maxlength)
mol_vects = sm_en.transform(mols)
mol_vects.shape
a = sm_en.reverse_transform(mol_vects[0:])
b = sm_en.reverse_transform(mol_vects[0:2], strip=False)
len(a[5])
len(b[0])


from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
x = torch.randn(200,133,35)
lstm_layer = nn.LSTM(35,256, 2,batch_first=True,bidirectional=True)
linear_layer = nn.Linear((1 + 1) *2*256,128)
_, (_, x) = lstm_layer(x)
x.shape#([4, 200, 256])
x = x.permute(1, 2, 0).contiguous().view(200, -1)
x.shape#[200, 1024]
x = linear_layer(x)
x.shape#([200, 128])


latent2hidden_layer = nn.Linear(128, 256)
c0 = latent2hidden_layer(x)
c0 = c0.unsqueeze(0).repeat(lstm_layer.num_layers, 1, 1)
h0 = torch.zeros_like(c0)
states = (h0, c0)
states[0].shape#([2, 200, 256])

lstm_layer = nn.LSTM(35,256,2,batch_first=True)
linear_layer = nn.Linear(256,35)
x1 = torch.randn(200,133,35)

x, states = lstm_layer(x1,states)
x.shape#([200, 133, 256])
states[0].shape#([2, 200, 256])
x = linear_layer(x)
x.shape#([200, 133, 35])