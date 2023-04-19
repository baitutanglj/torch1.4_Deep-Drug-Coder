import torch
from my_ddc.utils import CharVocab
from my_ddc.config import args
from my_ddc3.model import AAE
n_batch = 10
max_len = 100
latent_size = 128

with open('datasets/ChEMBL_training_set', "r") as f:
    data = f.read().splitlines()
vocabulary = CharVocab.from_data(data)

def tensor2string(tensor):
    ids = tensor.tolist()
    string = vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)
    return string

def sample_latent(n):
    return torch.randn(n,latent_size)

model = AAE(vocabulary,args)
model.load_state_dict(torch.load('torch_models/_train_040.pt'))
model = model.eval()

samples = []
lengths = torch.zeros(n_batch, dtype=torch.long)#tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
states = sample_latent(n_batch)
print(states.shape)#torch.Size([10, 128])
prevs = torch.empty(n_batch, 1, dtype=torch.long).fill_(vocabulary.bos)#ensor([[32],10个,[32]])
one_lens = torch.ones(n_batch, dtype=torch.long)#tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
is_end = torch.zeros(n_batch, dtype=torch.bool)#tensor([False, False, False, False, False, False, False, False, False, False])

for i in range(max_len):
    logits, _, states = model.decoder(prevs, one_lens,states, i == 0)
    print(logits.shape)#torch.Size([10, 1, 36])
    print(_.shape)# torch.Size([10])
    print(states[0].shape)#torch.Size([2, 10, 512])
    logits = torch.softmax(logits, 2)
    print(logits.shape)# torch.Size([10, 1, 36])
    shape = logits.shape[:-1]#torch.Size([10, 1])
    logits = logits.contiguous().view(-1, logits.shape[-1])
    print(logits.shape)#torch.Size([10,36])
    #torch.distributions.Categorical:根据概率分布来产生sample，产生的sample是输入tensor的index
    currents = torch.distributions.Categorical(logits).sample()#tensor([21, 18, 18, 18, 22, 18, 22, 18, 18, 18])
    currents = currents.view(shape)#tensor([[21],[18],...[18]])
    currents = torch.tensor([21, 18, 33, 18, 22, 18, 22, 18, 18, 18])
    is_end[currents.view(-1) == vocabulary.eos] = 1
    if is_end.sum() == max_len:
        break

    currents[is_end, :] = vocabulary.pad
    samples.append(currents.cpu())#[tensor([21, 18, 33, 18, 22, 18, 22, 18, 18, 18])]
    lengths[~is_end] += 1

    prevs = currents

if len(samples):
    samples = torch.cat(samples, dim=-1)#tensor([21, 18, 33, 18, 22, 18, 22, 18, 18, 18])
    samples = [
        tensor2string(t[:l])
        for t, l in zip(samples, lengths)
    ]
else:
    samples = ['' for _ in range(n_batch)]




