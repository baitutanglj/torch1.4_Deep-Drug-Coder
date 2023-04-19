from moses.char_rnn.model import CharRNN
from moses.char_rnn.trainer import CharRNNTrainer
from moses.char_rnn import config
from moses.utils import CharVocab, Logger
from moses.script_utils import add_train_args
configp = config.get_config()
train_data = ['Cc1cccn2c(CN(C)C3CCCc4ccccc43)c(C(=O)N3CCOCC3)nc12',
             'COC(=O)NN=C(c1ccc(O)cc1)C1C(=O)N(C)C(=O)N(C)C1=O',
             'CCc1cc(CC)nc(OCCCn2c3c(c4cc(-c5nc(C)no5)ccc42)CC(F)(F)CC3)n1',
             'Cc1ccc2c(C(=O)Nc3ccccc3)c(SSc3c(C(=O)Nc4ccccc4)c4ccc(C)cc4n3C)n(C)c2c1',
             'Cc1cccc(-c2ccccc2)c1Oc1nc(O)nc(NCc2ccc3occc3c2)n1',
             'Cn1nnnc1SCC(=O)NN=Cc1ccc(Cl)cc1',
             'COc1cccc(NS(=O)(=O)c2ccc(OC)c(OC)c2)c1',
             'COc1ccc(OC)c(S(=O)(=O)n2nc(C)cc2C)c1',
             'NCCCn1cc(C2=C(c3ccncc3)C(=O)NC2=O)c2ccccc21',
             'CN(C)C(=O)N1CCN(C(c2ccc(Cl)cc2)c2cccnc2)CC1']
# vocabulary = [CharVocab.from_data(data).string2ids(c, add_bos=False, add_eos=False) for c in data]
vocabulary = CharVocab.from_data(train_data)
model = CharRNN(vocabulary,configp)
charrnntrainer = CharRNNTrainer(configp)
t = charrnntrainer.fit(model = model,train_data = train_data, val_data=None)