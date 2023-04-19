import argparse
parser = argparse.ArgumentParser(description='PyTorch DDC WAE-GAN')
parser.add_argument('-batch_size', type=int, default=10, metavar='N', help='input batch size for training (default: 10)')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs to train (default:10)')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('-lstm_dim', type=int, default=256, help='lstm_dim (default:256)')
parser.add_argument('-bn_momentum', type=int, default=0.9, help='bn_momentum (default: 0.9)')
parser.add_argument('-bn', type=bool, default=True, help='bn (default:True)')
parser.add_argument('-noise_std', type=float, default=0.01, help='noise_std (default:0.01)')
parser.add_argument('-codelayer_dim', type=int, default=128, help='codelayer_dim (default: 128)')
parser.add_argument('-td_dense_dim', type=int, default=0, help='td_dense_dim (default: 0)')
parser.add_argument('-input_shape', type=tuple, default=(133, 35),help='input_shape (default:(133, 33))')
parser.add_argument('-dec_dims', type=list, default=[132, 35], help='dec_dims of (default:[132, 33])')
parser.add_argument('-dec_input_shape', type=list, default=[132, 35], help='dec_input_shape (default:[132, 35])')
parser.add_argument('-output_dims', type=int, default=35, help='output_dims (default: 35)')
parser.add_argument('-num_layer', type=int, default=2, help='num_layer (default: 2)')
parser.add_argument('-clip_value', type=float, default=0.0, help='clip_value (default: 0.0)')
parser.add_argument('-patience', type=int, default=10, help='patience (default: 10)')
parser.add_argument('-maxlen', type=int, default=133, help='maxlen (default: 133)')
parser.add_argument('-charset', type=str, default="r(nF56l=3Bc1-CS2]H#8)\7+[s4oON", help='patience (default: "r(nF56l=3Bc1-CS2]H#8)\7+[s4oON")')
# args = parser.parse_args()
args, unparsed = parser.parse_known_args()
# input_shape = smilesvec1.dims#(138, 35)
# dec_dims = list(smilesvec1.dims)#[138, 35]
# dec_dims[0] = dec_dims[0] - 1
# dec_input_shape = dec_dims#[137, 35]
# output_len = smilesvec1.dims[0] - 1#137
# output_dims = smilesvec1.dims[-1]#35