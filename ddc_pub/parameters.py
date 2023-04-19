import argparse

parser = argparse.ArgumentParser(description='PyTorch DDC WAE-GAN')
parser.add_argument('-batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('-lstm_dim', type=int, default=128, help='lstm_dim (default:128)')
parser.add_argument('-bn_momentum', type=int, default=35, help='bn_momentum (default: 0.9)')
parser.add_argument('-bn', type=bool, default=True, help='bn (default:True)')
parser.add_argument('-noise_std', type=float, default=0.01, help='noise_std (default:0.01)')
parser.add_argument('-codelayer_dim', type=int, default=128, help='codelayer_dim (default: 128)')
parser.add_argument('-td_dense_dim', type=int, default=0, help='td_dense_dim (default: 0)')
parser.add_argument('-input_shape', type=tuple, default=(133, 35),help='input_shape (default:(133, 35))')
parser.add_argument('-dec_dims', type=list, default=[132, 35], help='dec_dims of (default:[132, 35])')
parser.add_argument('-dec_input_shape', type=list, default=[132, 35], help='dec_dims of (default:[132, 35])')
parser.add_argument('-output_dims', type=int, default=35, help='output_dims (default: 35)')
parser.add_argument('-dec_layers', type=int, default=2, help='dec_layers (default: 2)')
args = parser.parse_args()

