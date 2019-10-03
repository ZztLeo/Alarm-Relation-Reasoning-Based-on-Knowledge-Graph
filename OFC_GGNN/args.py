import argparse

parser = argparse.ArgumentParser(description='GGNN alarm Graph')
parser.add_argument('--alarm_data', type=str , metavar='PATH', help='path to dataset')
parser.add_argument('--Graph_structure', type=str , metavar='PATH', help='path to graph structure')
parser.add_argument('--cuda', action='store_true',default='True', help='enables cuda')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
					help='path to weights (default: none)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N',
					help='mini-batch size (default: 1)')
parser.add_argument('--print', default='True', help='write out or not')

args = parser.parse_args()
