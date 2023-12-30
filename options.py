import argparse
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--dataset', type=str, default = 'ucf', choices=['ucf', 'xd'])
    parser.add_argument('--metric', type=str, default='AUC', choices=['AUC', 'AP'])
    parser.add_argument('--in_channel', type=int, default = 1024, help='channel of feature')
    
    parser.add_argument('--model_type', type=str, default = 'ur', choices=['rtfm', 'ur'], help='type of model forward')

    parser.add_argument('--need_audio', action='store_true')
    parser.add_argument('--audio_dir', type = str, default = 'data/')

    parser.add_argument('--root_dir', type = str, default = 'data/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--model_path', type = str, default = 'ckpts/')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
    parser.add_argument('--max_iter', type = int, default = 3000, help = 'number of train iteration')
    parser.add_argument('--batch_nor', type = int, default = 64)
    parser.add_argument('--batch_abn', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--num_segments', type = int, default = 200)

    parser.add_argument('--w_bag', type = float, nargs='+', default=[0.1, 0.1, 0.01, 0.01])

    parser.add_argument('--seed', type = int, default = 2022, help = 'random seed (-1 for no manual seed)')
    
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--plot_freq', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--version', type=str, default='train')
    
    return init_args(parser.parse_args())


def init_args(args):

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    datasetname = args.dataset
    if 'ucf' in datasetname:
        args.metric = 'AUC'
    elif 'xd' in datasetname:
        args.metric = 'AP'
    else:
        raise RuntimeError(f'Unknown Dataset: {datasetname}')

    if args.need_audio:
        args.in_channel = 1152
        args.root_dir = [args.root_dir, args.audio_dir]

    if args.debug:
        args.max_iter = 3
        args.plot_freq = 1
        args.version = 'debug'

    return args
