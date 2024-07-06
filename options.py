import argparse
import os
import torch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()

    # Data parameter
    parser.add_argument('--dataset_name', default='REDDIT-BINARY',
                        choices=[
                            # Node classification
                            'Wiki-CS', 'Computers', 'Photo', 'CS', 'Physics',
                            # Graph classification with node feature
                            'NCI1', 'DD', 'PROTEINS', 'COLLAB',
                            # Graph classification without node feature
                            'REDDIT-BINARY', 'REDDIT-MULTI-5K'
                        ], help='dataset name')
    parser.add_argument('--data_path', default='./data', help='data path')
    parser.add_argument('--mask_path', default='./mask', help='path of dataset splits')
    parser.add_argument('--batch_size', type=int, default=16, help='here is the number of graph loaded at once')

    # Model parameter
    parser.add_argument('--gnn_layers', nargs='+', type=int, default=[64, 64, 64, 64],
                        help='the output dimension of each layer of gnn')
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[128], help='the output dimension of each layer of mlp')
    parser.add_argument('--gnn_drop_p', type=float, default=0.1)
    parser.add_argument('--mlp_drop_p', type=float, default=0.5)

    # Run parameter
    parser.add_argument('--device', type=int, default=1, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=20, help='the number of pretext task training epochs')
    parser.add_argument('--seed', type=int, default=123, help='random seed, to fix result')
    parser.add_argument('--runs', type=int, default=10, help='the number of runs')
    parser.add_argument('--log_steps', type=int, default=1, help='frequency of log output')
    parser.add_argument('--cls_epochs', type=int, default=200, help='the number of downstream task training epochs')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save model')

    # Learning parameter
    parser.add_argument('--lr_gnn', type=float, default=1e-5, help='learning rate of gnn model')
    parser.add_argument('--lr_vg', type=float, default=1e-2, help='leaning rate of view generator model')
    parser.add_argument('--lr_cls', type=float, default=1e-2, help='learning rate of classifier')
    parser.add_argument('--wd_gnn', type=float, default=5e-4, help='weight decay of gnn model')
    parser.add_argument('--wd_vg', type=float, default=5e-4, help='weight decay of view generator model')
    parser.add_argument('--wd_cls', type=float, default=5e-4, help='weight decay of classifier')
    parser.add_argument('--lambda_norm', type=float, default=0, help='regular strength')

    # Save parameter
    parser.add_argument('--save_suffix', type=str, default='fast-gcl-{}'.format(datetime.now().strftime('%y%m%d')),
                        help='some comment for model or test result dir')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')

    # Resume parameter
    parser.add_argument('--load_model_path', type=str,
                        default='checkpoints/fast-gcl-20220114/model/run01epoch001.pth',
                        help='model path for pretrain or test')
    parser.add_argument('--load_triplet_path', type=str,
                        default='checkpoints/fast-gcl-20220114/result/run01epoch001_triplet.pth',
                        help='model path for pretrain or test')

    args = parser.parse_args()
    args.num_layers = len(args.gnn_layers)
    args.device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    if args.dataset_name in ['Wiki-CS', 'Computers', 'Photo', 'CS', 'Physics']:
        args.task = 'node classification'
        args.gnn_emb_dim = args.gnn_layers[-1]
    else:
        args.task = 'graph classification'
        args.gnn_emb_dim = sum(args.gnn_layers)
    return args


def make_dir(args):
    base_dir = os.path.join('checkpoints', args.dataset_name.lower() + '_' + args.save_suffix)
    model_dir = os.path.join(base_dir, 'model')
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)

    result_dir = os.path.join(base_dir, 'result')
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)

    args.model_dir = model_dir
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, args.save_suffix + '_args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_args():
    args = parse_args()
    make_dir(args)
    save_args(args, args.model_dir)
    return args
