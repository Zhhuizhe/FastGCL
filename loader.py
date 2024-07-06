import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Coauthor, Amazon, WikiCS
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected


def get_data_loader(args):
    if args.dataset_name in ['CS', 'Physics']:
        # Load Coauthor-CS/Coauthor-Physics
        dataset = Coauthor(args.data_path, args.dataset_name, transform=T.Compose([T.AddSelfLoops(), T.ToSparseTensor()]))
    elif args.dataset_name == 'Wiki-CS':
        # Load Wiki-CS
        dataset = WikiCS(os.path.join(args.data_path, 'Wiki-CS'), transform=T.Compose([T.AddSelfLoops(), T.ToSparseTensor()]))
        dataset.data.edge_index = to_undirected(dataset.data.edge_index)
    elif args.dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(args.data_path, args.dataset_name, transform=T.Compose([T.AddSelfLoops(), T.ToSparseTensor()]))
    else:
        # Load Graph classification dataset
        dataset = TUDataset(args.data_path, args.dataset_name, transform=T.ToSparseTensor())

    if args.task == 'node classification':
        train_mask, val_mask, test_mask = load_mask(os.path.join(args.mask_path, '{}_mask.pth'.format(args.dataset_name)))
        dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = train_mask, val_mask, test_mask

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataset, loader


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint['gnn_model'], checkpoint['vg_model']


def load_mask(path):
    assert path is not None
    return torch.load(path)
