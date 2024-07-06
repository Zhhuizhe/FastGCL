import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from conv.wgin_conv import WGINConv
from torch_geometric.nn import global_add_pool


class ViewGenerator(nn.Module):
    def __init__(self, gnn_embed_dim, mlp_layers, mlp_drop_p=0.5):
        super(ViewGenerator, self).__init__()
        assert len(mlp_layers) > 0
        self.gnn_embed_dim = gnn_embed_dim
        self.mlp_layers = mlp_layers
        self.dropout = mlp_drop_p

        mlp_module = []
        hid_dim = gnn_embed_dim
        for out_dim in self.mlp_layers[:-1]:
            mlp_module.append(nn.Linear(hid_dim, out_dim))
            mlp_module.append(nn.ReLU())
            mlp_module.append(nn.Dropout(p=self.dropout))
            hid_dim = out_dim
        mlp_module.append(nn.Linear(hid_dim, self.mlp_layers[-1]))
        self.mlp = nn.Sequential(*mlp_module)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_emb, adj_t):
        edge_index = adj_t.coo()
        pos_edge_index = torch.vstack((edge_index[0], edge_index[1]))
        neg_edge_index = torch.vstack((torch.arange(node_emb.size(0)), torch.arange(node_emb.size(0)))).to(node_emb.device)

        z = self.mlp(node_emb)
        e_u = z[edge_index[0]]
        e_v = z[edge_index[1]]
        edge_weights = self.sigmoid(torch.sum(torch.mul(e_u, e_v), dim=1))

        return pos_edge_index, neg_edge_index, edge_weights


class GCN(nn.Module):
    def __init__(self, feat_dim, gnn_layers, gnn_drop_p=0.5, add_self_loops=False):
        super(GCN, self).__init__()
        # assert len(gnn_layers) > 1
        self.feat_dim = feat_dim
        self.gnn_layers = gnn_layers
        self.dropout = gnn_drop_p

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)

        hid_dim = self.feat_dim
        for out_dim in gnn_layers[:-1]:
            self.convs.append(GCNConv(hid_dim, out_dim, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(out_dim))
            hid_dim = out_dim
        self.convs.append(GCNConv(hid_dim, gnn_layers[-1], add_self_loops=add_self_loops))

    def forward(self, x, adj_t, batch=None, edge_weight=None):
        assert batch is None
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, adj_t, edge_weight)
        return x, x  # node emb for mlp and loss


class GIN(torch.nn.Module):
    def __init__(self, feat_dim, gnn_layers, gnn_drop_p=0.5):
        super(GIN, self).__init__()
        self.feat_dim = feat_dim if feat_dim != 0 else 1
        self.gnn_layers = gnn_layers
        self.dropout = gnn_drop_p

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)

        hid_dim = self.feat_dim
        for out_dim in gnn_layers:
            self.convs.append(WGINConv(nn.Sequential(
                nn.Linear(hid_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))))
            self.bns.append(nn.BatchNorm1d(out_dim))
            hid_dim = out_dim

    def forward(self, x, adj_t, batch, edge_weight=None):
        assert batch is not None
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(batch.device)

        xs = []  # node emb
        for i in range(len(self.gnn_layers)):
            x = self.relu(self.convs[i](x, adj_t, edge_weight))
            x = self.bns[i](x)
            x = self.dropout(x)
            xs.append(x)

        x_pool = [global_add_pool(x, batch) for x in xs]
        x_pool = torch.cat(x_pool, dim=1)

        return xs[-1], x_pool  # node emb for mlp, graph emb for loss


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        return x.log_softmax(dim=-1)
