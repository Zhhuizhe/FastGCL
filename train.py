import torch
import torch.nn.functional as F
from logger import Logger
from model import GCN, ViewGenerator, GIN
import numpy as np
from tqdm import tqdm
from utils import fix_seed
from loss import nce_loss
from options import prepare_args
from eval import Evaluator


class Trainer:
    def __init__(self, args, logger, evaluator, run):
        self.args = args
        self.run = run
        self.logger = logger
        self.evaluator = evaluator

        self.dataset, self.loader = evaluator.dataset, evaluator.loader

        # Model definition
        self.vg_model = ViewGenerator(self.args.gnn_layers[-1], self.args.mlp_layers, self.args.mlp_drop_p).to(self.args.device)
        if self.args.task == 'node classification':
            self.data = evaluator.data  # get first graph (only one graph)
            self.gnn_model = GCN(self.dataset.num_node_features, self.args.gnn_layers, self.args.gnn_drop_p).to(self.args.device)
        else:
            self.gnn_model = GIN(self.dataset.num_node_features, self.args.gnn_layers, self.args.gnn_drop_p).to(self.args.device)
        self.optimizer = torch.optim.Adam([{'params': self.gnn_model.parameters(),
                                            'lr': self.args.lr_gnn, 'weight_decay': self.args.wd_gnn},
                                           {'params': self.vg_model.parameters(),
                                            'lr': self.args.lr_vg, 'weight_decay': self.args.wd_vg}])

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            # 1. Train for one epoch
            loss = self.train_per_epoch()

            # 2. Val for one epoch
            result = self.val_per_epoch()
            self.logger.add_result(self.run, result)

            # 3. Save model
            if self.args.save_model:
                self.logger.save_checkpoint(self.gnn_model, self.vg_model, self.run + 1, epoch)

            if epoch % self.args.log_steps == 0:
                if len(result) > 1:
                    train_acc, valid_acc, test_acc = result
                    print('Run: {:02d}, Epoch: {:02d}, Loss: {:.4f}, Train: {:.2f}%, Valid: {:.2f}% Test: {:.2f}%'
                          ''.format(self.run + 1, epoch, loss, 100 * train_acc, 100 * valid_acc, 100 * test_acc))
                else:
                    print('Run: {:02d}, Epoch: {:02d}, Loss: {:.4f}, Test: {:.2f}%'
                          ''.format(self.run + 1, epoch, loss, 100 * result.item()))
        self.logger.print_statistics(self.run)

    def train_per_epoch(self):
        def loss_and_backward(gnn_model, vg_model, optimizer, feat, adj_t, batch=None):
            # Optimizer for a graph
            node_emb, anchor_emb = gnn_model(feat, adj_t, batch)
            adj_pos_t, adj_neg_t, pos_weight = vg_model(node_emb, adj_t)
            _, pos_emb = gnn_model(feat, adj_pos_t, batch, pos_weight)
            _, neg_emb = gnn_model(feat, adj_neg_t, batch)

            loss = nce_loss(anchor_emb, pos_emb, neg_emb) - self.args.lambda_norm * torch.mean(F.softplus(1 - pos_weight))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return loss.item()

        # Switch to train mode
        self.gnn_model.train()
        self.vg_model.train()

        # Self-supervise learning
        total_loss = []
        if self.args.task == 'node classification':
            total_loss.append(loss_and_backward(self.gnn_model, self.vg_model, self.optimizer, self.data.x, self.data.adj_t))

        elif self.args.task == 'graph classification':
            for data in tqdm(self.loader, desc='Embedding learning ... '):
                data = data.to(self.args.device)
                total_loss.append(loss_and_backward(self.gnn_model, self.vg_model, self.optimizer, data.x, data.adj_t, data.batch))

        return np.mean(total_loss)

    def val_per_epoch(self):
        # Eval by origin graph
        self.gnn_model.eval()

        if self.args.task == 'node classification':
            with torch.no_grad():
                input_emb, _ = self.gnn_model(self.data.x, self.data.adj_t)
            return self.evaluator.node_cls_eval(input_emb)
        else:
            with torch.no_grad():
                input_emb, labels = [], []
                for data in tqdm(self.loader, desc='Graph Readout ... '):
                    data = data.to(self.args.device)
                    _, graph_emb = self.gnn_model(data.x, data.adj_t, data.batch)
                    input_emb.append(graph_emb)
                    labels.append(data.y)
                input_emb = torch.cat(input_emb, dim=0).squeeze(1)
                labels = torch.cat(labels, dim=0)

            return self.evaluator.graph_cls_eval(input_emb, labels)


def main():
    args = prepare_args()
    print(args)
    logger = Logger(args)
    evaluator = Evaluator(args, logger)
    for run in range(args.runs):
        fix_seed(args.seed + run)
        trainer = Trainer(args, logger, evaluator, run)
        trainer.train()
    logger.print_statistics()


if __name__ == "__main__":
    main()
