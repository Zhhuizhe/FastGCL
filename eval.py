import torch
import torch.nn.functional as F
from model import GCN, GIN, ViewGenerator, Classifier
from metrics import eval_graph_cls_acc, eval_node_cls_acc
from loader import get_data_loader, load_checkpoint
from sklearn.model_selection import StratifiedKFold
import numpy as np


class Evaluator:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.dataset, self.loader = get_data_loader(self.args)
        if self.args.task == 'node classification':
            self.data = self.dataset[0].to(args.device)
            self.lr_train_mask = self.data.train_mask
            self.lr_val_mask = self.data.val_mask
            self.lr_test_mask = self.data.test_mask
            print('Split for Embedding Learning: Train: {}, Val: {}, Test: {}'
                  ''.format(self.lr_train_mask[0].sum().item(), self.lr_val_mask[0].sum().item(),
                            self.lr_test_mask.size(0) if len(self.lr_test_mask.shape) == 1 else self.lr_test_mask[0].sum().item()))

        # Model definition
        if self.args.task == 'node classification':
            self.gnn_model = GCN(self.dataset.num_node_features, self.args.gnn_layers, self.args.gnn_drop_p).to(self.args.device)
        else:
            self.gnn_model = GIN(self.dataset.num_node_features, self.args.gnn_layers, self.args.gnn_drop_p).to(self.args.device)
        self.vg_model = ViewGenerator(self.args.gnn_layers[-1], self.args.mlp_layers, self.args.mlp_drop_p).to(self.args.device)

    def node_cls_eval(self, input_emb):
        all_preds = []
        input_emb = F.normalize(input_emb, dim=1)
        for i in range(self.lr_train_mask.shape[0]):
            curr_train_mask = self.lr_train_mask[i]
            y_pred = self.linear_regression(input_emb[curr_train_mask], self.data.y[curr_train_mask], input_emb)
            all_preds.append(y_pred)

        # Compute acc over 20 random splits.
        acc_dict = eval_node_cls_acc(
            self.data.y, all_preds, self.lr_train_mask, self.lr_val_mask, self.lr_test_mask)

        return acc_dict["train_acc_mean"], acc_dict["val_acc_mean"], acc_dict["test_acc_mean"]

    def graph_cls_eval(self, input_emb, labels):
        # Must input labels, because dataloader shuffle=Ture, use dataset.data.y is wrong.
        accuracies = []
        x, y = input_emb.cpu().numpy(), labels.cpu().numpy()
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.args.seed)
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            x_train, y_train = torch.from_numpy(x_train).to(self.args.device), torch.from_numpy(y_train).to(self.args.device)
            x_test, y_test = torch.from_numpy(x_test).to(self.args.device), torch.from_numpy(y_test).to(self.args.device)

            y_pred = self.linear_regression(x_train, y_train, x_test)

            # Compute accuracy
            acc_dict = eval_graph_cls_acc(y_test, y_pred)
            accuracies.append(list(acc_dict.values()))
        test_acc = np.mean(accuracies, axis=0)
        return test_acc

    def linear_regression(self, x_train, y_train, x_test):
        # Train LR, need to instantiate new LR model every time as downstream task
        classifier = Classifier(self.args.gnn_emb_dim, self.dataset.num_classes).to(self.args.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.args.lr_cls, weight_decay=self.args.wd_cls)
        classifier.train()
        optimizer.zero_grad()

        # Conduct supervised classification
        total_loss = []
        for _ in range(1, self.args.cls_epochs + 1):
            pred = classifier(x_train)

            loss = F.nll_loss(pred, y_train)
            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        # ==========Predict==========
        classifier.eval()
        with torch.no_grad():
            out = classifier(x_test)
        y_pred = out.argmax(dim=-1, keepdim=True).squeeze(-1)

        return y_pred

    def load_model_from_checkpoint(self):
        model = load_checkpoint(self.args.load_model_name)
        self.gnn_model = model['gnn_model']
        self.vg_model = model['vg_model']
