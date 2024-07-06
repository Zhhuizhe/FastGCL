import torch
import numpy as np


def eval_graph_cls_acc(y_true, y_pred):
    n_corr = torch.eq(y_true, y_pred).sum().item()
    acc = n_corr / len(y_true)
    return {
        'test_acc': acc
    }


def eval_node_cls_acc(y_true, y_pred, train_mask, val_mask, test_mask):
    train_acc_arr, val_acc_arr, test_acc_arr = [], [], []

    for i in range(len(y_pred)):
        curr_train_mask, curr_val_mask = train_mask[i], val_mask[i]
        if len(test_mask.shape) == 1:
            curr_test_mask = test_mask
        else:
            curr_test_mask = test_mask[i]
        curr_y_pred = y_pred[i]

        train_acc = (torch.eq(curr_y_pred[curr_train_mask], y_true[curr_train_mask]).sum()
                     / curr_train_mask.sum()).item()
        val_acc = (torch.eq(curr_y_pred[curr_val_mask], y_true[curr_val_mask]).sum()
                   / curr_val_mask.sum()).item()
        test_acc = (torch.eq(curr_y_pred[curr_test_mask], y_true[curr_test_mask]).sum()
                    / curr_test_mask.sum()).item()

        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
        test_acc_arr.append(test_acc)

    return {
        "train_acc_mean": np.mean(train_acc_arr),
        "train_acc_std": np.std(train_acc_arr),
        "val_acc_mean": np.mean(val_acc_arr),
        "val_acc_std": np.std(val_acc_arr),
        "test_acc_mean": np.mean(test_acc_arr),
        "test_acc_std": np.std(test_acc_arr)
    }
