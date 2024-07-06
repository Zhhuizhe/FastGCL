from openTSNE import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
import random


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tsne_plt(embeddings, labels, save_path=None, title='Title'):
    print('Drawing t-SNE plot ...')
    tsne = TSNE(perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=False)
    embeddings = embeddings.cpu().numpy()
    c = labels.cpu().numpy()

    emb = tsne.fit(embeddings)  # Training

    plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=c, marker='o')
    plt.colorbar()
    plt.grid(True)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def print_model_grad(model):
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requires:', parms.requires_grad, "-->gradient: ", parms.grad)


def print_model_params(model):
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->prams:', parms.weight, "-->gradient: ", parms)
