import torch
import os
import numpy as np


class Logger(object):
    def __init__(self, args=None):
        self.args = args
        self.results = [[] for _ in range(args.runs)]

    def add_result(self, run, result):
        if self.args.task == 'node classification':
            assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def save_checkpoint(self, gnn_model, exp_model, run, epoch):
        checkpoint_name = 'run{:02d}epoch{:03d}.pth'.format(run, epoch)
        path = os.path.join(self.args.model_dir, checkpoint_name)
        torch.save({'gnn_model': gnn_model.state_dict(),
                    'vg_model': exp_model.state_dict()}, path)

    def print_statistics(self, run=None):
        def node_cls_stat(results, name):
            if run is not None:
                result = 100 * torch.tensor(results[run])
                argmax = result[:, 1].argmax().item()
                print('==========Run {:02d} {}=========='.format(run + 1, name))
                print('Highest Train: {:.2f}%'.format(result[:, 0].max()))
                print('Highest Valid: {:.2f}%'.format(result[:, 1].max()))
                print('  Final Train: {:.2f}%'.format(result[argmax, 0]))
                print('   Final Test: {:.2f}%'.format(result[argmax, 2]))
                print('==========Run {:02d} {}=========='.format(run + 1, name))
            else:
                result = 100 * torch.tensor(results)

                best_results = []
                for r in result:
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test = r[r[:, 1].argmax(), 2].item()
                    best_results.append((train1, valid, train2, test))

                best_result = torch.tensor(best_results)

                print('==========All Runs {}=========='.format(name))
                r = best_result[:, 0]
                print('Highest Train: {:.2f} ± {:.2f}'.format(r.mean(), r.std()))
                r = best_result[:, 1]
                print('Highest Valid: {:.2f} ± {:.2f}'.format(r.mean(), r.std()))
                r = best_result[:, 2]
                print('  Final Train: {:.2f} ± {:.2f}'.format(r.mean(), r.std()))
                r = best_result[:, 3]
                print('   Final Test: {:.2f} ± {:.2f}'.format(r.mean(), r.std()))

        def graph_cls_stat(results, name):
            if run is not None:
                result = 100 * np.array(results[run])
                print('==========Run {:02d} {}=========='.format(run + 1, name))
                print('Final Test: {:.2f}%'.format(np.max(result)))
                print('==========Run {:02d} {}=========='.format(run + 1, name))
            else:
                result = 100 * np.array(results)
                best = []
                for r in result:
                    best.append(r.max())
                print('==========All Runs {}=========='.format(name))
                print('Final Test: {:.2f} ± {:.2f}'.format(np.mean(best), np.std(best)))

        if self.args.task == 'node classification':
            node_cls_stat(self.results, name='Node classification')
        else:
            graph_cls_stat(self.results, name='Graph classification')
