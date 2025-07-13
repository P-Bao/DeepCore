from .earlytrain import EarlyTrain
import torch
from torch.utils.data import DataLoader
from .methods_utils import FacilityLocation, submodular_optimizer, OrthogonalMP_REG, OrthogonalMP_REG_Parallel
import numpy as np
from .methods_utils.euclidean import euclidean_dist_pair_np
import methods_utils.euclidean as eu_module
from ..nets.nets_utils import MyDataParallel

class OMP(EarlyTrain):
    """
    Selector using Orthogonal Matching Pursuit (regularized) on gradient vectors.
    A: gradient matrix of shape (n_samples, feature_dim)
    b: sum of gradient vectors across samples.
    Supports both CPU and CUDA via OrthogonalMP_REG(_Parallel).
    """
    def __init__(self, dst_train, args, lam=0.1, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=True, greedy="LazyGreedy", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self.balance = balance
        self.lam = lam

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def calc_gradient(self, index=None):
        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch, num_workers=self.args.workers)
        sample_num = len(self.dst_val.targets) if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(outputs.requires_grad_(True),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1,
                                                                                       self.embedding_dim).repeat(1,
                                                                                                                  self.args.num_classes,
                                                                                                                  1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())

        gradients = np.concatenate(gradients, axis=0)

        self.model.train()
        return euclidean_dist_pair_np(gradients)

    # def calc_gradient_vectors(self):
    #     orig = eu_module.euclidean_dist_pair_np
    #     # patch to identity over first argument
    #     eu_module.euclidean_dist_pair_np = lambda *args, **kwargs: args[0]
    #     try:
    #         # explicitly call Craig's implementation
    #         return calc_gradient(self, index)
    #     finally:
    #         eu_module.euclidean_dist_pair_np = orig

    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        # Build A and b using original calc_gradient
        A = self.calc_gradient()
        b = A.sum(axis=0)
        nnz = int(self.fraction * A.shape[0])

        # Run OMP based on device
        if self.args.device == 'cuda':
            A_t = torch.from_numpy(A).to('cuda')
            b_t = torch.from_numpy(b).to('cuda')
            x_t = OrthogonalMP_REG_Parallel(A_t, b_t, tol=1e-4,
                                            nnz=nnz, positive=False,
                                            lam=self.lam, device='cuda')
            x = x_t.cpu().numpy()
        else:
            x = OrthogonalMP_REG(A, b, tol=1e-4,
                                 nnz=nnz, positive=False,
                                 lam=self.lam)

        selected = np.nonzero(x)[0]
        weights = x[selected]
        return {"indices": selected.astype(np.int32), "weights": weights}
    
    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
