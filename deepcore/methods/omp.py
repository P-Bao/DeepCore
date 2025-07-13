from .earlytrain import EarlyTrain
import torch
from torch.utils.data import DataLoader
from .methods_utils import FacilityLocation, submodular_optimizer, OrthogonalMP_REG, OrthogonalMP_REG_Parallel
import numpy as np
from .methods_utils.euclidean import euclidean_dist_pair_np, euclidean_dist_np
# import methods_utils.euclidean as eu_module
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

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                            torch.utils.data.Subset(self.dst_train, index),
                                            batch_size=self.args.selection_batch,
                                            num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding_recorder.embedding

        self.model.no_grad = False
        return matrix
    
    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
        self.model.no_grad = True
        with self.model.embedding_recorder:
            # chọn nguồn dữ liệu để xây A
            if self.use_embedding:
                self.emb_dim = self.model.embedding_recorder.embedding.shape[1]
                A = self.construct_matrix()
            else:
                A = torch.from_numpy(self.calc_gradient_vectors()).to(self.args.device)
            b = A.sum(dim=0)
            nnz = int(self.fraction * A.shape[0])

            if self.args.device == 'cuda':
                x_t = OrthogonalMP_REG_Parallel(A, b, tol=1e-4, nnz=nnz,
                                                positive=False, lam=self.lam, device='cuda')
                x = x_t.cpu().numpy()
            else:
                A_np = A.cpu().numpy()
                b_np = b.cpu().numpy()
                x = OrthogonalMP_REG(A_np, b_np, tol=1e-4, nnz=nnz, positive=False, lam=self.lam)

            indices = np.nonzero(x)[0].astype(np.int32)
            mask = np.zeros_like(x, dtype=bool)
            mask[indices] = True

            if self.use_embedding:
                matrix = -1.0 * euclidean_dist_np(A_np[indices], A_np[indices])
            else:
                matrix = -1.0 * self.calc_gradient(indices)
                matrix -= np.min(matrix) - 1e-3

            weights = self.calc_weights(matrix, mask)
        self.model.no_grad = False
        return {"indices": indices, "weights": weights}


    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
