from typing import Callable, Iterable, Union, List

import sigkernel
import torchcde
import torch
import signatory

from src.gan.base import MLP

import random
from src.utils.helper_functions.data_helper_functions import subtract_initial_point

class SummedKernel(object):
    def __init__(self, kernels: List[sigkernel.SigKernel]):
        self._kernels = kernels
        self.n_kernels = len(kernels)

    def compute_scoring_rule(self, X, y, max_batch=128):
        res = 0
        for k in self._kernels:
            res += k.compute_scoring_rule(X, y, max_batch=max_batch)
        return res/self.n_kernels

    def compute_mmd(self, X, Y, max_batch=128):
        res = 0
        for k in self._kernels:
            res += k.compute_mmd(X, Y, max_batch=max_batch)

        return res/self.n_kernels


def get_kernel(kernel_type, dyadic_order, sigma=1.):
    if kernel_type.lower() == "rbf":
        static_kernel = sigkernel.RBFKernel(sigma=sigma)
    else:
        # elif kernel_type.lower() == "linear":
        static_kernel = sigkernel.LinearKernel()

    return sigkernel.SigKernel(static_kernel=static_kernel, dyadic_order=dyadic_order)


def initialise_signature_kernel(**kwargs) -> Union[sigkernel.SigKernel, SummedKernel]:
    """
    Helper function for classes that use the signature kernel

    :param kwargs:  Signature kernel kwargs, must include kernel_type and dyadic_order
    :return:        SigKernel object
    """
    kernel_type  = kwargs.get("kernel_type")
    dyadic_order = kwargs.get("dyadic_order")
    sigma        = kwargs.get("sigma")

    if type(sigma) == float:
        return get_kernel(kernel_type, dyadic_order, sigma=sigma)
    elif type(sigma) == list:
        return SummedKernel([get_kernel(kernel_type, dyadic_order, sigma=sig) for sig in sigma])


class KernelDiscriminator(torch.nn.Module):
    def __init__(self, kernel_kwargs: dict):
        super().__init__()
        self.kernel_kwargs = kernel_kwargs
        # self._kernel = self._init_kernel()
        self._kernel = None
        self._metric = None

    def _init_kernel(self):
        raise NotImplementedError

    def _init_metric(self) -> Callable:
        """
        We could set this here in abstract but methods may be implemented already within a package eg. sigkernel.

        :return:    MMD callable object
        """
        raise NotImplementedError

    def forward(self, x, y) -> torch.tensor:
        raise NotImplementedError

### This class can go
class PathMMDDiscriminator(KernelDiscriminator):
    """
    Path-space discrimination
    """
    def __init__(self, kernel_kwargs: dict, path_dim: int, adversarial: bool = True):
        super().__init__(kernel_kwargs)

        if adversarial:
            inits = torch.ones(path_dim)
            self._sigma = torch.nn.Parameter(inits, requires_grad=True)
        else:
            self._sigma = None

        # self._kernel = self._init_kernel()
        self._kernel = None
        self._metric = None

    def _init_kernel(self):
        raise NotImplementedError

    def _init_metric(self) -> Callable:
        raise NotImplementedError

    def forward(self, x, y):
        """
        Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
        also have some initial point penalty.

        :param x:   Path data, shape (batch, stream, channel). Must require grad for training
        :param y:   Path data, shape (batch, stream, channel). Should not require grad
        :return:    Mixture MMD + initial point loss
        """
        mu = torch.clone(x.type(torch.float64))
        nu = torch.clone(y.type(torch.float64))

        if self._sigma is not None:
            mu[..., 1:] *= self._sigma

            # with torch.no_grad():
            #    nu[..., 1:] *= self._sigma

        return self._metric(mu, nu)


class SigKerMMDDiscriminator(PathMMDDiscriminator):
    def __init__(self, kernel_type: str, dyadic_order: int, path_dim: int, sigma: float = 1., adversarial: bool = True,
                 max_batch: int = 128, use_phi_kernel = False, n_scalings = 0, **kwargs):
        """
        Init method for MMD discriminator using the signature kernel in the MMD calculation.

        :param kernel_type:     Type of static kernel to compose with the signature kernel. Current choices are "rbf"
                                and "linear".
        :param dyadic_order:    Dyadic partitioning of PDE solver used to estimate the lifted signature kernel.
        :param path_dim:        Dimension of path outputs, to specify number of scalings in each path dimension.
                                Should not include time
        :param sigma:           Optional fixed scaling parameter for use in the RBF kernel.
        :param adversarial:     Whether to adversarialise the discriminator or not
        :param max_batch:       Maximum batch size used in computation
        :param use_phi_kernel:  Optional. Whether to implement an approximation for the generalized signature kernel
                                <., .>_\phi where \phi(k) = (k/2)!
        :param n_scalings:      Number of scalings to use in the phi-kernel.
        """
        kernel_kwargs = {
            "kernel_type": kernel_type, "dyadic_order": dyadic_order, "sigma": sigma
        }

        self.max_batch = max_batch

        super().__init__(kernel_kwargs, path_dim, adversarial)

        if use_phi_kernel:
            self._phi_kernel = True
            self._scalings   = torch.zeros(n_scalings).exponential_()
        else:
            self._phi_kernel = False
            self._scalings = None

        self._kernel = self._init_kernel()
        self._metric = self._init_metric()

    def _init_kernel(self) -> sigkernel.SigKernel:
        """
        Inits kernel for SigKerMMDDiscriminator object, using the SigKer package
        :return:
        """

        return initialise_signature_kernel(**self.kernel_kwargs)

    def _init_metric(self):
        """
        Initialises the MMD calculation for the Signature Kernel MMD Discriminator
        :return:
        """

        def metric(X, Y, pi=None):
            if pi is None:
                return self._kernel.compute_mmd(X, Y, max_batch=self.max_batch)
            else:
                piX = X.clone()*pi
                piY = X.clone()*pi
                K_XX = self._kernel.compute_Gram(piX, X, sym=False, max_batch=self.max_batch)
                # Because k_\phi(\piX, Y) = k_\phi(X, \piY), we only need to calculate wrt one scaling
                K_XY = self._kernel.compute_Gram(piX, Y, sym=False, max_batch=self.max_batch)
                K_YY = self._kernel.compute_Gram(piY, Y, sym=False, max_batch=self.max_batch)

                mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
                mK_YY = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

                return mK_XX + mK_YY - 2.*torch.mean(K_XY)

        if self._phi_kernel:
            def _weighted_metric(x, y):
                loss = 0
                n_scalings = len(self._scalings)
                for scale in self._scalings:
                    mu = x.clone()
                    nu = y.clone()

                    mu *= scale

                    loss += metric(mu, nu)

                return loss/n_scalings

            return _weighted_metric
        else:
            return metric


class WeightedSigKerDiscriminator(SigKerMMDDiscriminator):
    def __init__(self, path_scalings: List[float], weights: List[float], kernel_type: str, dyadic_order: int,
                 path_dim: int, sigma: float = 1., adversarial: bool = True, max_batch=128):

        assert len(path_scalings) == len(weights), "Error: number of weights must match number of scalings"
        assert sum(weights) == 1, "Error: initial weights do not sum to 1"
        assert (all([w <= 1 for w in weights])) and (all([w >= 0 for w in weights])), \
            "Error: weights not between 0 and 1"

        super().__init__(kernel_type, dyadic_order, path_dim, sigma, adversarial, max_batch)

        if adversarial:
            _weights     = torch.tensor(weights)
            self._weights = torch.nn.Parameter(_weights, requires_grad=True)

            _sigma = torch.tensor(path_scalings)
            self._sigma = torch.nn.Parameter(_sigma, requires_grad=True)
        else:
            self._weights = torch.tensor(weights)
            self._sigma = torch.tensor(path_scalings)

        self._kernel = self._init_kernel()
        self._mmd    = lambda x, y: self._kernel.compute_mmd(x, y, max_batch=self.max_batch)
        self._metric = self._weighted_metric()

    def _weighted_metric(self):
        def _metric(x, y):
            # 2. Initialize loss
            loss = 0

            # 3. Iterate over scales and weights
            for scale, weight in zip(self._sigma, self._weights):
                mu = x.clone()
                nu = y.clone()

                mu[..., 1:] *= scale

                with torch.no_grad():
                    nu[..., 1:] *= scale

                loss += weight*self._mmd(mu, nu)
            return loss
        return _metric

    def forward(self, x, y):
        mu = torch.clone(x.type(torch.float64))
        nu = torch.clone(y.type(torch.float64))

        return self._metric(mu, nu)

    #def _clip_and_scale_weights(self):
    #    self._weights = torch.nn.Parameter(self._weights.clone().clip(0, 1), requires_grad=True)
    #    self._weights = torch.nn.Parameter(self._weights.clone()/torch.sum(self._weights.clone()), requires_grad=True)


class ScaledSigKerDiscriminator(SigKerMMDDiscriminator):
    def __init__(self, path_scalings: List[float], kernel_type: str, dyadic_order: int, path_dim: int, sigma: float = 1.,
                 adversarial: bool = True, max_batch = 128):

        super().__init__(kernel_type, dyadic_order, path_dim, sigma, adversarial, max_batch)

        self._path_scalings = path_scalings
        self._mmd    = lambda x, y: self._kernel.compute_mmd(x, y, max_batch=self.max_batch)
        self._metric = self._scaled_metric()

    def _scaled_metric(self):
        scalings   = self._path_scalings
        n_scalings = len(scalings)

        def _metric(x, y):
            res = 0
            for scale in scalings:
                mu = x.clone()
                nu = y.clone()

                mu[..., 1:] = x[..., 1:]*scale
                nu[..., 1:] = y[..., 1:]*scale

                res += self._mmd(mu, nu)

            return res/n_scalings
        return _metric


class SigKerScoreDiscriminator(PathMMDDiscriminator):
    """
    Discriminator that a) uses the signature kernel and b) uses a scoring rule instead of the MMD.
    Have to implement initialisation of signature kernel again, until I can figure out grandparent class inheritance.
    """
    def __init__(self, kernel_type: str, dyadic_order: int, path_dim: int, sigma: float = 1., adversarial: bool = True,
                 max_batch = 128, use_phi_kernel = False, n_scalings = 0):

        kernel_kwargs = {
            "kernel_type": kernel_type, "dyadic_order": dyadic_order, "sigma": sigma
        }

        self.max_batch = max_batch

        super().__init__(kernel_kwargs, path_dim, adversarial)

        if use_phi_kernel:
            self._phi_kernel = True
            self._scalings   = torch.zeros(n_scalings).exponential_()
        else:
            self._phi_kernel = False
            self._scalings   = None

        self._kernel       = self._init_kernel()
        self._metric       = self._init_metric()

    def _init_kernel(self) -> sigkernel.SigKernel:
        """
        Inits kernel for SigKerMMDDiscriminator object, using the SigKer package
        :return:
        """

        return initialise_signature_kernel(**self.kernel_kwargs)

    def _init_metric(self) -> Callable:

        def scoring_rule(X, y, pi=None):
            if pi is None:
                return self._kernel.compute_scoring_rule(X, y.unsqueeze(0), max_batch=self.max_batch)
            else:
                piX = X.clone()*pi
                K_XX = self._kernel.compute_Gram(piX, X, sym=False, max_batch=self.max_batch)
                K_Xy = self._kernel.compute_Gram(piX, y.unsqueeze(0), sym=False, max_batch=self.max_batch)

                mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

                return mK_XX - 2.*torch.mean(K_Xy)

        if self._phi_kernel:
            def _weighted_metric(x, y):
                loss = 0
                n_scalings = len(self._scalings)
                for scale in self._scalings:
                    mu = x.clone()
                    nu = y.clone()

                    loss += scoring_rule(mu, nu, pi=torch.sqrt(scale))

                return loss / n_scalings

            return _weighted_metric
        else:
            return scoring_rule

    def forward(self, x, y):
        """
        Forward method for discriminator that uses scoring rules. The discriminator is given by
                                        D(P, Q) = E_Q[S(P, Y)],
        where S(P, Y) = E_P[k(X, X)] - 2E_P[k(X, Y)].


        :param x:       Set of data in form (batch, stream, channel). Generated data, must require grad
        :param y:       Set of data in form (batch, stream, channel). Real data, must NOT require grad
        :return:        Discriminator loss (kernel scoring rule loss between mu and nu)
        """

        mu = torch.clone(x.type(torch.float64))
        nu = torch.clone(y.type(torch.float64))

        if self._sigma is not None:
            mu[..., 1:] *= self._sigma

            #with torch.no_grad():
            #    nu[..., 1:] *= self._sigma

        return self._metric(mu, nu)


class ConditionalSigKerMMDDiscriminator(PathMMDDiscriminator):
    def __init__(self, kernel1_type: str, kernel2_type: str,  dyadic_order: int, path_dim: int, sigma1: float = 1.,
                 sigma2: float = 1., lambd_: float = 1., adversarial: bool = True, max_batch: int = 128):

        kernel_kwargs = {
            "kernel1_type": kernel1_type,
            "kernel2_type": kernel2_type,
            "dyadic_order": dyadic_order,
            "sigma1": sigma1,
            "sigma2": sigma2
        }

        self.max_batch = max_batch

        super().__init__(kernel_kwargs, path_dim, adversarial)

        self._kernel1 = self._init_kernel(1)
        self._kernel2 = self._init_kernel(2)
        self._metric = self._init_metric()

        self.lambd_ = lambd_

    def _init_kernel(self, kernel_number=None) -> sigkernel.SigKernel:
        """
        Inits kernel for SigKerMMDDiscriminator object, using the SigKer package. Adapted for multiple kernels
        :return:
        """

        if kernel_number is not None:
            kernel_kwargs = {
                "kernel_type": self.kernel_kwargs.get(f"kernel{kernel_number}_type"),
                "dyadic_order": self.kernel_kwargs.get("dyadic_order"),
                "sigma": self.kernel_kwargs.get(f"sigma{kernel_number}")
            }
        else:
            kernel_kwargs = self.kernel_kwargs

        return initialise_signature_kernel(**kernel_kwargs)

    def _init_metric(self):
        """
        Initialises the MMD calculation for the Signature Kernel MMD Discriminator
        :return:
        """

        def cmmd(x, y, z):
            max_batch = self.max_batch
            device    = x.device
            N         = x.size(0)

            K      = self._kernel1.compute_Gram(x, x, sym=True, max_batch=max_batch)
            L_gen  = self._kernel2.compute_Gram(y, y, sym=True, max_batch=max_batch)
            L_true = self._kernel2.compute_Gram(z, z, sym=True, max_batch=max_batch)

            L_mix = self._kernel2.compute_Gram(y, z, sym=False, max_batch=max_batch)

            Ktildeinv = torch.inverse(K + torch.eye(N, device=device) * self.lambd_)

            L_1 = torch.matmul(torch.matmul(K, Ktildeinv), torch.matmul(L_gen, Ktildeinv))
            L_2 = torch.matmul(torch.matmul(K, Ktildeinv), torch.matmul(L_true, Ktildeinv))
            L_3 = torch.matmul(torch.matmul(K, Ktildeinv), torch.matmul(L_mix, Ktildeinv))

            return torch.trace(L_1) + torch.trace(L_2) - 2*torch.trace(L_3)

        return cmmd

    def forward(self, x, y, z):
        return self._metric(x, y, z)


class TruncatedDiscriminator(PathMMDDiscriminator):
    def __init__(self, order: int, scalar_term: bool, path_dim: int, adversarial: bool, **kwargs):
        """
        MMD discriminator that uses the truncated signature kernel MMD instead of the PDE-solved one.

        :param order:           Order of signature to truncate to.
        :param scalar_term:     Whether to take the leading 1 or not in the signature calculation.
        :param path_dim:        Dimension of the output path
        :param adversarial:     Whether to train adversarially or not
        """
        kernel_kwargs = {"order": order, "scalar_term": scalar_term}
        super().__init__(kernel_kwargs, path_dim, adversarial)
        self._kernel = self._init_kernel()
        self._metric = self._init_metric()

    def _init_kernel(self):
        order       = self.kernel_kwargs.get("order")
        scalar_term = self.kernel_kwargs.get("scalar_term")

        def kernel(x, y):
            x = x.type(torch.float64)
            y = y.type(torch.float64)

            sx = signatory.signature(x, order, scalar_term = scalar_term).double()
            sy = signatory.signature(y, order, scalar_term=scalar_term).double()

            return torch.einsum("ip,jp->ij", sx, sy)

        return kernel

    def _init_metric(self):

        def _mmd(x, y):

            kxx = self._kernel(x, x)
            kxy = self._kernel(x, y)
            kyy = self._kernel(y, y)

            return torch.mean(kxx) - 2 * torch.mean(kxy) + torch.mean(kyy)

        return _mmd

class CDEDiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class CDEDiscriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers, **kwargs):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func    = CDEDiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()

def rbf_kernel_matrix(X, Y, sigma):
    """
    Compute the RBF kernel matrix between two sets of samples X and Y.
    
    Args:
    - X (torch.Tensor): Samples from the first distribution, shape (n_samples_X, n_features).
    - Y (torch.Tensor): Samples from the second distribution, shape (n_samples_Y, n_features).
    - sigma (float): The bandwidth parameter for the RBF kernel.
    
    Returns:
    - torch.Tensor: The RBF kernel matrix, shape (n_samples_X, n_samples_Y).
    """
    XX = torch.sum(X ** 2, dim=1, keepdim=True)
    YY = torch.sum(Y ** 2, dim=1, keepdim=True)
    distances = XX + YY.T - 2 * torch.mm(X, Y.T)
    kernel_matrix = torch.exp(-distances / (2 * sigma ** 2))
    return kernel_matrix

def expected_rbf_kernel(X, Y, sigma):
    """
    Compute the expected RBF kernel E[k(x, y)] where x and y are samples
    drawn from distributions represented by X and Y respectively.
    
    Args:
    - X (torch.Tensor): Samples from the first distribution, shape (n_samples_X, n_features).
    - Y (torch.Tensor): Samples from the second distribution, shape (n_samples_Y, n_features).
    - sigma (float): The bandwidth parameter for the RBF kernel.
    
    Returns:
    - float: The expected RBF kernel value.
    """
    kernel_matrix = rbf_kernel_matrix(X, Y, sigma)
    expected_value = kernel_matrix.mean()
    return expected_value

def rbf_kernel_matrix_batched(X, Y, sigma, num_random_sigmas=10):
    """
    Compute the RBF kernel matrix between two sets of samples X and Y in batches.
    
    Args:
    - X (torch.Tensor): Samples from the first distribution, shape (batch_size, n_samples_X, n_features).
    - Y (torch.Tensor): Samples from the second distribution, shape (batch_size, n_samples_Y, n_features).
    - sigma (float): The bandwidth parameter for the RBF kernel.
    
    Returns:
    - torch.Tensor: The RBF kernel matrix, shape (batch_size, n_samples_X, n_samples_Y).
    """
    XX = torch.sum(X ** 2, dim=2, keepdim=True)  # Shape: (batch_size, n_samples_X, 1)
    YY = torch.sum(Y ** 2, dim=2, keepdim=True)  # Shape: (batch_size, n_samples_Y, 1)
    distances = XX + YY.transpose(1, 2) - 2 * torch.bmm(X, Y.transpose(1, 2))  # Shape: (batch_size, n_samples_X, n_samples_Y)
    if (sigma == 'random') or isinstance(sigma, list):
        # Generate random positive real numbers for sigma
        if sigma == 'random':
            sigmas = torch.exp(torch.abs(torch.randn(num_random_sigmas)))  # Shape: (num_random_sigmas,)
        else:
            sigmas = torch.tensor(sigma)
        sigmas = sigmas.to(X.device).view(-1, 1, 1, 1)  # Shape: (num_random_sigmas, 1, 1, 1)
        
        # Compute the kernel matrices for all random sigmas
        kernel_matrices = torch.exp(-distances.unsqueeze(0) / (2 * sigmas ** 2))  # Shape: (num_random_sigmas, batch_size, n_samples_X, n_samples_Y)
        
        # Compute the weighted average of the kernel matrices
        kernel_matrix = kernel_matrices.mean(dim=0)  # Shape: (batch_size, n_samples_X, n_samples_Y)
    else:
        kernel_matrix = torch.exp(-distances / (2 * sigma ** 2))
    return kernel_matrix

def expected_rbf_kernel_batched(X, Y, sigma, XisY=False):
    """
    Compute the expected RBF kernel E[k(x, y)] where x and y are samples
    drawn from distributions represented by X and Y respectively in batches.
    
    Args:
    - X (torch.Tensor): Samples from the first distribution, shape (batch_size, n_samples_X, n_features).
    - Y (torch.Tensor): Samples from the second distribution, shape (batch_size, n_samples_Y, n_features).
    - sigma (float): The bandwidth parameter for the RBF kernel.
    
    Returns:
    - torch.Tensor: The expected RBF kernel value for each batch, shape (batch_size,).
    """
    kernel_matrix = rbf_kernel_matrix_batched(X, Y, sigma)
    if XisY:
        # Create a mask to ignore the diagonal elements
        batch_size, n_samples, _ = X.shape
        mask = ~torch.eye(n_samples, dtype=bool).unsqueeze(0).expand(batch_size, -1, -1).to(X.device)
        kernel_matrix = kernel_matrix.masked_select(mask).view(batch_size, -1)
        expected_value = kernel_matrix.mean(dim=1)  # Mean over the non-diagonal elements
    else:
        expected_value = kernel_matrix.mean(dim=(1, 2))  # Mean over the sample dimensions
    return expected_value

class RFDDiscriminator(torch.nn.Module):
    def __init__(self, n_dims="random", sigma=1.0, n_rand_dims=32, **kwargs):
        super().__init__()
        self._n_dims = n_dims
        self.sigma = sigma
        self.n_rand_dims = n_rand_dims

    def forward(self, x, y):
        """
        Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
        also have some initial point penalty.

        :param x:   Path data, shape (batch, stream, channel). Must require grad for training
        :param y:   Path data, shape (batch, stream, channel). Should not require grad
        :return:    Mixture MMD + initial point loss
        """
        T = x.shape[1]
        Nx = x.shape[0]
        Ny = y.shape[0]

        if self._n_dims == "random":
            selected_marginals = torch.multinomial(torch.ones(T), random.randint(1, self.n_rand_dims-1), replacement=False)
        else:
            selected_marginals = torch.multinomial(torch.ones(T), self._n_dims, replacement=False)
        x_fidi = x[:, selected_marginals, :].reshape((Nx, -1))
        y_fidi = y[:, selected_marginals, :].reshape((Ny, -1))

        # Split x_fidi into two random sets
        indices = torch.randperm(Nx)
        x1 = x_fidi[indices[:Nx // 2]]
        x2 = x_fidi[indices[Nx // 2:]]
        
        # Compute E[k(x1, x2)]
        E_k_x1_x2 = expected_rbf_kernel(x1, x2, self.sigma)
        
        # Compute E[k(x, y)]
        E_k_x_y = expected_rbf_kernel(x_fidi, y_fidi, self.sigma)
        return E_k_x1_x2 - 2* E_k_x_y

def piecewise_linear_function(t, x, t_new):
    assert torch.all(t[:-1] <= t[1:]), "t must be sorted in ascending order"

    indices = torch.searchsorted(t, t_new, right=True)
    indices = torch.clamp(indices, 1, len(t) - 1)

    t1 = t[indices - 1]
    t2 = t[indices]
    t1 = t1.unsqueeze(0).expand(x.size(0), -1)
    t2 = t2.unsqueeze(0).expand(x.size(0), -1)
    t_new_expanded = t_new.unsqueeze(0).expand(x.size(0), -1)
    
    x1 = x[:, indices - 1]
    x2 = x[:, indices]
    x_new = x1 + (x2 - x1) * ((t_new_expanded - t1) / (t2 - t1)).unsqueeze(-1)
    return x_new

def fidi_stochastic_loss(ts, real_samples, generator, discriminator, transformer, subtract_start):
    '''
    Request values from randomly selected time
    '''
    batch_size = real_samples.shape[0]
    rand_ts = torch.rand(ts.shape[0], device=ts.device) * ts[-1].item()
    rand_ts[0] = 0
    rand_ts[-1] = ts[-1]
    sorted_indices = torch.argsort(rand_ts)
    rand_ts = rand_ts[sorted_indices]
    generated_samples = generator(rand_ts, batch_size)
    generated_samples = transformer(generated_samples)
    if subtract_start:
        real_samples      = subtract_initial_point(real_samples)
        generated_samples = subtract_initial_point(generated_samples)
    real_samples_interp = piecewise_linear_function(ts, real_samples.detach(), rand_ts)
    return discriminator(generated_samples, real_samples_interp.detach()), piecewise_linear_function(rand_ts, generated_samples, ts)

class FDDiscriminator(torch.nn.Module):
    def __init__(self, sigma=1.0, **kwargs):
        super().__init__()
        self.sigma = sigma

    def forward(self, x, y):
        """
        Forward method for pathwise MMD discriminators, which apply a scaling as the adversarial component. They
        also have some initial point penalty.

        :param x:   Path data, shape (batch, stream, channel). Must require grad for training
        :param y:   Path data, shape (batch, stream, channel). Should not require grad
        :return:    Mixture MMD + initial point loss
        """
        T = x.shape[1]
        Nx = x.shape[0]
        Ny = y.shape[0]

        x_fds = torch.cat((x[:, :-1, :], x[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)
        y_fds = torch.cat((y[:, :-1, :], y[:, 1:, :]), axis=1) # (batch, stream-1, 2*channel)

        x_fds = torch.transpose(x_fds, 0, 1) # (stream-1, batch, 2*channel)
        y_fds = torch.transpose(y_fds, 0, 1) # (stream-1, batch, 2*channel)
        
        # Compute E[k(x1, x2)]
        #E_k_x1_x2 = expected_rbf_kernel_batched(x1, x2, self.sigma).mean()
        E_k_x1_x2 = expected_rbf_kernel_batched(x_fds, x_fds, self.sigma, True).mean()

        # Compute E[k(x, y)]
        E_k_x_y = expected_rbf_kernel_batched(x_fds, y_fds, self.sigma).mean()
        return E_k_x1_x2 - 2* E_k_x_y
