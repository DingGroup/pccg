import torch
import numpy as np
from scipy import optimize
from collections import namedtuple

def NCE(log_q_noise, log_q_data,
        basis_noise, basis_data,
        verbose = True,
):

    """
    Noise contrastive estimation

    Args:
        log_q_noise (Tensor):  1-D vector, the logrithm of probability density for noise data under the noise distribution
        log_q_data (Tensor): 1-D vector, the logrithm of probability density for target data under the noise distribution
        basis_noise (Tensor): 2-D vector, the design matrix contraining basis values of noise data for compute the logrithm of probablity density for the target distribution
        basis_data (Tensor): 2-D vector, the design matrix contraining basis values of target data for compute the logrithm of probablity density for the target distribution
        verbose (bool, default = ``True``): whether to print the optimization information

    Returns:
        A namedtuple (``theta``, ``dF``) where ``theta`` is a vector of learned 
        basis coefficients and ``dF`` is the free energy difference between
        the ensemble defined by the target energy function and that defined
        by the noise energy function. 
    """

    assert basis_noise.shape[-1] == basis_data.shape[-1]
    assert len(log_q_noise) == basis_noise.shape[0]
    assert len(log_q_data) == basis_data.shape[0]

    basis_size = basis_noise.shape[-1]
    theta = torch.zeros(basis_size, dtype = torch.float64)
    dF = torch.zeros(1, dtype = torch.float64)

    x_init = np.concatenate([theta.data.numpy(), dF])

    def compute_loss_and_grad(x):
        theta = torch.tensor(x[0:basis_size], requires_grad=True)
        dF = torch.tensor(x[-1], requires_grad=True)

        u_data = torch.matmul(basis_data, theta)
        u_noise = torch.matmul(basis_noise, theta)

        num_samples_p = basis_data.shape[0]
        num_samples_q = basis_noise.shape[0]

        nu = dF.new_tensor([num_samples_q / num_samples_p])

        log_p_data = -(u_data - dF) - torch.log(nu)
        log_p_noise = -(u_noise - dF) - torch.log(nu)

        log_q = torch.cat([log_q_noise, log_q_data])
        log_p = torch.cat([log_p_noise, log_p_data])

        logit = log_p - log_q
        target = torch.cat([torch.zeros_like(log_q_noise), torch.ones_like(log_q_data)])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        loss.backward()

        grad = torch.cat([theta.grad, dF.grad[None]]).numpy()

        return loss.item(), grad

    loss, grad = compute_loss_and_grad(x_init)

    options={"disp": verbose, "gtol": 1e-6}
    results = optimize.minimize(
        compute_loss_and_grad, x_init, jac=True, method="L-BFGS-B", options=options
    )
    x = results["x"]

    # x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
    #                                  x_init,
    #                                  iprint = 1,
    #                                  pgtol = 1e-6,
    #                                  factr = 100)

    theta = x[0:basis_size]
    dF = x[-1]

    nce_result = namedtuple('nce_result', ['theta', 'dF'])
    out = nce_result(torch.from_numpy(theta), dF)
    
    return out


def contrastive_learning_numpy(log_q_noise, log_q_data, basis_noise, basis_data):

    """
    Contrastive learning coefficients

    Parameters
    ----------
    log_q_noise: 1-dimensional array
        the logrithm of probability density for noise data under the noise distribution
    log_q_data: 1-dimensional array
        the logrithm of probability density for target data under the noise distribution
    basis_noise: 2-dimensional array
        the design matrix contraining basis values of noise data for compute the logrithm of probablity density for the target distribution
    basis_data: 2-dimensional array
        the design matrix contraining basis values of target data for compute the logrithm of probablity density for the target distribution

    Returns
    -------
    alpha:

    """

    assert basis_noise.shape[-1] == basis_data.shape[-1]
    assert len(log_q_noise) == basis_noise.shape[0]
    assert len(log_q_data) == basis_data.shape[0]

    basis_size = basis_noise.shape[-1]
    theta = np.zeros(basis_size)
    dF = np.zeros(1)

    x_init = np.concatenate([theta, dF])

    log_q = np.concatenate([log_q_noise, log_q_data])
    y = np.concatenate([np.zeros_like(log_q_noise), np.ones_like(log_q_data)])

    basis = np.concatenate([basis_noise, basis_data])

    num_samples_p = basis_data.shape[0]
    num_samples_q = basis_noise.shape[0]

    log_nu = np.log(num_samples_q / float(num_samples_p))

    def compute_loss_and_grad(x):
        theta = x[0:basis_size]
        dF = x[-1]

        ## compute loss = -(y*h - np.log(1 + np.exp(h)))
        h = -(np.matmul(basis, theta) - dF) - log_q - log_nu
        loss = -(y * h - (np.maximum(h, 0) + np.log(1 + np.exp(-np.abs(h)))))
        loss = np.mean(loss, 0)

        ## compute gradients
        p = 1.0 / (1 + np.exp(-np.abs(h)))
        p[h < 0] = 1 - p[h < 0]

        grad_theta = np.matmul(basis.T, y - p) / y.shape[0]
        grad_dF = -np.mean(y - p, keepdims=True)

        grad = np.concatenate([grad_theta, grad_dF])

        return loss, grad

    loss, grad = compute_loss_and_grad(x_init)
    x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad, x_init, iprint=1)

    theta = x[0:basis_size]
    dF = x[-1]

    return theta, dF
