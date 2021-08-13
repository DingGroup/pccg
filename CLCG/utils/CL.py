import torch
import numpy as np
from scipy import optimize
import simtk.unit as unit

def contrastive_learning(log_q_noise, log_q_data,
                         basis_noise, basis_data):
    
    assert(basis_noise.shape[-1] == basis_data.shape[-1])
    assert(len(log_q_noise) == basis_noise.shape[0])
    assert(len(log_q_data) == basis_data.shape[0])

    basis_size = basis_noise.shape[-1]
    alphas = torch.randn(basis_size)
    F = torch.zeros(1)
    
    x_init = np.concatenate([alphas.data.numpy(), F])
    
    def compute_loss_and_grad(x):
        alphas = torch.tensor(x[0:basis_size], requires_grad = True)
        F = torch.tensor(x[-1], requires_grad = True)

        u_data = torch.matmul(basis_data, alphas)
        u_noise = torch.matmul(basis_noise, alphas)

        num_samples_p = basis_data.shape[0]
        num_samples_q = basis_noise.shape[0]

        nu = F.new_tensor([num_samples_q / num_samples_p])
        
        log_p_data = - (u_data - F) - torch.log(nu)
        log_p_noise = - (u_noise - F) - torch.log(nu)

        logit = torch.stack(
            [ torch.cat([log_q_noise, log_q_data]),
              torch.cat([log_p_noise, log_p_data])]
            ).t()        
        target = torch.cat([torch.zeros_like(log_q_noise), torch.ones_like(log_q_data)]).long()
        loss = torch.nn.functional.cross_entropy(logit, target)    
        loss.backward()
        
        grad = torch.cat([alphas.grad, F.grad[None]]).numpy()

        return loss.item(), grad

    loss, grad = compute_loss_and_grad(x_init)
    x, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                     x_init,
                                     iprint = 1)

    alphas = x[0:basis_size]
    F = x[-1]

    return alphas, F

    
    
