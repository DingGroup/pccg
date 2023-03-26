#### define the Muller potential
import torch

def compute_Muller_potential(scale, x):
    A = (-200.0, -100.0, -170.0, 15.0)
    beta = (0.0, 0.0, 11.0, 0.6)
    alpha_gamma = (
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-1.0, -10.0]),
        x.new_tensor([-6.5, -6.5]),
        x.new_tensor([0.7, 0.7]),
    )

    ab = (
        x.new_tensor([1.0, 0.0]),
        x.new_tensor([0.0, 0.5]),
        x.new_tensor([-0.5, 1.5]),
        x.new_tensor([-1.0, 1.0]),
    )

    U = 0
    for i in range(4):
        diff = x - ab[i]
        U = U + A[i] * torch.exp(
            torch.sum(alpha_gamma[i] * diff**2, -1) + beta[i] * torch.prod(diff, -1)
        )

    U = scale * U
    return U


#### plot the Muller potential
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
    x1 = torch.linspace(x1_min, x1_max, size)
    x2 = torch.linspace(x2_min, x2_max, size)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.stack([grid_x1, grid_x2], dim=-1)
    x = grid.reshape((-1, 2))
    return x


x1_min, x1_max = -1.5, 1.0
x2_min, x2_max = -0.5, 2.0

grid_size = 100
x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, grid_size)
fig, axes = plt.subplots()
scale = 0.05
U = compute_Muller_potential(scale, x_grid)
U = U.reshape(100, 100)
U[U > 9] = 9
U = U.T
plt.contourf(
    U,
    levels=np.linspace(-9, 9, 19),
    extent=(x1_min, x1_max, x2_min, x2_max),
    cmap=cm.viridis_r,
)
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
plt.colorbar()
axes.set_aspect("equal")
plt.tight_layout()
plt.savefig("./data/mp.png")
plt.close()
# plt.show()

#### draw samples from the Müller potential

import os

if os.path.exists("./data/samples.csv"):
    x_data = np.loadtxt("./data/samples.csv", delimiter=",")
    x_data = torch.from_numpy(x_data)
    n_data = x_data.shape[0]
else:
    num_reps = 10  # number of replicas
    scales = torch.linspace(0.0, scale, num_reps)

    num_steps = 1010000
    x_record = []
    accept_rate = 0
    x = torch.stack(
        (
            x1_min + torch.rand(num_reps) * (x1_max - x1_min),
            x2_min + torch.rand(num_reps) * (x2_max - x2_min),
        ),
        dim=-1,
    )
    energy = compute_Muller_potential(1.0, x)

    for k in range(num_steps):
        if (k + 1) % 10000 == 0:
            print("steps: {} out of {} total steps".format(k, num_steps))

        ## sampling within each replica
        delta_x = torch.normal(0, 1, size=(num_reps, 2)) * 0.3
        x_p = x + delta_x
        energy_p = compute_Muller_potential(1.0, x_p)

        ## accept based on energy
        accept_prop = torch.exp(-scales * (energy_p - energy))
        accept_flag = torch.rand(num_reps) < accept_prop

        ## considering the bounding effects
        accept_flag = (
            accept_flag
            & torch.all(x_p > x_p.new_tensor([x1_min, x2_min]), -1)
            & torch.all(x_p < x_p.new_tensor([x1_max, x2_max]), -1)
        )

        x_p[~accept_flag] = x[~accept_flag]
        energy_p[~accept_flag] = energy[~accept_flag]
        x = x_p
        energy = energy_p

        ## calculate overall accept rate
        accept_rate = accept_rate + (accept_flag.float() - accept_rate) / (k + 1)

        ## exchange
        if k % 10 == 0:
            for i in range(1, num_reps):
                accept_prop = torch.exp(
                    (scales[i] - scales[i - 1]) * (energy[i] - energy[i - 1])
                )
                accept_flag = torch.rand(1) < accept_prop
                if accept_flag.item():
                    tmp = x[i].clone()
                    x[i] = x[i - 1]
                    x[i - 1] = tmp

                    tmp = energy[i].clone()
                    energy[i] = energy[i - 1]
                    energy[i - 1] = tmp

            if k >= 10000:
                x_record.append(x.clone())

    x_data = torch.stack(x_record)[:, -1, :]
    n_data = x_data.shape[0]
    np.savetxt("./data/samples.csv", x_data.numpy(), fmt="%.10e", delimiter=",")

#### plot samples
fig = plt.figure()
fig.clf()
plt.plot(x_data[::10, 0].numpy(), x_data[::10, 1].numpy(), ".", alpha=0.5)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
axes.set_aspect("equal")
plt.tight_layout()
plt.savefig("./data/mp_samples.png")
plt.close()
# plt.show()

#### define a noise distribution and generate noise samples
import math

def compute_log_q(x, x1_limits=(x1_min, x1_max), x2_limits=(x2_min, x2_max)):
    x1_min, x1_max = x1_limits
    x2_min, x2_max = x2_limits
    log_q = math.log(1 / (x1_max - x1_min) * 1 / (x2_max - x2_min))
    return torch.ones(x.shape[0]) * log_q


n_noise = n_data
x1_noise = torch.rand(n_noise) * (x1_max - x1_min) + x1_min
x2_noise = torch.rand(n_noise) * (x2_max - x2_min) + x2_min
x_noise = torch.stack((x1_noise, x2_noise), dim=1)

#### learn an energy function
import PC


def compute_2d_cubic_spline_basis(
    x, M1=10, M2=10, x1_limits=(x1_min, x1_max), x2_limits=(x2_min, x2_max)
):
    x1_min, x1_max = x1_limits
    x2_min, x2_max = x2_limits

    ## degree of spline
    k = 3

    num_knots_x1 = M1 - k - 2
    num_knots_x2 = M2 - k - 2

    ## knots of cubic spline
    knots_x1 = torch.linspace(x1_min, x1_max, num_knots_x1 + 2)[1:-1]
    knots_x2 = torch.linspace(x2_min, x2_max, num_knots_x2 + 2)[1:-1]

    boundary_knots_x1 = torch.tensor([x1_min, x1_max])
    boundary_knots_x2 = torch.tensor([x2_min, x2_max])

    basis_x1 = PC.spline.bs(x[:, 0], knots_x1, boundary_knots_x1)
    basis_x2 = PC.spline.bs(x[:, 1], knots_x2, boundary_knots_x2)

    basis = basis_x1[:, :, None] * basis_x2[:, None, :]
    basis = basis.reshape(-1, M1 * M2)
    return basis


basis_data = compute_2d_cubic_spline_basis(x_data)
basis_noise = compute_2d_cubic_spline_basis(x_noise)
log_q_data = compute_log_q(x_data)
log_q_noise = compute_log_q(x_noise)
theta, dF = PC.NCE(log_q_noise, log_q_data, basis_noise, basis_data)

#### plot the learned energy function
basis_grid = compute_2d_cubic_spline_basis(x_grid)
U_grid = torch.matmul(basis_grid, theta)
U_grid = U_grid.reshape((grid_size, grid_size))

U_grid = U_grid - U_grid.min() + U.min()
U_grid[U_grid > 9] = 9
fig, axes = plt.subplots()
plt.contourf(
    U_grid.T.numpy(),
    levels=np.linspace(-9, 9, 19),
    extent=(x1_min, x1_max, x2_min, x2_max),
    cmap=cm.viridis_r,
)
plt.xlabel(r"$x_1$", fontsize=24)
plt.ylabel(r"$x_2$", fontsize=24)
plt.colorbar()
axes.set_aspect("equal")
plt.tight_layout()
plt.savefig("./data/learned_potential.png")
plt.close()
