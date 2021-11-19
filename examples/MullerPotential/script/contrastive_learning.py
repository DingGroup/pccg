__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2021/01/09 19:34:07"

import numpy as np
from functions import *
from sys import exit
import argparse
from scipy.interpolate import BSpline
from scipy import optimize
import matplotlib as mpl
from matplotlib import cm
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/CLCG")
from utils.functions import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--alpha", type = float)

args = argparser.parse_args()
alpha = args.alpha

with open("./output/range.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
x1_min, x1_max = data['x1_min'], data['x1_max']
x2_min, x2_max = data['x2_min'], data['x2_max']

num_samples = 30    
x1 = np.random.rand(30)*(x1_max - x1_min) + x1_min
x2 = np.random.rand(30)*(x2_max - x2_min) + x2_min
x = np.vstack([x1, x2]).T

y = compute_cubic_spline_basis(x)

## samples from p
with open("./output/TREMC/x_record_alpha_{:.3f}.pkl".format(alpha), 'rb') as file_handle:
    data = pickle.load(file_handle)
xp = data['x_record'][:, -1, :]
num_samples_p = xp.shape[0]

## samples from q
num_samples_q = num_samples_p
x1_q = np.random.rand(num_samples_q)*(x1_max - x1_min) + x1_min
x2_q = np.random.rand(num_samples_q)*(x2_max - x2_min) + x2_min
xq = np.vstack([x1_q, x2_q]).T

x1_knots = np.linspace(x1_min, x1_max, num = 10, endpoint = False)[1:]
x2_knots = np.linspace(x2_min, x2_max, num = 10, endpoint = False)[1:]

x1_boundary_knots = np.array([x1_min, x1_max])
x2_boundary_knots = np.array([x2_min, x2_max])

def compute_design_matrix(x, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots):
    x1_design_matrix = bs(x[:,0], x1_knots, x1_boundary_knots)
    x2_design_matrix = bs(x[:,1], x2_knots, x2_boundary_knots)
    x_design_matrix = x1_design_matrix[:,:,np.newaxis] * x2_design_matrix[:,np.newaxis,:]
    x_design_matrix = x_design_matrix.reshape([x_design_matrix.shape[0], -1])
    return x_design_matrix

xp_design_matrix = compute_design_matrix(xp, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
xq_design_matrix = compute_design_matrix(xq, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)

# x1_p_design_matrix = bs(xp[:,0], x1_knots, x1_boundary_knots)
# x2_p_design_matrix = bs(xp[:,1], x2_knots, x2_boundary_knots)
# xp_design_matrix = x1_p_design_matrix[:,:,np.newaxis] * x2_p_design_matrix[:,np.newaxis,:]
# xp_design_matrix = xp_design_matrix.reshape([xp_design_matrix.shape[0], -1])

# x1_q_design_matrix = bs(xq[:,0], x1_knots, x1_boundary_knots)
# x2_q_design_matrix = bs(xq[:,1], x2_knots, x2_boundary_knots)
# xq_design_matrix = x1_q_design_matrix[:,:,np.newaxis] * x2_q_design_matrix[:,np.newaxis,:]
# xq_design_matrix = xq_design_matrix.reshape([xq_design_matrix.shape[0], -1])

## coefficients of cubic splines
theta = np.random.randn(xp_design_matrix.shape[-1])
F = np.zeros(1)

def compute_loss_and_grad(thetas):
    theta = thetas[0:xp_design_matrix.shape[-1]]
    F = thetas[-1]

    up_xp = np.matmul(xp_design_matrix, theta)
    logp_xp = -(up_xp - F)
    logq_xp = np.ones_like(logp_xp)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    up_xq = np.matmul(xq_design_matrix, theta)
    logp_xq = -(up_xq - F)
    logq_xq = np.ones_like(logp_xq)*np.log(1/((x1_max - x1_min)*(x2_max - x2_min)))

    nu = num_samples_q / num_samples_p
    
    G_xp = logp_xp - logq_xp
    G_xq = logp_xq - logq_xq

    h_xp = 1./(1. + nu*np.exp(-G_xp))
    h_xq = 1./(1. + nu*np.exp(-G_xq))

    loss = -(np.mean(np.log(h_xp)) + nu*np.mean(np.log(1-h_xq)))

    dl_dtheta = -(np.mean((1 - h_xp)[:, np.newaxis]*(-xp_design_matrix), 0) +
                  nu*np.mean(-h_xq[:, np.newaxis]*(-xq_design_matrix), 0))
    dl_dF = -(np.mean(1 - h_xp) + nu*np.mean(-h_xq))

    return loss, np.concatenate([dl_dtheta, np.array([dl_dF])])

thetas_init = np.concatenate([theta, F])
loss, grad = compute_loss_and_grad(thetas_init)

thetas, f, d = optimize.fmin_l_bfgs_b(compute_loss_and_grad,
                                 thetas_init,
                                 iprint = 1)
#                                 factr = 10)
theta = thetas[0:xp_design_matrix.shape[-1]]
F = thetas[-1]

x_grid = generate_grid(x1_min, x1_max, x2_min, x2_max, size = 100)
x_grid_design_matrix = compute_design_matrix(x_grid, x1_knots, x2_knots, x1_boundary_knots, x2_boundary_knots)
up = np.matmul(x_grid_design_matrix, theta)
up = up.reshape(100, 100)
up = up.T

fig, axes = plt.subplots()
plt.contourf(up, levels = 30, extent = (x1_min, x1_max, x2_min, x2_max), cmap = cm.viridis_r)
plt.xlabel(r"$x_1$", fontsize = 24)
plt.ylabel(r"$x_2$", fontsize = 24)
plt.tick_params(which='both', bottom=False, top=False, right = False, left = False, labelbottom=False, labelleft=False)
plt.colorbar()
plt.tight_layout()
axes.set_aspect('equal')
plt.savefig("./output/learned_Up_alpha_{:.3f}.eps".format(alpha))

exit()
