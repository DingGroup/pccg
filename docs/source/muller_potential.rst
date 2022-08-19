.. _muller_potential:

Learning the Müller potential
=============================

Here we use the Müller potential and samples from it to show how to use potential contrasting to learn a potential energy function that can reproduce a distribution of samples.

The Müller potential is defined as

.. math::

   U(x_1, x_2) = s \cdot \sum_{k = 1}^{4} A_k \cdot \exp\left( \alpha_k (x_1 - a_k)^2 + \beta_k (x_1 - a_k)(x_2 - b_k) + \gamma_k (x_2 - b_k)^2 \right),

where :math:`(A_1, A_2, A_3, A_4) = (-200, -100, -170, 15)`, :math:`(\alpha_1, \alpha_2, \alpha_3, \alpha_4) = (-1, -1, -6.5, 0.7)`, :math:`(\beta_1, \beta_2, \beta_3, \beta_4) = (0, 0, 11, 0.6)`, :math:`(\gamma_1, \gamma_2, \gamma_3, \gamma_4) = (-10, -10, -6.5, 0.7)`, :math:`(a_1, a_2, a_3, a_4) =  (1, 0, -0.5, -1)`, and :math:`(b_1, b_2, b_3, b_4) = (0, 0.5, 1.5, 1)`.
:math:`s` is a scaling parameter and is set to :math:`0.05` in this tutorial.

Now let us define the Müller potential in a ``Python`` function.

.. code-block:: python

   import torch
   
   def compute_Muller_potential(alpha, x):
       A = (-200., -100., -170., 15.)
       beta = (0., 0., 11., 0.6)    
       alpha_gamma = (
           x.new_tensor([-1.0, -10.0]),
           x.new_tensor([-1.0, -10.0]),
           x.new_tensor([-6.5,  -6.5]),
           x.new_tensor([ 0.7,   0.7])
       )
       
       ab = (
           x.new_tensor([ 1.0, 0.0]),
           x.new_tensor([ 0.0, 0.5]),
           x.new_tensor([-0.5, 1.5]),
           x.new_tensor([-1.0, 1.0])
       )
       
       U = 0    
       for i in range(4):
           diff = x - ab[i]
           U = U + A[i]*torch.exp(
	       torch.sum(alpha_gamma[i]*diff**2, -1) + beta[i]*torch.prod(diff, -1)
	       )
	       
       U = alpha * U
       return U

Because the potential function is defined over a two dimensional space, we can
visualize it using a two dimensional heatmap.

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib import cm   
   import numpy as np
   
   def generate_grid(x1_min, x1_max, x2_min, x2_max, size=100):
       x1 = torch.linspace(x1_min, x1_max, size)
       x2 = torch.linspace(x2_min, x2_max, size)
       grid_x1, grid_x2 = torch.meshgrid(x1, x2)
       grid = torch.stack([grid_x1, grid_x2], dim=-1)
       x = grid.reshape((-1, 2))
       return x
		
   x1_min, x1_max = -1.5, 1.0
   x2_min, x2_max = -0.5, 2.0

   x = generate_grid(x1_min, x1_max, x2_min, x2_max)
   fig, axes = plt.subplots()
   alpha = 0.05
   U = compute_Muller_potential(alpha, x)
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
   plt.show()   
   

       
