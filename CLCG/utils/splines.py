from scipy.interpolate import BSpline
import numpy as np
import math
import matplotlib.pyplot as plt

def bs(x, knots, boundary_knots, degree = 3, intercept = False):
    """
    Generate the B-spline basis matrix for a polynomial spline.
    This function mimick the function bs in R package splines

    Parameters
    ----------
    x : ndarray
        The sequence of values at which basis functions are evaludated
    knots: ndarray
        the internal breakpoints that define the spline.
    boundary_knots: ndarray
        Boundary points at which to anchor the B-spline basis
    degree: int, optional
        The degree of the piecewise polynomial. The default is '3' for cubic splines.
    intercept: bool, optional
        If True, an intercept is included in the basis; default is False.
    
    Returns:
    --------
    design_matrix: ndarray
        A matrix of dimension (len(x), df), where df = len(knots) + degree if intercept
        = False, df = len(knots) + degree + 1 if intercept = True.
    """
    
    knots = np.concatenate([knots, boundary_knots])
    knots.sort()

    augmented_knots = np.concatenate([np.array([boundary_knots[0] for i in range(degree + 1)]),
                                      knots,
                                      np.array([boundary_knots[1] for i in range(degree + 1)])])
    num_of_basis = len(augmented_knots) - 2*(degree + 1) + degree + 1

    spl_list = []
    for i in range(num_of_basis):
        coeff = np.zeros(num_of_basis)
        coeff[i] = 1.0
        spl = BSpline(augmented_knots, coeff, degree, extrapolate = False)
        spl_list.append(spl)

    design_matrix = np.array([spl(x) for spl in spl_list]).T

    ## if the intercept is Fales, drop the first basis term, which is often
    ## referred as the "intercept". Note that np.sum(design_matrix, -1) = 1.
    ## see https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
    if intercept is False:
        design_matrix = design_matrix[:, 1:]
        
    return design_matrix
    
def pbs(x, knots, boundary_knots = np.array([-math.pi, math.pi]), degree = 3, intercept = False):
    """
    Compute the design matrix of a periodic B-spline. 
    This function mimick the pbs function in R package pbs.

    Parameters
    ----------
    x : ndarray
        The sequence of values at which basis functions are evaludated
    knots: ndarray
        the internal breakpoints that define the spline.
    boundary_knots: ndarray
        Boundary points at which to anchor the B-spline basis
    degree: int, optional
        The degree of the piecewise polynomial. The default is '3' for cubic splines.
    intercept: bool, optional
        If True, an intercept is included in the basis; default is False.
    
    Returns:
    --------
    design_matrix: ndarray
        A matrix of dimension (len(x), df), where df = len(knots) if intercept
        = False, df = len(knots) + 1 if intercept = True
    """
    
    knots = np.concatenate([knots, boundary_knots])
    knots.sort()

    augmented_knots = np.copy(knots)
    for i in range(degree):
        augmented_knots = np.append(augmented_knots, knots[-1] + knots[i+1] - knots[0])
    for i in range(degree):
        augmented_knots = np.insert(augmented_knots, 0, knots[0] - (knots[-1] - knots[-1-(i+1)]))

    num_of_basis = len(augmented_knots) - 2*(degree + 1) + degree + 1

    spl_list = []
    for i in range(num_of_basis):
        coeff = np.zeros(num_of_basis)
        coeff[i] = 1.0
        spl = BSpline(augmented_knots, coeff, degree, extrapolate = False)
        spl_list.append(spl)

    design_matrix = np.array([spl(x) for spl in spl_list]).T
    design_matrix_left = design_matrix[:, 0:degree]
    design_matrix_right = design_matrix[:, -degree:]
    design_matrix_middle = design_matrix[:, degree:-degree]
    design_matrix = np.concatenate([design_matrix_middle, design_matrix_left + design_matrix_right], axis = -1)

    ## if the intercept is Fales, drop the first basis term, which is often
    ## referred as the "intercept".
    ## see https://cran.r-project.org/web/packages/crs/vignettes/spline_primer.pdf
    if intercept is False:
        design_matrix = design_matrix[:, 1:]
        
    return design_matrix

if __name__ == "__main__":
    knots = np.linspace(start = -math.pi, stop = math.pi, num = 10)
    knots = knots[1:-1]
    boundary_knots = np.array([-math.pi, math.pi])
    
    x = np.linspace(start = -math.pi, stop = math.pi, num = 200)
    degree = 3

    design_matrix_bs = bs(x, knots, boundary_knots, degree)
    design_matrix_pbs = pbs(x, knots, boundary_knots, degree)
    
    fig, axes = plt.subplots()
    for j in range(design_matrix_bs.shape[-1]):
        plt.plot(x, design_matrix_bs[:,j], label = f"{j}")
    plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_bs.pdf")
    
    fig, axes = plt.subplots()
    for j in range(design_matrix_pbs.shape[-1]):
        plt.plot(x, design_matrix_pbs[:,j], label = f"{j}")
    plt.legend()
    plt.tight_layout()
    fig.savefig("./output/design_matrix_pbs.pdf")
    
