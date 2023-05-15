import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def vec_X(X_list):
    """a func to generate X vector"""
    return sp.Matrix([X_list]).reshape(len(X_list), 1)


def vis_func(func, X_list_2d, x0a=-3, x0b=3, x1a=-2, x1b=2, points=500, levels=10):
    """implement a function to do the vis"""
    func_cal = sp.lambdify(X_list_2d, func)
    
    x0s = np.linspace(x0a, x0b, points)
    x1s = np.linspace(x1a, x1b, points)
    x0, x1 = np.meshgrid(x0s, x1s)
    Xs = np.c_[x0.ravel(), x1.ravel()]
    Ys = np.array([func_cal(Xi[0], Xi[1]) for Xi in Xs], dtype=float).reshape(x1.shape)

    levelsY = np.linspace(np.min(Ys), np.max(Ys), levels)

    plt.figure(figsize=(6, 4))
    plt.title(f"${sp.latex(func)}$", fontsize=16)
    plt.xlabel("$x_0$", fontsize=16)
    plt.ylabel("$x_1$", fontsize=16)

    plt.axis([x0a, x0b, x1a, x1b])
    plt.grid(True)
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.contourf(x0, x1, Ys, levels=levelsY)

    # plt.show() does not put here for expanding the plot code if you need 


def is_pdm(matrix):
    """a function which can tell if a matrix is positive definite"""
    if matrix.shape[0] != matrix.shape[1]:
        return "This matrix is not a quare matrix"
    
    for i in range(1, matrix.shape[0]+1):
        # principal subdeterminant
        psd = matrix[:i, :i].det()
        if psd <= 0:
            return False
    return True


def partials(X_list, func):
    """calculate a func's all first-order partial derivatives"""
    pars = []
    for x in X_list:
        pars.append(sp.diff(func, x))

    return pars


def dirc_derv(func, X_list, start, end, points):
    """calculate directional deribative"""
    d = len(X_list)
    delta_p = (end - start) / 10**5
    delta_p_norm = np.linalg.norm(delta_p)

    pars = partials(X_list, func)
    par_cals = [sp.lambdify(X_list, par) for par in pars]
    
    xis = [np.linspace(start[i], end[i], points) for i in range(d)]
    Xs = np.array(list(zip(*xis)))

    Ys = []
    for Xi in Xs: 
        Y = sum([par(*Xi) * (delta_p[i] / delta_p_norm) for par, i in zip(par_cals, range(d))])
        Ys.append(Y)
    Ys = np.array(Ys, dtype=float)


def gdt(func, X_list, X_value=None):
    """calculate gradient"""
    d = len(X_list)

    gdt_exp = partials(X_list, func)
    
    if X_value:
        gdt_val = []
        for gdt_e in gdt_exp:
            gdt_v = sp.lambdify(X_list, gdt_e)(*X_value)
            gdt_val.append(gdt_v)
        return sp.Matrix(gdt_val).reshape(d, 1)
    
    return sp.Matrix([gdt_exp]).reshape(d, 1)


def hess(func, X_list, X_value=None):
    """calculate heesian matrix"""
    d = len(X_list)

    # first-order derivative --- gradient vector
    gdt_exp = partials(X_list, func)
    # second-order derivative --- hessian matrix
    hess_exp = []
    for gdt_e in gdt_exp:
        hess_exp.append(partials(X_list, gdt_e))

    if X_value:
        hess_val = []
        for hess_e in hess_exp:
            hess_v = [sp.lambdify(X_list, hess_)(*X_value) for hess_ in hess_e]            
            hess_val.append(hess_v)
        return sp.Matrix(hess_val).reshape(d, d)
    
    return sp.Matrix(hess_exp).reshape(d, d)


def is_cvx(func, X_list):
    """test if a function is convex,
    ONLY for quadratic form"""
    d = len(X_list)
    
    hess_exp = hess(func, X_list)
    hess_val = hess(func, X_list, [0]*d)

    if hess_exp != hess_val:
        return "This func is not quadratic form!"
    
    return is_pdm(hess_exp)


def is_opt_unc(func, X_list, X_value, tol=10e-5):
    """optimal condition check for unconstrained problems
    check if X-value is a local minimum point
    for convex functions, local minimum point is their globle minmum point
    tol is the tolerance of this problem"""
    gdt_v = np.array(gdt(func, X_list, X_value), dtype=float)
    hess_v = hess(func, X_list, X_value)
    if np.linalg.norm(gdt_v) < tol and is_pdm(hess_v):
        return True
    return False