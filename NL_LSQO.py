#!/usr/bin/env python3
"""
Nonlinear least-squares inversion for Young's moduli estimation in multilayered hyperelastic cylinders.
Uses SciPy's least_squares to fit E1, E2, E3 directly to force-indentation data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import math

def load_data(path):
    df = pd.read_csv(path)
    df['ind'] = pd.to_numeric(df['ind'], errors='coerce') * 1e-3  # mm -> m
    df['force'] = pd.to_numeric(df['force'], errors='coerce') * 1e-3  # mN -> N
    return df['ind'].values, df['force'].values

def stiffness_calc(E1, E2, E3, delta, stif_params):
    """
    delta : array of indentation depths (m)
    stif_params: tuple (EInd, nu, nuInd, R, Rworm, h1, h2, h3)
    returns Fpred array
    """
    EInd, nu, nuInd, R, Rworm, h1, h2, h3 = stif_params
    pi, L = math.pi, 1.0
    R1, r1 = Rworm, Rworm - h1
    R2, r2 = r1, r1 - h2
    R3 = h3 / 2
    I1 = pi * (R1**4 - r1**4) / 4
    I2 = pi * (R2**4 - r2**4) / 4
    I3 = pi * (R3**4) / 4
    maxX = np.max(delta)
    eXp, c1 = (1.5, 5.9) if maxX > 0.43 else (2.0, 9.3)
    kind = 2 * pi * EInd * R
    k1 = E1 * I1 / L**4
    k2 = E2 * I2 / L**4
    k3 = E3 * I3 / L**4
    ta = np.array([h1*1000, h2*1000, h3*1000])
    uy = maxX / 2
    khy = (uy + ta)*(2*uy**2 + 2*ta*uy + ta**2)/(2*uy + ta)**3
    hy1, hy2, hy3 = khy if maxX >= 0.7 else np.array([1,1,1])
    kstruct = c1 * (hy1*k1 + hy2*k2 + hy3*k3) * 1000
    k_eff = 1.0/(1.0/kind + 1.0/kstruct)
    return k_eff * delta**eXp

def residuals(params, delta, force_obs, stif_params, E_min_max, ordering_penalty=0.0):
    E1, E2, E3 = params
    # main data residuals
    res_data = stiffness_calc(E1, E2, E3, delta, stif_params) - force_obs

    # ordering hinge-penalty as 1-d array of length 1
    ord_val = ordering_penalty * max(E3 - E2, 0.0)
    res_order = np.atleast_1d(ord_val)  # now shape (1,)

    # soft bound-violation penalties
    lower, upper = zip(*E_min_max)
    lb_violation = np.maximum(np.array(lower) - params, 0.0)  # shape (3,)
    ub_violation = np.maximum(params - np.array(upper), 0.0)  # shape (3,)
    res_lb = 1e3 * lb_violation
    res_ub = 1e3 * ub_violation

    # concatenate everything into one 1-D residual vector
    return np.concatenate([res_data, res_order, res_lb, res_ub])

if __name__ == '__main__':
    # user inputs
    data_path = 'data.csv'
    E_min_max = [(30,250), (300,1200), (30,300)]
    ordering_penalty = 1e-2

    # load data
    delta, force_obs = load_data(data_path)

    # stiffness params (same as PGNN generate_data dummy)
    EInd, nu, nuInd, R, Rworm = 210e6, 0.33, 0.33, 5e-3, 28e-3
    h1, h2, h3 = 0.6e-3, 1.7e-3, 52.6e-3
    stif_params = (EInd, nu, nuInd, R, Rworm, h1, h2, h3)

    # initial guess: midpoints
    x0 = np.array([(lo+hi)/2 for lo,hi in E_min_max])

    # bounds for least_squares
    lb, ub = zip(*E_min_max)

    # run solver
    result = least_squares(
        residuals, x0,
        args=(delta, force_obs, stif_params, E_min_max, ordering_penalty),
        bounds=(lb, ub)
    )
    Eopt = result.x
    print(f"Optimized E1, E2, E3 = {Eopt}")

    # plot fit
    Fpred = stiffness_calc(Eopt[0], Eopt[1], Eopt[2], delta, stif_params)
    plt.figure()
    plt.plot(delta, force_obs, 'bo', label='meas')
    plt.plot(delta, Fpred, 'r-', label='fit')
    plt.legend()
    plt.show()

    # save results
    df_out = pd.DataFrame({
        'ind_mm': delta*1e3,
        'force_N': force_obs,
        'fit_N': Fpred
    })
    name = "_".join(str(int(round(e))) for e in Eopt)
    df_out.to_csv(f'inversion_ls_E{name}.csv', index=False)
