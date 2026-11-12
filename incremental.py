"""Source-calibrated reconstruction and implicit predictor–corrector update."""
import numpy as np
from ic_solver import solve


def update(prior, old_target, new_target, variance, gamma=1.0):
    old = solve(prior, old_target, variance, gamma)
    delta = np.asarray(new_target, float)-np.asarray(old_target, float)
    hessian = old['cov'] + np.diag(np.asarray(variance, float)/gamma)
    predicted_theta = old['theta']-np.linalg.solve(hessian,delta)
    predicted_p = old['p']+old['Jp']@delta
    corrected = solve(prior, new_target, variance, gamma, start=predicted_theta)
    cold = solve(prior, new_target, variance, gamma)
    return {'previous':old,'corrected':corrected,'cold':cold,'linear_probability':predicted_p}
