import numpy as np
from scipy.stats import linregress
from EEconstructor import eEconstructor


def fit_log_entropy_as_function_of_right_endpoint(phi_right_values, S_values, phi_left, phi_min, phi_max, a,  fit_phi_right_min=None, fit_phi_right_max=None):
    """
    Fits

        S([phi_L, phi_R]) = (c/3) log(2*Deltaphi/pi*a * sin(pi*l/L) + s_a

    for fixed phi_L and varying phi_R.

    Parameters
    ----------
    phi_right_values:
        Array of left endpoints.
    S_values:
        Entropy values corresponding to phi_right_values.
    phi_left:
        Fixed left endpoint.
    fit_phi_right_min, fit_phi_right_max:
        Optional fitting window in phi_L.

    Returns
    -------
    Dictionary containing c_fit, s_a_fit, slope, intercept, errors, and mask.
    """

    phi_right_values = np.asarray(phi_right_values)
    S_values = np.asarray(S_values)

    l_values = phi_right_values - phi_left

    mask = np.isfinite(S_values)
    mask &= l_values > 0

    if fit_phi_right_min is not None:
        mask &= phi_right_values >= fit_phi_right_min

    if fit_phi_right_max is not None:
        mask &= phi_right_values <= fit_phi_right_max

    x = np.log(2*(phi_max-phi_min)/(np.pi*a)*np.sin(np.pi * l_values[mask] / (phi_max - phi_min)))
    y = S_values[mask]

    result = linregress(x, y)

    slope = result.slope
    intercept = result.intercept

    c_fit = 3.0 * slope
    s_a_fit = intercept

    c_error = 3.0 * result.stderr
    s_a_error = result.intercept_stderr

    return {
        "c_fit": c_fit,
        "s_a_fit": s_a_fit,
        "c_error": c_error,
        "s_a_error": s_a_error,
        "slope": slope,
        "intercept": intercept,
        "r_value": result.rvalue,
        "p_value": result.pvalue,
        "mask": mask,
        "log_L_fit": x,
        "S_fit_data": y,
    }

def fit_log_entropy_as_function_of_length(
    L_values,
    S_values,
    fit_L_min=None,
    fit_L_max=None,
):
    """
    Fits

        S(L) = (c/3) log(L) + s_a

    so that

        c_fit = 3 * slope.
    """
    L_values = np.asarray(L_values)
    S_values = np.asarray(S_values)

    mask = np.isfinite(S_values)
    mask &= L_values > 0

    if fit_L_min is not None:
        mask &= L_values >= fit_L_min

    if fit_L_max is not None:
        mask &= L_values <= fit_L_max

    x = np.log(L_values[mask])
    y = S_values[mask]

    result = linregress(x, y)

    return {
        "c_fit": 3.0 * result.slope,
        "s_a_fit": result.intercept,
        "c_error": 3.0 * result.stderr,
        "s_a_error": result.intercept_stderr,
        "slope": result.slope,
        "intercept": result.intercept,
        "r_value": result.rvalue,
        "r_squared": result.rvalue**2,
        "p_value": result.pvalue,
        "mask": mask,
        "log_L_fit": x,
        "S_fit_data": y,
    }
    

if __name__ == "__main__":
    L_values, S_values = eEconstructor.entropy_as_function_of_length_with_fixed_center(
    phi_center=-40,
    L_min=1.0,
    L_max=10.0,
    mu=0.1,
    phi_min=-100,
    phi_max=10,
    a=0.1,
    b=1.0,
    )

    fit = fit_log_entropy_as_function_of_length(
    L_values,
    S_values,
    fit_L_min=2.0,
    fit_L_max=8.0,
    )

    print(fit["c_fit"])
    