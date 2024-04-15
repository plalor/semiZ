import numpy as np
from scipy.optimize import minimize
from .XCOM import mu_tot, mu_PE, mu_CS, mu_PP
from time import perf_counter as time

def calcLookupTables(phi_H, phi_L, D, E, a_H, b_H, c_H, a_L, b_L, c_L, lmbda_range, Z_range):
    """
    Creates a lookup table for computing the forward model and its derivatives.
    
    Parameters
    ----------
    phi_H : array_like
        Shape (n,) or shape (k, n) high energy photon beam spectrum.
        
    phi_L : array_like
        Shape (n,) or shape (k, n) low energy photon beam spectrum.

    D : array_like
        Shape (n,) detector response function.
        
    E : array_like
        Shape (n,) energy grid, on which `phi_H`, `phi_L`, and `D` are defined.
        
    a_H : array_like
        Shape (k,) array of photoelectric effect calibration parameters for
        the high energy beam.
        
    b_H : array_like
        Shape (k,) array of Compton scattering calibration parameters for
        the high energy beam.
        
    c_H : array_like
        Shape (k,) array of pair production calibration parameters for the
        high energy beam.
        
    a_L : array_like
        Shape (k,) array of photoelectric effect calibration parameters for
        the low energy beam.
        
    b_L : array_like
        Shape (k,) array of Compton scattering calibration parameters for
        the low energy beam.
        
    c_L : array_like
        Shape (k,) array of pair production calibration parameters for the
        low energy beam.
        
    lmbda_range : ndarray
        Shape (l,) lambda grid, for defining tables.
                
    Z_range : ndarray
        Shape (m,) Z grid, for defining tables.
        
    Returns
    -------
    
    tables : ndarray
        Shape (6, k, m, l) lookup table. tables[:, idx_k, idx_m, idx_l] gives
        alpha_H, alpha_L, alpha_H', alpha_L', alpha_H'', alpha_L'' evaluated
        at lmbda_range[idx_m], Z_range[idx_l] for detector idx_k.
    
    """
    k_bins = a_H.size
    n_bins = E.size
    m_bins = Z_range.size
    l_bins = lmbda_range.size
    
    t0 = time()
    print("Calculating attenuation matrices..", end='')
    mu_mat_H = np.zeros((k_bins, n_bins, m_bins))
    mu_mat_L = np.zeros((k_bins, n_bins, m_bins))
    for k in range(k_bins):
        for m in range(m_bins):
            Z = Z_range[m]
            tot, PE, CS, PP = mu_tot(E, Z), mu_PE(E, Z), mu_CS(E, Z), mu_PP(E, Z)
            mu_mat_H[k,:,m] = tot + (a_H[k]-1)*PE + (b_H[k]-1)*CS + (c_H[k]-1)*PP
            mu_mat_L[k,:,m] = tot + (a_L[k]-1)*PE + (b_L[k]-1)*CS + (c_L[k]-1)*PP
    
    t1 = time()
    print("completed in %.2f seconds" % (t1 - t0))
    print("Calculating lookup tables...", end='')
    tables = np.zeros((6, k_bins, m_bins, l_bins))
    if phi_H.ndim == 1:
        phi_H = np.tile(phi_H, (k_bins,1))
        phi_L = np.tile(phi_L, (k_bins,1))
    D_phi_H = D * phi_H
    D_phi_L = D * phi_L
    d_H = np.sum(D_phi_H, axis=1)
    d_L = np.sum(D_phi_L, axis=1)
    for k in range(k_bins):
        for m in range(m_bins):
            mu_H = mu_mat_H[k,:,m]
            mu_L = mu_mat_L[k,:,m]
            m0_H = np.exp(-np.outer(mu_H, lmbda_range))
            m1_H = -mu_H[:,None] * m0_H
            m2_H = -mu_H[:,None] * m1_H
            m0_L = np.exp(-np.outer(mu_L, lmbda_range))
            m1_L = -mu_L[:,None] * m0_L
            m2_L = -mu_L[:,None] * m1_L
            d_H0 = D_phi_H[k] @ m0_H
            d_L0 = D_phi_L[k] @ m0_L
            d_H1 = D_phi_H[k] @ m1_H
            d_L1 = D_phi_L[k] @ m1_L
            d_H2 = D_phi_H[k] @ m2_H
            d_L2 = D_phi_L[k] @ m2_L
            tables[0,k,m,:] = -np.log(d_H0 / d_H[k])
            tables[1,k,m,:] = -np.log(d_L0 / d_L[k])
            tables[2,k,m,:] = -d_H1 / d_H0
            tables[3,k,m,:] = -d_L1 / d_L0
            tables[4,k,m,:] = (d_H1**2 - d_H0 * d_H2) / d_H0**2
            tables[5,k,m,:] = (d_L1**2 - d_L0 * d_L2) / d_L0**2
            
    print("completed in %.2f seconds" % (time() - t1))
    return tables

def fitSemiempirical(alpha, lmbda, Z, phi, D, E):
    """
    Finds the optimal calibration parameters 'a', 'b', and 'c' for a set of
    calibration measurements.
    
    Parameters
    ----------
    alpha : array_like
        List of calibration log-transparency measurements. Must contain at
        least three measurements.
    
    lmbda : array_like
        List of area density values for each calibration material.
                
    Z : array_like
        List of atomic number values for each calibration material. If a
        calibration material is heterogeneous (i.e. polyethylene), the
        corresponding list entry can array_like.
             
    phi : array_like
        Shape (n,) photon beam spectrum.

    D : array_like, optional
        Shape (n,) detector response function.
        
    E : array_like, optional
        Shape (n,) energy grid, on which `phi` and `D` are defined.
        
    Returns
    -------
    
    a : float
        Optimal `a` photoelectric effect calibration parameter.
    
    b : float
        Optimal `b` Compton scattering calibration parameter.
        
    c : float
        Optimal `c` pair production calibration parameter.
    
    """
    def calcLoss(x):
        loss = 0
        for i in range(len(Z)):
            key = Z[i] if np.size(Z[i]) == 1 else tuple(Z[i])
            mu = mu_tot_dict[key] + (x[0]-1)*mu_PE_dict[key] + (x[1]-1)*mu_CS_dict[key] + (x[2]-1)*mu_PP_dict[key]
            if np.size(Z[i]) == 1:
                m0 = np.exp(-mu * lmbda[i])
            else:
                m0 = np.exp(-np.sum(mu * lmbda[i], axis=1))
            d0 = D_phi @ m0
            alpha0 = -np.log(d0 / d)
            loss += (alpha0 - alpha[i])**2
        return loss
    
    t0 = time()
    print("Running calibration...", end='')
    
    assert len(alpha) >= 3
    D_phi = D * phi
    d = np.sum(D_phi)
    mu_tot_dict = {}
    mu_PE_dict = {}
    mu_CS_dict = {}
    mu_PP_dict = {}
    for i in range(len(Z)):
        key = Z[i] if np.size(Z[i]) == 1 else tuple(Z[i])
        mu_tot_dict[key] = mu_tot(E, Z[i])
        mu_PE_dict[key] = mu_PE(E, Z[i])
        mu_CS_dict[key] = mu_CS(E, Z[i])
        mu_PP_dict[key] = mu_PP(E, Z[i])
        
    res = minimize(calcLoss, x0=(1, 1, 1), bounds=[(0, None)]*3)
    assert res.success
    a, b, c = res.x

    print("completed in %.3f seconds" % (time() - t0))
    print("Minimum found at a = %.4f, b = %.4f, c = %.4f with a loss of %.3e" % (a, b, c, res.fun))
    return a, b, c
