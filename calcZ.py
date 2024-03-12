import numpy as np
from time import perf_counter as time
       
def calcZ(im_H, im_L, lookup, imVar_H=None, imVar_L=None, labels=None, lmbda_range=None, Z_range=None, eps=0):
    """
    Approximates the area density (lambda) and atomic number (Z) of dual energy
    cargo images.
    
    Parameters
    ----------
    im_H : array_like
        Shape (h, l) high energy image of log-transparencies.
    
    im_L : array_like
        Shape (h, l) low energy image of log-transparencies.
                
    lookup : function, callable
        Given an array of detector indices, lambda values, and Z, returns
        transparency predictions and derivatives in the form (alpha_H, alpha_L,
        alpha_H', alpha_L', alpha_H'', and alpha_L'').
             
    imVar_H : array_like, optional
        Shape (h, l) variance matrix, where entry (i, j) is the variance of
        im_H[i,j]. Used when computing chi-squared. If not supplied, assumes the
        variance of every pixel is 1.

    imVar_L : array_like, optional
        Shape (h, l) variance matrix, where entry (i, j) is the variance of
        im_L[i,j]. Used when computing chi-squared. If not supplied, assumes the
        variance of every pixel is 1.
        
    labels : array_like, optional
        Shape (h, l) label map for pixel clustering. When computing chi-squared,
        all pixels with the same label are grouped together and enforced to have
        the same output Z, which helps reduce noise. If not supplied, each pixel
        is assumed to belong to its own cluster.
        
    lmbda_range : array_like, optional
        Bounds for performing search for best lambda.
        
    Z_range : ndarray, optional
        Grid for performing search for best Z.
        
    eps : float, optional
        Parameter to help break Z degeneracies by biasing towards the lower-Z
        solution. A larger value yields fewer spurious high-Z regions.
        
    Returns
    -------
    
    im_lambda : ndarray
        shape (h, l) approximation of the area density of each pixel
    
    im_Z : ndarray
        shape (h, l) approximation of the atomic number of each pixel
    
    """
    use_segmentation = labels is not None
    include_uncertainties = imVar_H is not None and imVar_L is not None
    if not use_segmentation:
        print("Warning: setting `labels = None` will result in a noisier output")
    
    if lmbda_range is None:
        lmbda_range = (0, 300)
    if Z_range is None:
        Z_range = np.arange(1, 101)
    if use_segmentation and include_uncertainties:
        return _calcZ_SegmentationUncertainties(im_H, im_L, lookup, imVar_H, imVar_L, labels, lmbda_range, Z_range, eps)
    elif use_segmentation:
        return _calcZ_Segmentation(im_H, im_L, lookup, labels, lmbda_range, Z_range, eps)
    elif include_uncertainties:
        return _calcZ_Uncertainties(im_H, im_L, lookup, imVar_H, imVar_L, lmbda_range, Z_range, eps)
    else:
        return _calcZ(im_H, im_L, lookup, lmbda_range, Z_range, eps)
        

def _calcZ(im_H, im_L, lookup, lmbda_range, Z_range, eps):
    h, l = im_H.shape
    n = im_H.size
    lmbda_min, lmbda_max = lmbda_range[0], lmbda_range[-1]
    m = Z_range.size
    
    alpha_H = im_H.ravel()
    alpha_L = im_L.ravel()
    k_arr = np.repeat(np.arange(h), l)[::-1]

    print("Processing image...", end='')
    t0 = time()
    
    lmbda = np.zeros(n)
    lmbda_opt = np.ones(n)
    Z_opt = np.zeros(n, dtype=Z_range.dtype)
    loss_opt = np.full(n, np.inf)

    for i in range(m):
        Z = Z_range[i]
        nsteps = 4 if i == 0 else 1
        for _ in range(nsteps):
            alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
            diff_H = alpha_H0 - alpha_H
            diff_L = alpha_L0 - alpha_L
            grad = diff_H * alpha_H1 + diff_L * alpha_L1
            hess = diff_H * alpha_H2 + alpha_H1**2 + diff_L * alpha_L2 + alpha_L1**2
            lmbda = lmbda - grad / hess
            lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

        alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
        loss = (alpha_H0 - alpha_H)**2 + (alpha_L0 - alpha_L)**2 + eps*Z
        cut = loss < loss_opt
        lmbda_opt[cut] = lmbda[cut]
        Z_opt[cut] = Z
        loss_opt[cut] = loss[cut]

    print("completed in %.2f seconds" % (time() - t0))
    print("    Average loss: %.3e" % np.mean(loss_opt))
    return lmbda_opt.reshape(h, l), Z_opt.reshape(h, l)

def _calcZ_Uncertainties(im_H, im_L, lookup,  imVar_H, imVar_L, lmbda_range, Z_range, eps):
    h, l = im_H.shape
    n = im_H.size
    lmbda_min, lmbda_max = lmbda_range[0], lmbda_range[-1]
    m = Z_range.size
    
    alpha_H = im_H.ravel()
    alpha_L = im_L.ravel()
    var_H = imVar_H.ravel()
    var_L = imVar_L.ravel()
    k_arr = np.repeat(np.arange(h), l)[::-1]

    print("Processing image...", end='')
    t0 = time()
    
    lmbda = np.zeros(n)
    lmbda_opt = np.ones(n)
    Z_opt = np.zeros(n, dtype=Z_range.dtype)
    chi2_opt = np.full(n, np.inf)

    for i in range(m):
        Z = Z_range[i]
        nsteps = 4 if i == 0 else 1
        for _ in range(nsteps):
            alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
            diff_H = (alpha_H0 - alpha_H) / var_H
            diff_L = (alpha_L0 - alpha_L) / var_L
            grad = diff_H * alpha_H1 + diff_L * alpha_L1
            hess = diff_H * alpha_H2 + alpha_H1**2 / var_H + diff_L * alpha_L2 + alpha_L1**2 / var_L
            lmbda = lmbda - grad / hess
            lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

        alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
        chi2 = (alpha_H0 - alpha_H)**2 + (alpha_L0 - alpha_L)**2 + eps*Z
        cut = chi2 < chi2_opt
        lmbda_opt[cut] = lmbda[cut]
        Z_opt[cut] = Z
        chi2_opt[cut] = chi2[cut]

    print("completed in %.2f seconds" % (time() - t0))
    print("    Average chi2: %.3e" % np.mean(chi2_opt))
    return lmbda_opt.reshape(h, l), Z_opt.reshape(h, l)

def _calcZ_Segmentation(im_H, im_L, lookup, labels, lmbda_range, Z_range, eps):
    h, l = im_H.shape
    n = im_H.size
    lmbda_min, lmbda_max = lmbda_range[0], lmbda_range[-1]
    m = Z_range.size
    
    alpha_H = im_H.ravel()
    alpha_L = im_L.ravel()
    k_arr = np.repeat(np.arange(h), l)[::-1]
    labels = labels.ravel()
    n_labels = np.max(labels) + 1

    t0 = time()
    print("Processing image...", end='')

    lmbda = np.zeros(n)
    loss_opt = np.full(n_labels, np.inf)
    Z_opt = np.full(n_labels, 0, dtype=Z_range.dtype)

    for i in range(m):
        Z = Z_range[i]
        nsteps = 4 if i == 0 else 1 
        for _ in range(nsteps):
            alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
            diff_H = alpha_H0 - alpha_H
            diff_L = alpha_L0 - alpha_L
            grad = diff_H * alpha_H1 + diff_L * alpha_L1
            hess = diff_H * alpha_H2 + alpha_H1**2 + diff_L * alpha_L2 + alpha_L1**2
            lmbda = lmbda - grad / hess
            lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

        alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
        loss_arr = (alpha_H0 - alpha_H)**2 + (alpha_L0 - alpha_L)**2 + eps*Z
        loss = np.bincount(labels, weights = loss_arr)
        cut = loss < loss_opt
        loss_opt[cut] = loss[cut] 
        Z_opt[cut] = Z

    Z = Z_opt[labels]
    for _ in range(3):
        alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
        diff_H = alpha_H0 - alpha_H
        diff_L = alpha_L0 - alpha_L
        grad = diff_H * alpha_H1 + diff_L * alpha_L1
        hess = diff_H * alpha_H2 + alpha_H1**2 + diff_L * alpha_L2 + alpha_L1**2
        lmbda = lmbda - grad / hess
        lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

    alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
    loss_arr = (alpha_H0 - alpha_H)**2 + (alpha_L0 - alpha_L)**2

    print("completed in %.3f seconds" % (time() - t0))
    print("    Average loss: %.3e" % np.mean(loss_arr))
    return lmbda.reshape(h, l), Z.reshape(h, l)

def _calcZ_SegmentationUncertainties(im_H, im_L, lookup, imVar_H, imVar_L, labels, lmbda_range, Z_range, eps):
    h, l = im_H.shape
    n = im_H.size
    lmbda_min, lmbda_max = lmbda_range[0], lmbda_range[-1]
    m = Z_range.size
    
    alpha_H = im_H.ravel()
    alpha_L = im_L.ravel()
    var_H = imVar_H.ravel()
    var_L = imVar_L.ravel()
    k_arr = np.repeat(np.arange(h), l)[::-1]
    labels = labels.ravel()
    n_labels = np.max(labels) + 1

    t0 = time()
    print("Processing image...", end='')

    lmbda = np.zeros(n)
    chi2_opt = np.full(n_labels, np.inf)
    Z_opt = np.full(n_labels, 0, dtype=Z_range.dtype)

    for i in range(m):
        Z = Z_range[i]
        nsteps = 4 if i == 0 else 1 
        for _ in range(nsteps):
            alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
            diff_H = (alpha_H0 - alpha_H) / var_H
            diff_L = (alpha_L0 - alpha_L) / var_L
            grad = diff_H * alpha_H1 + diff_L * alpha_L1
            hess = diff_H * alpha_H2 + alpha_H1**2 / var_H + diff_L * alpha_L2 + alpha_L1**2 / var_L
            lmbda = lmbda - grad / hess
            lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

        alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
        chi2_arr = (alpha_H0 - alpha_H)**2 + (alpha_L0 - alpha_L)**2 + eps*Z
        chi2 = np.bincount(labels, weights = chi2_arr)
        cut = chi2 < chi2_opt
        chi2_opt[cut] = chi2[cut] 
        Z_opt[cut] = Z

    Z = Z_opt[labels]
    for _ in range(3):
        alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2 = lookup(k_arr, lmbda, Z)
        diff_H = (alpha_H0 - alpha_H) / var_H
        diff_L = (alpha_L0 - alpha_L) / var_L
        grad = diff_H * alpha_H1 + diff_L * alpha_L1
        hess = diff_H * alpha_H2 + alpha_H1**2 / var_H + diff_L * alpha_L2 + alpha_L1**2 / var_L
        lmbda = lmbda - grad / hess
        lmbda = np.clip(lmbda, lmbda_min, lmbda_max)

    alpha_H0, alpha_L0 = lookup(k_arr, lmbda, Z, return_derivatives=False)
    chi2_arr = (alpha_H0 - alpha_H)**2 / var_H + (alpha_L0 - alpha_L)**2 / var_L
    chi2 = np.sum(chi2_arr)
    dof = n - n_labels

    print("completed in %.3f seconds" % (time() - t0))
    print("    Reduced chi2: %.3f" % (chi2 / dof))
    return lmbda.reshape(h, l), Z.reshape(h, l)
