import numpy as np

class Lookup:
    """
    Class for fetching the forward model and its derivatives from precomputed
    lookup tables.
    
    Parameters
    ----------
    tables : array_like
        Shape (6, k, m, l) lookup table. tables[:, idx_k, idx_m, idx_l] gives
        alpha_H, alpha_L, alpha_H', alpha_L', alpha_H'', alpha_L'' evaluated
        at lmbda_range[idx_m], Z_range[idx_l] for detector idx_k.
    
    lmbda_range : array_like
        Shape (l,) lambda grid, on which tables is defined
                
    Z_range : array_like
        Shape (m,) Z grid, on which tables is defined
             
    interpolate_lmbda : bool, optional
        When tables is evaluated at a lambda which is not in lmbda_range,
        determines whether to interpolate between the two adjacent lambda
        bins, or just choose the closest bin.

    interpolate_Z : bool, optional
        When tables is evaluated at a Z which is not in Z_range, determines
        whether to interpolate between the two adjacent Z bins, or just
        choose the closest bin.
        
    Returns
    -------
    
    Lookup object, which can be called to compute the forward model. See
    documentation under the __call__ method for more details.
    
    """
    def __init__(self, tables, lmbda_range, Z_range, interpolate_lmbda = False, interpolate_Z = False):
        lmbda_diff = np.diff(lmbda_range)
        Z_diff = np.diff(Z_range)
        if not np.allclose(lmbda_diff, lmbda_diff[0]) or not np.allclose(Z_diff, Z_diff[0]):
            raise ValueError("Must use equispaced grid to properly define Lookup object")
        self.lmbda_min = lmbda_range[0]
        self.dLmbda = lmbda_diff[0]
        self.Z_min = Z_range[0]
        self.dZ = Z_diff[0]
        self.tables = tables
        if interpolate_lmbda and interpolate_Z:
            self.lookup = self._lookup_interpolate_lmbda_Z
        elif interpolate_lmbda:
            self.lookup = self._lookup_interpolate_lmbda
        elif interpolate_Z:
            self.lookup = self._lookup_interpolate_Z
        else:
            self.lookup = self._lookup
            
    def __call__(self, idx_k, lmbda, Z, return_derivatives = True):
        """
        Function for calculating the forward model and its derivatives.

        Parameters
        ----------
        idx_k : array_like
            The detector index to evaluate the forward model.

        lmbda : array_like
            The area density to evaluate the forward model.

        Z : array_like
            The atomic number to evaluate the forward model.

        return_derivatives : bool, optional
            If True, returns the first and second derivatives in addition to the 
            log-transparencies. If False, just returns the log-transparencies.

        Returns
        -------

        alpha_H : array_like
            The high energy log-transparency.
        
        alpha_L : array_like
            The low energy log-transparency.
        
        alpha_H' : array_like
            The derivative of high energy log-transparency with respect to
            lambda. Only returned if return_derivatives = True.
        
        alpha_L' : array_like
            The derivative of low energy log-transparency with respect to
            lambda. Only returned if return_derivatives = True.
            
        alpha_H'' : array_like
            The second derivatve of high energy log-transparency with respect
            to lambda. Only returned if return_derivatives = True.
        
        alpha_L'' : array_like
            The second derivative of low energy log-transparency with respect
            to lambda. Only returned if return_derivatives = True.
        
        """
        return self.lookup(idx_k, lmbda, Z, return_derivatives)
        
    def _lookup(self, idx_k, lmbda, Z, return_derivatives = True):
        """No interpolation"""
        idx_lmbda = np.rint((lmbda - self.lmbda_min) / self.dLmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_Z = np.rint((Z - self.Z_min) / self.dZ).astype(int)
        if return_derivatives:
            return self.tables[:, idx_k, idx_Z, idx_lmbda]
        else:
            return self.tables[:2, idx_k, idx_Z, idx_lmbda]
        
    def _lookup_interpolate_Z(self, idx_k, lmbda, Z, return_derivatives = True):
        """Performs linear interpolation on Z"""
        idx_lmbda = np.rint((lmbda - self.lmbda_min) / self.dLmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_Z = (Z - self.Z_min) / self.dZ
        idx_Z_low = np.floor(idx_Z).astype(int)
        idx_Z_high = np.ceil(idx_Z).astype(int)
        d = idx_Z % 1
        if return_derivatives:
            alpha_H0_low, alpha_L0_low, alpha_H1_low, alpha_L1_low, alpha_H2_low, alpha_L2_low = self.tables[:, idx_k, idx_Z_low, idx_lmbda]
            alpha_H0_high, alpha_L0_high, alpha_H1_high, alpha_L1_high, alpha_H2_high, alpha_L2_high = self.tables[:, idx_k, idx_Z_high, idx_lmbda]
            alpha_H0 = d * alpha_H0_high + (1-d) * alpha_H0_low
            alpha_L0 = d * alpha_L0_high + (1-d) * alpha_L0_low
            alpha_H1 = d * alpha_H1_high + (1-d) * alpha_H1_low
            alpha_L1 = d * alpha_L1_high + (1-d) * alpha_L1_low
            alpha_H2 = d * alpha_H2_high + (1-d) * alpha_H2_low
            alpha_L2 = d * alpha_L2_high + (1-d) * alpha_L2_low
            return alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2
        else:
            alpha_H0_low, alpha_L0_low = self.tables[:2, idx_k, idx_Z_low, idx_lmbda]
            alpha_H0_high, alpha_L0_high = self.tables[:2, idx_k, idx_Z_high, idx_lmbda]
            alpha_H0 = d * alpha_H0_high + (1-d) * alpha_H0_low
            alpha_L0 = d * alpha_L0_high + (1-d) * alpha_L0_low
            return alpha_H0, alpha_L0
        
    def _lookup_interpolate_lmbda(self, idx_k, lmbda, Z, return_derivatives = True):
        """Performs linear interpolation on lambda"""
        idx_lmbda = (lmbda - self.lmbda_min) / self.dLmbda
        idx_lmbda_low = np.floor(idx_lmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_lmbda_high = np.ceil(idx_lmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_Z = np.rint((Z - self.Z_min) / self.dZ).astype(int)
        c = idx_lmbda % 1
        if return_derivatives:
            alpha_H0_low, alpha_L0_low, alpha_H1_low, alpha_L1_low, alpha_H2_low, alpha_L2_low = self.tables[:, idx_k, idx_Z, idx_lmbda_low]
            alpha_H0_high, alpha_L0_high, alpha_H1_high, alpha_L1_high, alpha_H2_high, alpha_L2_high = self.tables[:, idx_k, idx_Z, idx_lmbda_high]
            alpha_H0 = c * alpha_H0_high + (1-c) * alpha_H0_low
            alpha_L0 = c * alpha_L0_high + (1-c) * alpha_L0_low
            alpha_H1 = c * alpha_H1_high + (1-c) * alpha_H1_low
            alpha_L1 = c * alpha_L1_high + (1-c) * alpha_L1_low
            alpha_H2 = c * alpha_H2_high + (1-c) * alpha_H2_low
            alpha_L2 = c * alpha_L2_high + (1-c) * alpha_L2_low
            return alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2
        else:
            alpha_H0_low, alpha_L0_low = self.tables[:2, idx_k, idx_Z, idx_lmbda_low]
            alpha_H0_high, alpha_L0_high = self.tables[:2, idx_k, idx_Z, idx_lmbda_high]
            alpha_H0 = c * alpha_H0_high + (1-c) * alpha_H0_low
            alpha_L0 = c * alpha_L0_high + (1-c) * alpha_L0_low
            return alpha_H0, alpha_L0
        
    def _lookup_interpolate_lmbda_Z(self, idx_k, lmbda, Z, return_derivatives = True):
        """Performs linear interpolation and lambda and Z"""
        idx_lmbda = (lmbda - self.lmbda_min) / self.dLmbda
        idx_lmbda_low = np.floor(idx_lmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_lmbda_high = np.ceil(idx_lmbda, out=np.zeros(lmbda.size, int), casting='unsafe')
        idx_Z = (Z - self.Z_min) / self.dZ
        idx_Z_low = np.floor(idx_Z).astype(int)
        idx_Z_high = np.ceil(idx_Z).astype(int)
        c = idx_lmbda % 1
        d = idx_Z % 1
        if return_derivatives:
            alpha_H0_00, alpha_L0_00, alpha_H1_00, alpha_L1_00, alpha_H2_00, alpha_L2_00 = self.tables[:, idx_k, idx_Z_low, idx_lmbda_low]
            alpha_H0_01, alpha_L0_01, alpha_H1_01, alpha_L1_01, alpha_H2_01, alpha_L2_01 = self.tables[:, idx_k, idx_Z_low, idx_lmbda_high]
            alpha_H0_10, alpha_L0_10, alpha_H1_10, alpha_L1_10, alpha_H2_10, alpha_L2_10 = self.tables[:, idx_k, idx_Z_high, idx_lmbda_low]
            alpha_H0_11, alpha_L0_11, alpha_H1_11, alpha_L1_11, alpha_H2_11, alpha_L2_11 = self.tables[:, idx_k, idx_Z_high, idx_lmbda_high]
            alpha_H0 = c*d*alpha_H0_11 + (1-c)*d*alpha_H0_10 + c*(1-d)*alpha_H0_01 + (1-c)*(1-d)*alpha_H0_00
            alpha_L0 = c*d*alpha_L0_11 + (1-c)*d*alpha_L0_10 + c*(1-d)*alpha_L0_01 + (1-c)*(1-d)*alpha_L0_00
            alpha_H1 = c*d*alpha_H1_11 + (1-c)*d*alpha_H1_10 + c*(1-d)*alpha_H1_01 + (1-c)*(1-d)*alpha_H1_00
            alpha_L1 = c*d*alpha_L1_11 + (1-c)*d*alpha_L1_10 + c*(1-d)*alpha_L1_01 + (1-c)*(1-d)*alpha_L1_00
            alpha_H2 = c*d*alpha_H2_11 + (1-c)*d*alpha_H2_10 + c*(1-d)*alpha_H2_01 + (1-c)*(1-d)*alpha_H2_00
            alpha_L2 = c*d*alpha_L2_11 + (1-c)*d*alpha_L2_10 + c*(1-d)*alpha_L2_01 + (1-c)*(1-d)*alpha_L2_00
            return alpha_H0, alpha_L0, alpha_H1, alpha_L1, alpha_H2, alpha_L2
        else:
            alpha_H0_00, alpha_L0_00 = self.tables[:2, idx_k, idx_Z_low, idx_lmbda_low]
            alpha_H0_01, alpha_L0_01 = self.tables[:2, idx_k, idx_Z_low, idx_lmbda_high]
            alpha_H0_10, alpha_L0_10 = self.tables[:2, idx_k, idx_Z_high, idx_lmbda_low]
            alpha_H0_11, alpha_L0_11 = self.tables[:2, idx_k, idx_Z_high, idx_lmbda_high]
            alpha_H0 = c*d*alpha_H0_11 + (1-c)*d*alpha_H0_10 + c*(1-d)*alpha_H0_01 + (1-c)*(1-d)*alpha_H0_00
            alpha_L0 = c*d*alpha_L0_11 + (1-c)*d*alpha_L0_10 + c*(1-d)*alpha_L0_01 + (1-c)*(1-d)*alpha_L0_00
            return alpha_H0, alpha_L0
