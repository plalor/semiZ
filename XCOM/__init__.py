from .XCOM import XCOM

_XCOM = XCOM()

def mu_tot(E, Z):
    """
    Calculates the total mass attenuation coefficient (cm^2/g) of atomic 
    number Z at energy E.
    
    Parameters
    ----------
    E : float or array_like
        Energy (in MeV) at which to calculate mu.
        
    Z : float or ndarray
        Atomic number at which to calculate mu. If a non-integer is
        given, performs linear interpolation between floor(Z) and ceil(Z).
        
    Returns
    -------
    
    mu : float or ndarray
        Total mass attenuation coefficient, evaluated at (E, Z).
    
    """
    return _XCOM.mu_tot(E, Z)

def mu_PE(E, Z):
    """
    Calculates the photoelectric effect mass attenuation coefficient (cm^2/g) 
    of atomic number Z at energy E.
    
    Parameters
    ----------
    E : float or array_like
        Energy (in MeV) at which to calculate mu.
        
    Z : float or ndarray
        Atomic number at which to calculate mu. If a non-integer is
        given, performs linear interpolation between floor(Z) and ceil(Z).
        
    Returns
    -------
    
    mu : float or ndarray
        Photoelectric effect mass attenuation coefficient, evaluated at (E, Z).
    
    """
    return _XCOM.mu_PE(E, Z)

def mu_CS(E, Z):
    """
    Calculates the Compton scattering mass attenuation coefficient (cm^2/g) 
    of atomic number Z at energy E.
    
    Parameters
    ----------
    E : float or array_like
        Energy (in MeV) at which to calculate mu.
        
    Z : float or ndarray
        Atomic number at which to calculate mu. If a non-integer is
        given, performs linear interpolation between floor(Z) and ceil(Z).
        
    Returns
    -------
    
    mu : float or ndarray
        Compton scattering mass attenuation coefficient, evaluated at (E, Z).
    
    """
    return _XCOM.mu_CS(E, Z)

def mu_PP(E, Z):
    """
    Calculates the pair production mass attenuation coefficient (cm^2/g) 
    of atomic number Z at energy E.
    
    Parameters
    ----------
    E : float or array_like
        Energy (in MeV) at which to calculate mu.
        
    Z : float or ndarray
        Atomic number at which to calculate mu. If a non-integer is
        given, performs linear interpolation between floor(Z) and ceil(Z).
        
    Returns
    -------
    
    mu : float or ndarray
        Pair production mass attenuation coefficient, evaluated at (E, Z).
    
    """
    return _XCOM.mu_PP(E, Z)
