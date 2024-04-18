import numpy as np
from scipy.interpolate import interp1d
import pkgutil

class XCOM:
    def __init__(self):
        self.IncoherScatterInterpolators = []
        self.PhotoelAbsorbInterpolators = []
        self.NuclearPrPrdInterpolators = []
        self.TotWCoherentInterpolators = []

        for Z in range(1, 101):
            filename = "%02d.txt" % Z
            PhotonEnergy_arr, IncoherScatter_arr, PhotoelAbsorb_arr, NuclearPrPrd_arr, TotWCoherent_arr = self.loadMassAtten(filename)

            self.IncoherScatterInterpolators.append(self.buildInterpolator(PhotonEnergy_arr, IncoherScatter_arr))
            self.PhotoelAbsorbInterpolators.append(self.buildInterpolator(PhotonEnergy_arr, PhotoelAbsorb_arr))
            self.NuclearPrPrdInterpolators.append(self.buildInterpolator(PhotonEnergy_arr, NuclearPrPrd_arr))
            self.TotWCoherentInterpolators.append(self.buildInterpolator(PhotonEnergy_arr, TotWCoherent_arr))
            
    def mu_tot(self, E, Z):
        if np.size(Z) > 1:
            return np.array([self.mu_tot(E, Z[i]) for i in range(np.size(Z))]).T
        if Z < 1 or Z > 100:
            raise ValueError("Invalid value: Z = %d; Z must be between 1 and 100" % Z)
        if np.min(E) < 1e-3 or np.max(E) > 1e5:
            raise ValueError("Energy must be between 1 keV and 100 GeV")
        
        idx1 = np.floor(Z-1).astype('int')
        idx2 = np.ceil(Z-1).astype('int')
        coef1 = self.TotWCoherentInterpolators[idx1](E)
        if idx1 == idx2:
            return coef1
        else:
            coef2 = self.TotWCoherentInterpolators[idx2](E)
            f = Z % 1
            return f * coef2 + (1 - f) * coef1
        
    def mu_PE(self, E, Z):
        if np.size(Z) > 1:
            return np.array([self.mu_PE(E, Z[i]) for i in range(np.size(Z))]).T
        if Z < 1 or Z > 100:
            raise ValueError("Invalid value: Z = %d; Z must be between 1 and 100" % Z)
        if np.min(E) < 1e-3 or np.max(E) > 1e5:
            raise ValueError("Energy must be between 1 keV and 100 GeV")
        
        idx1 = np.floor(Z-1).astype('int')
        idx2 = np.ceil(Z-1).astype('int')
        coef1 = self.PhotoelAbsorbInterpolators[idx1](E)
        if idx1 == idx2:
            return coef1
        else:
            coef2 = self.PhotoelAbsorbInterpolators[idx2](E)
            f = Z % 1
            return f * coef2 + (1 - f) * coef1
        
    def mu_CS(self, E, Z):
        if np.size(Z) > 1:
            return np.array([self.mu_CS(E, Z[i]) for i in range(np.size(Z))]).T
        if Z < 1 or Z > 100:
            raise ValueError("Invalid value: Z = %d; Z must be between 1 and 100" % Z)
        if np.min(E) < 1e-3 or np.max(E) > 1e5:
            raise ValueError("Energy must be between 1 keV and 100 GeV")
        
        idx1 = np.floor(Z-1).astype('int')
        idx2 = np.ceil(Z-1).astype('int')
        coef1 = self.IncoherScatterInterpolators[idx1](E)
        if idx1 == idx2:
            return coef1
        else:
            coef2 = self.IncoherScatterInterpolators[idx2](E)
            f = Z % 1
            return f * coef2 + (1 - f) * coef1
        
    def mu_PP(self, E, Z):
        if np.size(Z) > 1:
            return np.array([self.mu_PP(E, Z[i]) for i in range(np.size(Z))]).T
        if Z < 1 or Z > 100:
            raise ValueError("Invalid value: Z = %d; Z must be between 1 and 100" % Z)
        if np.min(E) < 1e-3 or np.max(E) > 1e5:
            raise ValueError("Energy must be between 1 keV and 100 GeV")
        
        idx1 = np.floor(Z-1).astype('int')
        idx2 = np.ceil(Z-1).astype('int')
        coef1 = self.NuclearPrPrdInterpolators[idx1](E)
        if idx1 == idx2:
            return coef1
        else:
            coef2 = self.NuclearPrPrdInterpolators[idx2](E)
            f = Z % 1
            return f * coef2 + (1 - f) * coef1
    
    def loadMassAtten(self, filename):
        """Loads NIST mass attenuation coefficient data"""    
        data = pkgutil.get_data(__name__, "MassAttenCoefs/%s" % filename).decode("utf-8").split("\n")[3:-1]
        n = len(data)

        PhotonEnergy_arr = np.zeros(n)
        IncoherScatter_arr = np.zeros(n)
        PhotoelAbsorb_arr = np.zeros(n)
        NuclearPrPrd_arr = np.zeros(n)
        TotWCoherent_arr = np.zeros(n)

        for i in range(n):
            PhotonEnergy, CoherentScatter, IncoherScatter, PhotoelAbsorb, NuclearPrPrd, ElectronPrPrd, TotWCoherent, TotWoCoherent = data[i].split()
            PhotonEnergy_arr[i] = PhotonEnergy
            IncoherScatter_arr[i] = IncoherScatter
            PhotoelAbsorb_arr[i] = PhotoelAbsorb
            NuclearPrPrd_arr[i] = NuclearPrPrd
            TotWCoherent_arr[i] = TotWCoherent

        ### Changing absorption edge energies so they don't have identical x values
        for i in range(n-1):
            if PhotonEnergy_arr[i] == PhotonEnergy_arr[i+1]:
                energyStr = np.format_float_scientific(PhotonEnergy_arr[i], precision=5)
                idx = energyStr.find("e")
                energyPrefix = energyStr[:idx]
                energySuffix = energyStr[idx:]
                energyPrefixNew = str(float(energyPrefix) - 1e-5)
                PhotonEnergy_arr[i] = float(energyPrefixNew + energySuffix)

        return PhotonEnergy_arr, IncoherScatter_arr, PhotoelAbsorb_arr, NuclearPrPrd_arr, TotWCoherent_arr

    def buildInterpolator(self, x, y):
        """Returns a function that takes energy as input and
        returns the corresponding attenuationg using log-log
        interpolation"""
        with np.errstate(divide='ignore'):
            interpRaw = interp1d(np.log(x), np.log(y))
        interp = lambda x: np.exp(interpRaw(np.log(x)))
        return interp
