import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.special import j0, jn_zeros
from scipy.special import roots_legendre
from scipy import integrate
from tqdm import tqdm  # For progress bars
from scipy.special import jv
from .utility import *
from .constants import *

@njit
def integrand_xipm_helper(ell, pkappas, ell_values, J04):
    """
    Helper function to compute the integrand for Xi_+-.

    Parameters
    ----------
    ell : float
        The ell value.
    pkappas : numpy.ndarray
        Precomputed power spectrum values.
    ell_values : numpy.ndarray
        Array of ell values.
    J04 : float
        The value of zeroth/fourth Bessel function function at ell times theta.

    Returns
    -------
    float
        The result of the integrand computation.
    """
    # pk = custom_interpolate_1d(ell_values, pkappas, ell)
    pk = 10**custom_interpolate_1d(ell_values, pkappas, np.log10(ell))  
    # print(ell, pk, W)
    return ell * pk * J04

class xipm:
    """
    Class for calculating xi+- for weak gravitational lensing
    from power spectrum. 

    Parameters
    ----------
    limber : object
        Limber approximation object for power spectrum computation.
    verbose : bool, optional
        Control amount of output
    """
    def __init__(self,  
                 limber : object,
                 verbose=True,
                 nell = 256, ell_min = 1, ell_max = 20000):
        
        
        self.limber=limber
        self.Ntomos = self.limber.Ntomo

        self.verbose=verbose

        if self.verbose:
            print("Precomputing projected powerspectrum")
    
        self.precompute_pkappa_allTomos(ell_min= ell_min, ell_max = ell_max, nell = nell)

    
    def precompute_pkappa_allTomos(self, ell_min = 1, ell_max = 20000, nell = 1000):
        """
        Precomputes the power spectrum (pkappa) on a 1D grid for all unique tomography combinations.

        Parameters
        ----------
        ell_range : list, optional, default=[1, 20000]
            Range of `ell` values for power spectrum grid.

        Notes
        -----
        Saves computed `pkappa` values in `pkappas`, keyed by tomography combination.
        """

        self.pkappas={}

        self.ell_values = np.logspace(np.log10(ell_min), np.log10(ell_max), nell)  # Logarithmically spaced
        
        for tomo1 in range(self.Ntomos):
            for tomo2 in tqdm(range(tomo1, self.Ntomos), desc=f"preparing tomo {tomo1}"):
                combi=f"{tomo1}-{tomo2}"

                self.pkappas[combi]=self.precompute_pkappa(self.ell_values, tomo1, tomo2)


        self.ell_values = np.log10(self.ell_values)

    

    def precompute_pkappa(self, ell_values, tomo1, tomo2):
        """
        Computes the power spectrum grid for a specific tomography combination.

        Parameters
        ----------
        ell_values : numpy.ndarray
            1D array of `ell` values for the grid.
        tomo1, tomo2 : int
            Tomography bin indices.

        Returns
        -------
        numpy.ndarray
            1D array of computed `pkappa` values for the specified tomography combination.
        """

        # Compute bkappa values on the grid
        pkappa_grid = self.limber.powerspectrum_projected(ell_values, tomo1, tomo2)
        
        
        pkappa_grid[np.where(pkappa_grid<=0)[0]] = np.min(pkappa_grid[np.where(pkappa_grid>0)[0]])/10
        pkappa_grid = np.log10(pkappa_grid)
        pkappa_grid[np.where(np.isnan(pkappa_grid))]=0.0
        pkappa_grid[np.where(np.isinf(pkappa_grid))]=0.0

        return pkappa_grid


    def integrand_xiplus(self, ell, theta, order, tomo1, tomo2):
        """
        Computes the integrand for the En COSEBI moment.

        Parameters
        ----------
        ell : float
            Ell value for integration.
        theta : int
            separarion of xi+-.
        tomo1 : int
            First tomography bin index.
        tomo2 : int
            Second tomography bin index.

        Returns
        -------
        float
            Value of the integrand for En.
        """
        combi=f"{tomo1}-{tomo2}"
        
        j04 = jv(order, ell * theta)
        
        return integrand_xipm_helper(ell, self.pkappas[combi], self.ell_values, j04)


    def xiplus(self, theta, nell, tomo1, tomo2):
        """
        Computes xiplus at theta for given tomography bins.

        Parameters
        ----------
        theta : float
            separarion of xi+.
        nell :
            number of ell for integrtion
        tomo1 : int
            First tomography bin index.
        tomo2 : int
            Second tomography bin index.

        Returns
        -------
        float
            Value of the xi+.
        """
            # Set up the integration range and points for Simpson's rule
        ell_values = np.geomspace(10**np.min(self.ell_values), 10**np.max(self.ell_values), nell)  # 1000 points for accuracy
        integrand_values = [self.integrand_xiplus(ell, theta, 0, tomo1, tomo2) for ell in ell_values]
        
        # Perform integration using Simpson's rule
        result = integrate.simpson(integrand_values, x=ell_values)
        
        return result/(2*np.pi)
    
    
    def ximinus(self, theta, nell, tomo1, tomo2):
        """
        Computes ximinus at theta for given tomography bins.

        Parameters
        ----------
        theta : float
            separarion of xi-.
        nell :
            number of ell for integrtion
        tomo1 : int
            First tomography bin index.
        tomo2 : int
            Second tomography bin index.

        Returns
        -------
        float
            Value of the xi-.
        """
            # Set up the integration range and points for Simpson's rule
        ell_values = np.geomspace(10**np.min(self.ell_values), 10**np.max(self.ell_values), nell)  # 1000 points for accuracy
        integrand_values = [self.integrand_xiplus(ell, theta, 4, tomo1, tomo2) for ell in ell_values]
        
        # Perform integration using Simpson's rule
        result = integrate.simpson(integrand_values, x=ell_values)
        
        return result/(2*np.pi)
