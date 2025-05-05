import numpy as np

from scipy import integrate
from numba import njit
from .constants import *
from tqdm import tqdm  # For progress bars

from .utility import *

@njit
def uHat(eta):
    """
    Computes the aperture statistics filter function \( \hat{u}(\eta) = 0.5 \cdot \eta^2 \cdot \exp(-0.5 \cdot \eta^2) \).
    This is the aperture filter from Crittenden 2002

    Parameters
    ----------
    eta : float or numpy.ndarray
        Input value(s) for the function. Corresponds to theta_ap * ell. dimensionless

    Returns
    -------
    float or numpy.ndarray
        The result of \( \hat{u}(\eta) \) computed for the given `eta`.

    """
    tmp = 0.5 * eta**2
    return tmp * np.exp(-tmp)


@njit
def integrand_Map3_helper(ell1, ell2, phi, theta1, theta2, theta3, bkappas, ell1_values, ell2_values, phi_values):
    """
    Computes the integrand for the third-order aperture mass statistic (Map3) using 
    trilinear interpolation on a 3D grid of `bkappa` values.

    Parameters
    ----------
    ell1 : float
        The first angular frequency value.
    ell2 : float
        The second angular frequency value.
    phi : float
        Angle between `ell1` and `ell2`, in radians.
    theta1 : float
        First aperture radius, in radians.
    theta2 : float
        Second aperture radius, in radians.
    theta3 : float
        Third aperture radius, in radians
    bkappas : numpy.ndarray
        3D grid of `bkappa` values defined over (`ell1`, `ell2`, `phi`) coordinates.
    ell1_values : numpy.ndarray
        1D array of available `ell1` values in `bkappas`.
    ell2_values : numpy.ndarray
        1D array of available `ell2` values in `bkappas`.
    phi_values : numpy.ndarray
        1D array of available `phi` values in `bkappas`.

    Returns
    -------
    float
        The computed value of the Map3 integrand for the given parameters.

    Notes
    -----
    This function applies trilinear interpolation to estimate `bkappa` at (`ell1`, `ell2`, `phi`) 
    and then calculates the integrand by multiplying interpolated `bkappa` with scaled values of `uHat` 
    functions for each aperture radius. The function is optimized with `@njit` for performance.
    """


    ell3 = np.sqrt(ell1**2 + ell2**2 + 2 * ell1 * ell2 * np.cos(phi))

    
    # Interpolate bkappa
    bk=10**custom_interpolate(ell1_values, ell2_values, phi_values, bkappas, np.log10(ell1), np.log10(ell2), phi)
    # bk=custom_interpolate(ell1_values, ell2_values, phi_values, bkappas, ell1, ell2, phi)

    return ell1 * ell2 * bk * uHat(ell1 * theta1) * uHat(ell2 * theta2) * uHat(ell3 * theta3)


@njit
def integrand_Map2_helper(ell, theta, pkappas, ell_values):
    """
    Computes the integrand for the second-order aperture mass statistic (Map2) using 
    linear interpolation on a 1D grid of `pkappa` values.

    Parameters
    ----------
    ell : float
        Angular frequency value.
    theta : float
        Aperture radius in radians
    pkappas : numpy.ndarray
        1D array of `pkappa` values corresponding to each `ell` in `ell_values`.
    ell_values : numpy.ndarray
        1D array of `ell` values associated with `pkappas`.

    Returns
    -------
    float
        The computed value of the Map2 integrand for the given parameters.

    Notes
    -----
    This function linearly interpolates `pkappa` at the specified `ell` value and 
    calculates the integrand as the square of `uHat(ell * theta)` scaled by `pkappa`.
    The function is optimized with `@njit` for performance.
    """


    # Interpolate bkappa
    pk = 10**custom_interpolate_1d(ell_values, pkappas, np.log10(ell))    
    # pk = custom_interpolate_1d(ell_values, pkappas, ell)    

    return ell * pk * uHat(ell * theta) **2



class apertureStatistics:
    """
    Computes aperture mass statistics, including second-order (Map2) and third-order (Map3) statistics,
    using precomputed bispectrum and power spectrum data on configurable grids. Supports calculations 
    for multiple tomography bins and can account for intrinsic alignments.

    Parameters
    ----------
    limber : object
        An instance of a Limber class that provides bispectrum and power spectrum computations.
    precomputeBispectra : bool, optional, default=True
        Whether to precompute bispectra on a specified grid for all tomography combinations.
    precomputePowerspectra : bool, optional, default=True
        Whether to precompute power spectra on a specified grid for all tomography combinations.

    Attributes
    ----------
    limber : object
        Reference to the provided Limber instance.
    Ntomos : int
        Number of tomography bins in the analysis.
    calculateIA : bool
        Whether intrinsic alignments are considered in the calculations.
    bkappas : dict
        Precomputed bispectrum values for each tomography combination.
    pkappas : dict
        Precomputed power spectrum values for each tomography combination.
    ell1_values, ell2_values, phi_values : numpy.ndarray
        Grid values used for bispectrum interpolation.
    ell_values : numpy.ndarray
        Grid values used for power spectrum interpolation.
    """

    def __init__(self, limber, precomputeBispectra=True, precomputePowerspectra=True, z_points=200, nell = 20, nphi = 20, ell_min = 1, ell_max = 20000, phi_min = 0, phi_max = np.pi) -> None:
        """
        Initializes apertureStatistics, setting up tomography bin configurations and 
        optionally precomputing bispectrum and power spectrum grids.
        """

        self.limber = limber
        self.Ntomos = self.limber.Ntomo
        self.z_points = z_points

        if limber.AIA==None:
            print("No AIA specified, computing for only cosmic shear")
            self.calculateIA=False
        else:
            print(f"Using A_IA={limber.AIA} and eta={limber.eta}")
            self.calculateIA=True
        
        if precomputeBispectra:
            print("Starting precomputation of bispectra")
            
            self.precompute_bkappa_allTomos(ell_min= ell_min, ell_max = ell_max, nell = nell, nphi = nphi, phi_min = phi_min, phi_max=phi_max)
            

            print("Finished precomputation of bispectra")

        if precomputePowerspectra:
            print("Starting precomputation of Powerspectrum")

            self.precompute_pkappa_allTomos(ell_min= ell_min, ell_max = ell_max, nell = nell)

            print("Finished precomputing powerspectrum")


    def precompute_bkappa_allTomos(self, ell_min = 1, ell_max = 20000, nell = 20, phi_min=0, phi_max=np.pi, nphi = 20):
        """
        Precomputes the bispectrum (bkappa) on a 3D grid for all unique tomography combinations.

        Parameters
        ----------
        ell1_range : list, optional, default=[1, 20000]
            Range of `ell1` values for bispectrum grid.
        ell2_range : list, optional, default=[1, 20000]
            Range of `ell2` values for bispectrum grid.
        phi_range : list, optional, default=[0, 2*np.pi]
            Range of `phi` values for bispectrum grid. In radians.

        Notes
        -----
        Saves the computed `bkappa` grids in the `bkappas` dictionary, keyed by tomography combination.
        """

        self.bkappas={}

        self.ell1_values = np.logspace(np.log10(ell_min), np.log10(ell_max), nell)  # Logarithmically spaced
        self.ell2_values = np.logspace(np.log10(ell_min), np.log10(ell_max), nell)  # Logarithmically spaced
        self.phi_values = np.linspace(phi_min, phi_max+0.001, nphi)  # Linearly spaced the plus 0.001 is to avoid inf/nan at exactly pi


        for tomo1 in range(self.Ntomos):
            for tomo2 in range(tomo1, self.Ntomos):
                for tomo3 in tqdm(range(tomo2, self.Ntomos), desc=f"preparing tomo {tomo1}-{tomo2}"):
                    combi=f"{tomo1}-{tomo2}-{tomo3}"
                    ## making use of the symmetrie that B is symetric in from 0 pi and and then from pi two 2pi.
                    self.bkappas[combi] = self.precompute_bkappa(self.ell1_values, self.ell2_values, self.phi_values, tomo1, tomo2, tomo3)
           
        self.ell1_values = np.log10(self.ell1_values)
        self.ell2_values = np.log10(self.ell2_values) 


    def precompute_bkappa(self, ell1_values, ell2_values, phi_values, tomo1, tomo2, tomo3):
        """
        Computes the bispectrum grid for a specific tomography combination.

        Parameters
        ----------
        ell1_values : numpy.ndarray
            1D array of `ell1` values for the grid.
        ell2_values : numpy.ndarray
            1D array of `ell2` values for the grid.
        phi_values : numpy.ndarray
            1D array of `phi` values for the grid. In radians.
        tomo1, tomo2, tomo3 : int
            Tomography bin indices.

        Returns
        -------
        numpy.ndarray
            3D array of computed `bkappa` values for the specified tomography combination.
        """


        # Compute bkappa values on the grid
        ## making use of the symetrie that B(ell1,ell2,phi)=B(ell2,ell1,phi)
        bkappa_grid = np.zeros((len(ell1_values), len(ell2_values), len(phi_values)))
        for i in tqdm(np.arange(0,len(ell1_values)),desc='ell integration'):
            for j in np.arange(i,len(ell2_values)):
                for k, phi in enumerate(phi_values):
                    ell1, ell2 = ell1_values[i], ell2_values[j]
                    ell3 = np.sqrt(ell1**2 + ell2**2 + 2 * ell1 * ell2 * np.cos(phi))
                    bkappa_grid[i, j, k] = self.limber.bispectrum_projected_simps(ell1, ell2, ell3, tomo1, tomo2, tomo3, z_points=self.z_points)
                    bkappa_grid[j, i, k] = bkappa_grid[i, j, k]

        bkappa_grid[np.where(bkappa_grid<=0)]=np.min(bkappa_grid[np.where(bkappa_grid>0)])/10
        bkappa_grid = np.log10(bkappa_grid)
        bkappa_grid[np.where(np.isnan(bkappa_grid))]=0.0
        bkappa_grid[np.where(np.isinf(bkappa_grid))]=0.0
        
        return bkappa_grid
    

    def integrand_Map3(self, ell1, ell2, phi, theta1, theta2, theta3, tomo1, tomo2, tomo3):
        """
        Calculates the integrand for the third-order aperture mass (Map3) statistic.

        Parameters
        ----------
        ell1 : float
            First angular frequency.
        ell2 : float
            Second angular frequency.
        phi : float
            Angle between `ell1` and `ell2`. In Radians.
        theta1, theta2, theta3 : float
            Aperture radii.
        tomo1, tomo2, tomo3 : int
            Tomography bin indices.

        Returns
        -------
        float
            The value of the Map3 integrand at the specified inputs.
        """

        combi=f"{tomo1}-{tomo2}-{tomo3}"

        return integrand_Map3_helper(ell1, ell2, phi, theta1, theta2, theta3, self.bkappas[combi], 
                                     self.ell1_values, self.ell2_values, self.phi_values)


    def Map3(self, theta1, theta2, theta3, tomo1=0, tomo2=0, tomo3=0):
        """
        Computes the third-order aperture mass statistic (Map3) by integrating the bispectrum.

        Parameters
        ----------
        theta1, theta2, theta3 : float
            Aperture radii in radians.
        tomo1, tomo2, tomo3 : int, optional
            Tomography bin indices, default is 0.

        Returns
        -------
        float
            Computed value of the Map3 statistic for the specified apertures and tomography bins.
        """

        # Define the integration limits
        ell1_limits = [10**np.min(self.ell1_values), 10**np.max(self.ell1_values)]
        ell2_limits = [10**np.min(self.ell2_values), 10**np.max(self.ell2_values)]
        phi_limits = [0, np.pi]
        
        # Wrapper function to pass the additional parameters
        def integrand_wrapper(ell1, ell2, phi):
            return self.integrand_Map3(ell1, ell2, phi, theta1, theta2, theta3, tomo1, tomo2, tomo3)
        
        # Integration options
        opts = {'epsrel': Map3_epsrel, 'limit': 15}

        # Perform the integration
        result, error = integrate.nquad(integrand_wrapper, [ell1_limits, ell2_limits, phi_limits], opts=opts, full_output = False)
        
        # Prefactor
        prefactor = 1. / 8 / np.pi**3

        return 2 * result * prefactor
    
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
        pkappa_grid[np.where(pkappa_grid<=0)]=np.min(pkappa_grid[np.where(pkappa_grid>0)])/10
        pkappa_grid = np.log10(pkappa_grid)
        pkappa_grid[np.where(np.isnan(pkappa_grid))]=0.0
        pkappa_grid[np.where(np.isinf(pkappa_grid))]=0.0

        return pkappa_grid


    def integrand_Map2(self, ell, theta, tomo1, tomo2):
        """
        Calculates the integrand for the second-order aperture mass (Map2) statistic.

        Parameters
        ----------
        ell : float
            Angular frequency.
        theta : float
            Aperture radius in radians.
        tomo1, tomo2 : int
            Tomography bin indices.

        Returns
        -------
        float
            The value of the Map2 integrand at the specified inputs.
        """
        combi=f"{tomo1}-{tomo2}"

        return integrand_Map2_helper(ell, theta, self.pkappas[combi], self.ell_values)


    def Map2(self, theta, tomo1, tomo2):
        """
        Calculates the second-order aperture mass (Map2) statistic.

        Parameters
        ----------
        theta : float
            Aperture radius in radians.
        tomo1, tomo2 : int
            Tomography bin indices.

        Returns
        -------
        float
            The value of the Map2 at the specified inputs.
        """

        # Define the integration limits
        ell_limits = [10**np.min(self.ell_values), 10**np.max(self.ell_values)]
        
        # Wrapper function to pass the additional parameters
        def integrand_wrapper(ell):
            return self.integrand_Map2(ell, theta, tomo1, tomo2)
        
        # Integration options
        opts = {'epsrel': Map2_epsrel, 'limit': 15}

        # Perform the integration
        result, error = integrate.nquad(integrand_wrapper, [ell_limits], opts=opts)

        # Prefactor
        prefactor = 1. / 2 / np.pi

        return result * prefactor
    