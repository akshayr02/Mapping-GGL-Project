import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.special import j0, jn_zeros
from scipy.special import roots_legendre
from scipy import integrate
from tqdm import tqdm  # For progress bars


from .utility import *
from .constants import *

@njit
def integrand_En_helper(ell, pkappas, ell_values, W):
    """
    Helper function to compute the integrand for En.

    Parameters
    ----------
    ell : float
        The ell value.
    pkappas : numpy.ndarray
        Precomputed power spectrum values.
    ell_values : numpy.ndarray
        Array of ell values.
    W : float
        The value of W function at ell.

    Returns
    -------
    float
        The result of the integrand computation.
    """
    # pk = custom_interpolate_1d(ell_values, pkappas, ell)
    pk = 10**custom_interpolate_1d(ell_values, pkappas, np.log10(ell))  
    # print(ell, pk, W)
    return ell * pk * W

class cosebis:
    """
    Class for calculating COSEBIs (Complete Orthogonal Sets of E/B-mode Integrals) for weak gravitational lensing
    from power spectrum. Based on the prescription in Asgari et al. (2012) and Schneider et al. (2010)

    Parameters
    ----------
    thMin : float
        Minimum angular scale in arcminutes.
    thMax : float
        Maximum angular scale in arcminutes.
    nMax : int
        Maximum COSEBI order.
    limber : object
        Limber approximation object for power spectrum computation.
    cosebitype : str, optional, default="log"
        Type of COSEBIs to compute (only "log" is currently implemented).
    fileWs : str, optional
        Path to file storing precomputed Ws (filter functions).
    fileTs : str, optional
        Path to file storing precomputed Ts (Polynomial functions).
    folderRootsNorms : str, optional
        Folder containing roots and norms for T function calculations.
    verbose : bool, optional
        Control amount of output
    """
    def __init__(self, thMin: float, thMax: float, nMax: int, 
                 limber : object, cosebitype="log",
                 fileWs="None", fileTs="None", 
                 folderRootsNorms="None", verbose=True,
                 nell = 256, ell_min = 1, ell_max = 20000):
        
        
        self.thMin=thMin # Minimum angular scale in arcmin
        self.thMax=thMax # Maximum angular scale in arcmin
        self.nMax=nMax # Maximum cosebi order
        self.cosebitype=cosebitype # Cosebi type

        self.thMinRad=np.deg2rad(thMin/60) # Minimum angular scale in rad
        self.thMaxRad=np.deg2rad(thMax/60) # Maximum angular scale in rad

        self.ellMin=1 # Minimum ell
        self.ellMax=100*np.pi*self.nMax/self.thMaxRad # Maximum ell

        self.limber=limber
        self.Ntomos = self.limber.Ntomo

        self.verbose=verbose

        if self.nMax>20:
            raise ValueError("I only have the roots and norm for the COSEBI filter functions for n <= 20")
        
        if self.cosebitype!="log":
            raise NotImplementedError("Only log cosebis are implemented. Change cosebitype to 'log'")

        if os.path.exists(fileWs):
            print(f"Loading Ws from {fileWs}")
            self.loadWs(fileWs)
        else:
            if self.verbose:
                print(f"File for Ws {fileWs} does not exist. Will calculate Ws.")

            if os.path.exists(fileTs):
                if self.verbose:
                    print(f"Loading Ts from {fileTs}")
                self.loadTs(fileTs)
            else:
                if self.verbose:
                    print(f"File for Ts {fileTs} does not exist. Will calculate Ts.")
    
                if (not os.path.exists(folderRootsNorms)):
                    raise FileNotFoundError(f"Could not find folder {folderRootsNorms} for T roots and norms. Please provide this folder!")
                
                self.calculateTs(folderRootsNorms, filenameTs=fileTs)
    
                if self.verbose:
                    print("Finished calculating Ts")
            
            if self.verbose:
                print("Calculating Ws")
    
            self.calculateWs(filenameWs=fileWs)

            if self.verbose:
                print("Finished calculating Ts")

        if self.verbose:
            print("Precomputing projected powerspectrum")
    
        self.precompute_pkappa_allTomos(ell_min= ell_min, ell_max = ell_max, nell = nell)

    def readTRootsNorms(self, folderRootsNorms):
        """
        Reads the roots and normalization values for T functions from files.

        Parameters
        ----------
        folderRootsNorms : str
            Path to folder containing root and normalization files for T functions.
        """
        fn_norms=f"{folderRootsNorms}/Normalization_{self.thMin:.2f}-{self.thMax:.2f}.table"

        if self.verbose:
            print(f"Reading norms from {fn_norms}")

        self.Tnorms=readTableFile(fn_norms)

        fn_roots=f"{folderRootsNorms}/Root_{self.thMin:.2f}-{self.thMax:.2f}.table"
        
        if self.verbose:
            print(f"Reading roots from {fn_roots}")

        self.Troots=readTableFile(fn_roots)


    def calculateTs(self, folderRootsNorms, filenameTs="None"):
        """
        Calculates T functions based on roots and norms and saves them if needed.

        Parameters
        ----------
        folderRootsNorms : str
            Path to folder containing roots and normalization files.
        filenameTs : str, optional, default="None"
            File path to save the computed T functions. If "None" Ts are not saved
        """
        self.readTRootsNorms(folderRootsNorms)

        thetas_rad=np.geomspace(self.thMinRad, self.thMaxRad, 10000, dtype=np.longdouble)
        zs=np.log(thetas_rad/self.thMinRad)
        self.Ts={}

        for n in range(1, self.nMax):
            if self.verbose:
                print(f"Calculating T_{n}")

            roots=self.Troots[n]
            norm=self.Tnorms[n]

            t=norm*np.ones_like(thetas_rad)
            for i in range(0, n+1):
                t*=(zs-roots[i])
            

            # self.Ts[n] = CubicSpline(thetas_rad, t)
            self.Ts[n]  = interpolate.interp1d(x=thetas_rad,y=t,kind='linear')
            

        if filenameTs!="None":
            if self.verbose:
                print(f"Saving Ts to {filenameTs}")
            self.saveTs(filenameTs)



    def saveTs(self, filenameTs):
        """
        Saves the T function splines to a specified file.

        Parameters
        ----------
        filenameTs : str
            Path to save the T splines.
        """
        save_splines(self.Ts, filenameTs)

    def loadTs(self, filenameTs):
        """
        Loads T function splines from a specified file.

        Parameters
        ----------
        filenameTs : str
            Path to load the T splines.
        """
        self.Ts=load_splines(filenameTs)



    def calculateSingleW(self, ell, n, n_intervals=100):
        """
        Calculates a single W function (filter function) for a given ell and order n.

        Parameters
        ----------
        ell : float
            Ell value at which to evaluate the W function.
        n : int
            COSEBI order.
        n_intervals : int, optional, default=100
            Number of intervals for Gaussian quadrature.

        Returns
        -------
        float
            Calculated W function value.
        """

        # Step 1: Define the integration function for a single interval
        def integrate_interval(x0, x1, ell, T_spline, n_points=n_intervals//10):
            """
            Integrate over a single interval [x0, x1] using Gaussian quadrature.
            """
            # Gaussian quadrature points and weights for n_points
            xi, wi = roots_legendre(n_points)
            
            # Transform xi from [-1, 1] to [x0, x1]
            theta_i = 0.5 * (x1 - x0) * xi + 0.5 * (x1 + x0)
            
            # Calculate the integral for this interval
            integral = np.sum(wi * T_spline(theta_i) * j0(ell * theta_i) * theta_i) * 0.5 * (x1 - x0)
            return integral
        
        l_thresh=np.pi*n/self.thMaxRad # Threshold ell value to switch between methods

        # Step 2: Handle the small-ell case
        if ell < l_thresh:
            # Use a single Gaussian quadrature over the entire range [a, b]
            xi, wi = roots_legendre(n_intervals)
            theta_i = 0.5 * (self.thMaxRad - self.thMinRad) * xi + 0.5 * (self.thMaxRad + self.thMinRad)
            integral = np.sum(wi * self.Ts[n](theta_i) * j0(ell * theta_i) * theta_i) * 0.5 * (self.thMaxRad - self.thMinRad)
            return integral
        # Step 3: Handle the large-ell case with piecewise integration
        else:
            # Find the zeros of J0 to use as integration limits for each interval
            num_roots = int(np.ceil((self.thMaxRad - self.thMinRad) * ell / np.pi))
            j0_zeros = jn_zeros(0, num_roots) / ell  # Scaling roots by ell

            # Only keep zeros within [a, b]
            j0_zeros = j0_zeros[(j0_zeros > self.thMinRad) & (j0_zeros < self.thMaxRad)]
            
            # Integrate over each segment defined by the zeros of J0
            integral = 0.0
            current_thMin = self.thMinRad
            for zero in j0_zeros:
                integral += integrate_interval(current_thMin, zero, ell, self.Ts[n])
                current_thMin = zero
            integral += integrate_interval(current_thMin, self.thMaxRad, ell, self.Ts[n])  # Final segment
            
            return integral


    def calculateWs(self, filenameWs="None"):
        """
        Calculates and stores W functions for a range of ell values.

        Parameters
        ----------
        filenameWs : str, optional, default="None"
            Path to save the computed W splines.
        """

        ells=np.geomspace(self.ellMin, self.ellMax, 10000)

        self.Ws={}

        for n in range(1, self.nMax):
            if self.verbose:
                print(f"Calculating W_{n}")

            ws=np.zeros_like(ells)

            for i, ell in enumerate(ells):
                ws[i] = self.calculateSingleW(ell, n)

            # self.Ws[n] = CubicSpline(ells, ws)
            self.Ws[n]  = interpolate.interp1d(x=ells,y=ws,kind='linear')
            

        if filenameWs!="None":
            if self.verbose:
                print(f"Saving Ws to {filenameWs}")
            self.saveWs(filenameWs)


    def saveWs(self, filenameWs):
        """
        Saves W function splines to a specified file.

        Parameters
        ----------
        filenameWs : str
            Path to save the W splines.
        """
        save_splines(self.Ws, filenameWs)


    def loadWs(self, filenameWs):
        """
        Loads W function splines from a specified file.

        Parameters
        ----------
        filenameWs : str
            Path to load the W splines.
        """
        self.Ws=load_splines(filenameWs)



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


    def integrand_En(self, ell, n, tomo1, tomo2):
        """
        Computes the integrand for the En COSEBI moment.

        Parameters
        ----------
        ell : float
            Ell value for integration.
        n : int
            COSEBI order.
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
        
        return integrand_En_helper(ell, self.pkappas[combi], self.ell_values, self.Ws[n](ell))


    def En(self, n, tomo1, tomo2):
        """
        Computes the nth COSEBI moment (En) for given tomography bins.

        Parameters
        ----------
        n : int
            COSEBI order.
        tomo1 : int
            First tomography bin index.
        tomo2 : int
            Second tomography bin index.

        Returns
        -------
        float
            Value of the En COSEBI moment.
        """
            # Set up the integration range and points for Simpson's rule
        ell_values = np.geomspace(self.ellMin, self.ellMax, 1000)  # 1000 points for accuracy
        integrand_values = [self.integrand_En(ell, n, tomo1, tomo2) for ell in ell_values]
        
        # Perform integration using Simpson's rule
        result = integrate.simpson(integrand_values, x=ell_values)
        
        return result/(2*np.pi)
