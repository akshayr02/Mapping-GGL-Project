import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
import scipy.integrate as integrate
from .cosmopower_NN import cosmopower_NN 
from numba import njit
import sys
import baccoemu


from .constants import *
import warnings


c_over_H0=2997.92

@njit(fastmath=True)
def bispec_(k1, k2, k3, z, D1_, r_sigma_, n_eff_, sigma8, nS, h, OmC, OmM, OmB, OmL, w, norm_P):
        """
        Calculate the bihalofit bispectrum for a given set of wavenumbers and cosmological parameters.

        The bispectrum is computed as the sum of the one-halo and three-halo contributions.
        The one-halo term is calculated using parameters derived from the input variables, 
        while the three-halo term uses the F2 kernel for the bias and power spectrum.

        Parameters:
        ----------
        k1, k2, k3 : float
            Wavenumbers corresponding to the three different modes of the bispectrum. [h/Mpc]
        z : float
            Redshift at which the bispectrum is evaluated.
        D1_ : float
            Linear growth factor at the specified redshift.
        r_sigma_ : float
            Scale factor related to the nonlinear evolution of density perturbations.
        n_eff_ : float
            Effective spectral index of the density fluctuations.
        sigma8 : float
            Normalization of the density fluctuations on a scale of 8 h^-1 Mpc.
        nS : float
            Scalar spectral index.
        h : float
            Dimensionless Hubble parameter (H0/100 km/s/Mpc).
        OmC, OmM, OmB, OmL : float
            Density parameters for cold dark matter, total matter, baryonic matter, 
            and dark energy, respectively.
        w : float
            Equation of state parameter for dark energy.
        norm_P : float
            Normalization factor for the power spectrum.

        Returns:
        -------
        float
            The computed bispectrum value at the specified wavenumbers and cosmological parameters.

        Notes:
        -----
        This function uses Numba's just-in-time compilation for fast execution. 
        It computes dimensionless wavenumbers, sorts them, and applies 
        cosmological formulas to calculate the bispectrum contributions from 
        both one-halo and three-halo terms.
        """


        q = np.array(
            [0.0, k1 * r_sigma_, k2 * r_sigma_, k3 * r_sigma_]
        )  # dimensionless wavenumbers

        # sorting q[i] so that q[1] >= q[2] >= q[3]
        q[1:4] = sorted(q[1:4], reverse=True)
        r1 = q[3] / q[1]
        r2 = (q[2] + q[3] - q[1]) / q[1]

        q[1], q[2], q[3] = k1 * r_sigma_, k2 * r_sigma_, k3 * r_sigma_
        logsigma8z = np.log10(D1_ * sigma8)

        # 1-halo term parameters in Eq.(B7)
        an = 10 ** (
            -2.167
            - 2.944 * logsigma8z
            - 1.106 * logsigma8z**2
            - 2.865 * logsigma8z**3
            - 0.310 * r1 ** (10 ** (0.182 + 0.57 * n_eff_))
        )
        bn = 10 ** (
            -3.428 - 2.681 * logsigma8z + 1.624 * logsigma8z**2 - 0.095 * logsigma8z**3
        )
        cn = 10 ** (0.159 - 1.107 * n_eff_)
        alphan = 10 ** (
            -4.348
            - 3.006 * n_eff_
            - 0.5745 * n_eff_**2
            + 10 ** (-0.9 + 0.2 * n_eff_) * r2**2
        )
        if alphan > 1.0 - (2.0 / 3.0) * nS:
            alphan = 1.0 - (2.0 / 3.0) * nS
        betan = 10 ** (
            -1.731
            - 2.845 * n_eff_
            - 1.4995 * n_eff_**2
            - 0.2811 * n_eff_**3
            + 0.007 * r2
        )

        # 1-halo term bispectrum in Eq.(B4)
        BS1h = 1.0
        for i in range(1, 4):
            BS1h *= (
                1.0
                / (an * q[i] ** alphan + bn * q[i] ** betan)
                / (1.0 + 1.0 / (cn * q[i]))
            )

        # 3-halo term parameters in Eq.(B9)
        fn = 10 ** (-10.533 - 16.838 * n_eff_ - 9.3048 * n_eff_**2 - 1.8263 * n_eff_**3)
        gn = 10 ** (2.787 + 2.405 * n_eff_ + 0.4577 * n_eff_**2)
        hn = 10 ** (-1.118 - 0.394 * n_eff_)
        mn = 10 ** (-2.605 - 2.434 * logsigma8z + 5.71 * logsigma8z**2)
        nn = 10 ** (-4.468 - 3.08 * logsigma8z + 1.035 * logsigma8z**2)
        mun = 10 ** (
            15.312 + 22.977 * n_eff_ + 10.9579 * n_eff_**2 + 1.6586 * n_eff_**3
        )
        nun = 10 ** (1.347 + 1.246 * n_eff_ + 0.4525 * n_eff_**2)
        pn = 10 ** (0.071 - 0.433 * n_eff_)
        en = 10 ** (-0.632 + 0.646 * n_eff_)

        PSE = np.zeros(4)
        for i in range(1, 4):
            PSE[i] = (1.0 + fn * q[i] ** 2) / (
                1.0 + gn * q[i] + hn * q[i] ** 2
            ) * D1_**2 * linear_pk_(q[i] / r_sigma_, h, OmC, OmM, OmB, norm_P, nS) + 1.0 / (
                mn * q[i] ** mun + nn * q[i] ** nun
            ) / (
                1.0 + (pn * q[i]) ** -3
            )

        # 3-halo term bispectrum in Eq.(B5)
        BS3h = 2.0 * (
            F2_(k1, k2, k3, z, D1_, r_sigma_, sigma8, OmM, OmL, w) * PSE[1] * PSE[2]
            + F2_(k2, k3, k1, z, D1_, r_sigma_, sigma8, OmM, OmL, w) * PSE[2] * PSE[3]
            + F2_(k3, k1, k2, z, D1_, r_sigma_, sigma8, OmM, OmL, w) * PSE[3] * PSE[1]
        )
        for i in range(1, 4):
            BS3h *= 1.0 / (1.0 + en * q[i])

        return BS1h + BS3h

@njit(fastmath=True)
def linear_pk_(k, h, OmC, OmM, OmB, norm_P, nS):
        """
        Compute the linear matter power spectrum P(k) using the Eisenstein & Hu 
        model without wiggles.

        Parameters:
        ----------
        k : float
            Wavenumber in [h/Mpc].
        h : float
            Dimensionless Hubble parameter (H0/100 km/s/Mpc).
        OmC : float
            Density parameter for cold dark matter.
        OmM : float
            Total matter density parameter.
        OmB : float
            Baryonic matter density parameter.
        norm_P : float
            Normalization factor for the power spectrum.
        nS : float
            Scalar spectral index.

        Returns:
        -------
        float
            The linear matter power spectrum evaluated at the specified wavenumber.

        Notes:
        -----
        This function performs a series of calculations to derive the linear power 
        spectrum from cosmological parameters based on the Eisenstein & Hu framework.
        It includes unit conversion for wavenumbers and calculates various 
        parameters related to the background cosmology.
        """

        k *= h  # unit conversion from [h/Mpc] to [1/Mpc]

        fc = OmC / OmM
        fb = OmB / OmM
        theta = 2.728 / 2.7
        pc = 0.25 * (5.0 - np.sqrt(1.0 + 24.0 * fc))

        omh2 = OmM * h * h
        ombh2 = OmB * h * h

        zeq = 2.5e4 * omh2 / theta**4
        b1 = 0.313 * omh2**-0.419 * (1.0 + 0.607 * omh2**0.674)
        b2 = 0.238 * omh2**0.223
        zd = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * ombh2**b2)
        yd = (1.0 + zeq) / (1.0 + zd)
        sh = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)

        alnu = (
            fc
            * (5.0 - 2.0 * pc)
            / 5.0
            * (1.0 - 0.553 * fb + 0.126 * fb**3)
            * (1.0 + yd) ** -pc
            * (1.0 + 0.5 * pc * (1.0 + 1.0 / (7.0 * (3.0 - 4.0 * pc))) / (1.0 + yd))
        )
        geff = omh2 * (
            np.sqrt(alnu) + (1.0 - np.sqrt(alnu)) / (1.0 + (0.43 * k * sh) ** 4)
        )
        qeff = k / geff * theta * theta

        L = np.log(2.718281828 + 1.84 * np.sqrt(alnu) * qeff / (1.0 - 0.949 * fb))
        C = 14.4 + 325.0 / (1.0 + 60.5 * qeff**1.11)

        delk = (
            norm_P**2
            * (k * 2997.9 / h) ** (3.0 + nS)
            * (L / (L + C * qeff**2)) ** 2
        )
        pk = 2.0 * np.pi * np.pi / (k**3) * delk

        return h**3 * pk

@njit(fastmath=True)
def F2_(k1, k2, k3, z, D1, r_sigma, sigma8, OmM, OmL, w):
        """
        Compute the F2 kernel for the bihalofit bispectrum for the given wavenumbers and redshift.

        Parameters:
        ----------
        k1, k2, k3 : float
            Wavenumbers corresponding to the three modes of the F2 kernel.
        z : float
            Redshift at which the kernel is evaluated.
        D1 : float
            Linear growth factor at the specified redshift.
        r_sigma : float
            Scale factor related to the nonlinear evolution of density perturbations.
        sigma8 : float
            Normalization of the density fluctuations on a scale of 8 h^-1 Mpc.
        OmM : float
            Total matter density parameter.
        OmL : float
            Dark energy density parameter.
        w : float
            Equation of state parameter for dark energy.

        Returns:
        -------
        float
            The computed F2 kernel value.
        """

        q = k3 * r_sigma

        logsigma8z = np.log10(D1 * sigma8)
        a = 1.0 / (1.0 + z)
        omz = OmM / (
            OmM + OmL * a ** (-3.0 * w)
        )  # Omega matter at z

        dn = 10.0 ** (-0.483 + 0.892 * logsigma8z - 0.086 * omz)

        return F2_tree_(k1, k2, k3) + dn * q

@njit(fastmath=True)
def F2_tree_(k1, k2, k3):
        """
        Calculate the tree-level F2 kernel based on the wavenumbers.

        Parameters:
        ----------
        k1, k2, k3 : float
            Wavenumbers corresponding to the three modes of the F2 kernel.

        Returns:
        -------
        float
            The computed tree-level F2 kernel value.

        Notes:
        -----
        This function computes the tree-level contribution to the F2 kernel using 
        the wavenumbers and their geometric relations, which is used in 
        calculating the bispectrum.
        """
        costheta12 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2) / (k1 * k2)
        return (
            (5.0 / 7.0)
            + 0.5 * costheta12 * (k1 / k2 + k2 / k1)
            + (2.0 / 7.0) * costheta12 * costheta12
        )
        
@njit(fastmath=True)       
def bispec_tree_(k1, k2, k3, D1, h, OmC, OmM, OmB, norm_P, nS):
        """
        Compute the tree-level bispectrum for the given wavenumbers and cosmological parameters.

        Parameters:
        ----------
        k1, k2, k3 : float
            Wavenumbers corresponding to the three modes of the bispectrum.
        D1 : float
            Linear growth factor at the specified redshift.
        h : float
            Dimensionless Hubble parameter (H0/100 km/s/Mpc).
        OmC : float
            Density parameter for cold dark matter.
        OmM : float
            Total matter density parameter.
        OmB : float
            Baryonic matter density parameter.
        norm_P : float
            Normalization factor for the power spectrum.
        nS : float
            Scalar spectral index.

        Returns:
        -------
        float
            The computed tree-level bispectrum value.

        Notes:
        -----
        This function calculates the tree-level bispectrum by combining contributions 
        from the F2 kernels and the linear power spectrum for each pair of wavenumbers.
        It uses the linear growth factor to account for the growth of structure over time.
        """

        # Compute the tree-level bispectrum
        return (
            (D1**4)
            * 2.0
            * (
                F2_tree_(k1, k2, k3) * linear_pk_(k1, h, OmC, OmM, OmB, norm_P, nS) * linear_pk_(k2, h, OmC, OmM, OmB, norm_P, nS)
                + F2_tree_(k2, k3, k1) * linear_pk_(k2, h, OmC, OmM, OmB, norm_P, nS) * linear_pk_(k3, h, OmC, OmM, OmB, norm_P, nS)
                + F2_tree_(k3, k1, k2) * linear_pk_(k3, h, OmC, OmM, OmB, norm_P, nS) * linear_pk_(k1, h, OmC, OmM, OmB, norm_P, nS)
            )
        )



@njit(fastmath=True)
def lgr_func_(j, la, y, OmM, OmL, w):
    """Computes the linear growth rate differential equation.

    Args:
        j (int): The index for the derivative to compute (0 or 1).
        la (float): The logarithm of the scale factor.
        y (ndarray): The current values of the solution.
        OmM (float): The matter density parameter.
        OmL (float): The dark energy density parameter.
        w (float): The equation of state parameter for dark energy.

    Returns:
        float: The computed derivative or the value of y[1] based on j.
    
    Raises:
        ValueError: If j is not a valid index (not 0 or 1).
    """
    if(j==0): return y[1]
      
    a=np.exp(la)
    g=-0.5*(5.*OmM+(5.-3*w)*OmL*pow(a,-3.*w))*y[1]-1.5*(1.-w)*OmL*pow(a,-3.*w)*y[0]
    g=g/(OmM+OmL*pow(a,-3.*w))
    if(j==1): return g
    else:
        raise ValueError("lgr_func: j not a valid value.")


@njit(fastmath=True)
def lgr_(z, OmM, OmL, w, eps=1e-4):
    """Calculates the linear growth rate of the universe.

    Uses the Runge-Kutta method to integrate the differential equation
    governing the growth of structures in an expanding universe.

    Args:
        z (float): The redshift at which to evaluate the growth rate.
        OmM (float): The matter density parameter.
        OmL (float): The dark energy density parameter.
        w (float): The equation of state parameter for dark energy.
        eps (float, optional): The tolerance for convergence. Defaults to 1e-4.

    Returns:
        float: The linear growth rate at redshift z.
    """
    k1=np.zeros(2)
    k2=np.zeros(2)
    k3=np.zeros(2)
    k4=np.zeros(2)
    y=np.zeros(2)
    y2=np.zeros(2)
    y3=np.zeros(2)
    y4=np.zeros(2)


    a=1./(1.+z)
    a0=1./1100.

    yp=-1.
    n=10

    while(True):
        n*=2
        h=(np.log(a)-np.log(a0))/n

        x=np.log(a0)
        y[0]=1.
        y[1]=0.
        for i in range(n):
            for j in range(2):
                k1[j]=h*lgr_func_(j,x,y, OmM, OmL, w)
                y2[j]=y[j]+0.5*k1[j]
                k2[j]=h*lgr_func_(j,x+0.5*h,y2, OmM, OmL, w)
                y3[j]=y[j]+0.5*k2[j]
                k3[j]=h*lgr_func_(j,x+0.5*h,y3, OmM, OmL, w)
                y4[j]=y[j]+k3[j]
                k4[j]=h*lgr_func_(j,x+h,y4, OmM, OmL, w)
                y[j]+=(k1[j]+k4[j])/6.+(k2[j]+k3[j])/3.
            x+=h

        if(np.abs(y[0]/yp-1.)<0.1*1e-4): break
        yp=y[0]

    return a*y[0]

@njit(fastmath=True)
def window_(x, i):
    """Evaluates a window function based on the input parameters.

    Args:
        x (float): The input value for the window function.
        i (int): The index to select the type of window function.

    Returns:
        float: The value of the selected window function evaluated at x.

    Raises:
        ValueError: If i is not a recognized window function index.
    """
    if i == 0:
        return 3.0 / x**3 * (np.sin(x) - x * np.cos(x))  # top hat
    elif i == 1:
        return np.exp(-0.5 * x * x)  # gaussian
    elif i == 2:
        return x * np.exp(-0.5 * x * x)  # 1st derivative gaussian
    elif i == 3:
        return x * x * (1 - x * x) * np.exp(-x * x)
    elif i == 4:
        return (
            3 * (x * x - 3) * np.sin(x) + 9 * x * np.cos(x)
        ) / x**4  # 1st derivative top hat
    else:
        print("Window function not defined")
        return -1

@njit(fastmath=True)
def sigmam_(r, j, h, OmC, OmM, OmB, norm_P, nS, sigma8, eps=1e-4):
    """Calculates the variance of the density field smoothed with a given window function.

    Args:
        r (float): The smoothing scale in Mpc/h.
        j (int): The index to select the type of window function.
        h (float): The dimensionless Hubble parameter.
        OmC (float): The cold dark matter density parameter.
        OmM (float): The matter density parameter.
        OmB (float): The baryonic matter density parameter.
        norm_P (float): The normalization factor for the power spectrum.
        nS (float): The spectral index.
        sigma8 (float): The normalization of the linear density field.
        eps (float, optional): The tolerance for convergence. Defaults to 1e-4.

    Returns:
        float: The computed variance of the density field.
    """
    if sigma8 < 1e-8:
        return 0

    k1 = 2.0 * np.pi / r
    k2 = 2.0 * np.pi / r

    xxpp = -1.0
    while True:
        k1 /= 10.0
        k2 *= 2.0

        a = np.log(k1)
        b = np.log(k2)

        xxp = -1.0
        n = 2
        while True:
            n *= 2
            hh = (b - a) / n

            xx = 0.0
            for i in range(1, n):
                k = np.exp(a + hh * i)
                if j < 3:
                    xx += k * k * k * linear_pk_(k, h, OmC, OmM, OmB, norm_P, nS) * window_(k * r, j) ** 2
                else:
                    xx += k * k * k * linear_pk_(k, h, OmC, OmM, OmB, norm_P, nS) * window_(k * r, j)
            if j < 3:
                xx += 0.5 * (
                    k1 * k1 * k1 * linear_pk_(k1, h, OmC, OmM, OmB, norm_P, nS) * window_(k1 * r, j) ** 2
                    + k2
                    * k2
                    * k2
                    * linear_pk_(k2, h, OmC, OmM, OmB, norm_P, nS)
                    * window_(k2 * r, j) ** 2
                )
            else:
                xx += 0.5 * (
                    k1 * k1 * k1 * linear_pk_(k1, h, OmC, OmM, OmB, norm_P, nS) * window_(k1 * r, j)
                    + k2 * k2 * k2 * linear_pk_(k2, h, OmC, OmM, OmB, norm_P, nS) * window_(k2 * r, j)
                )

            xx *= hh

            if(abs((xx - xxp) / xx) < eps):
                break
            xxp = xx

        if(abs((xx - xxpp) / xx) < eps):
            break
        xxpp = xx

    if j < 3:
        return np.sqrt(xx / (2.0 * np.pi * np.pi))
    else:
        return xx / (2.0 * np.pi * np.pi)

@njit(fastmath=True)
def calc_r_sigma_(D1, h, OmC, OmM, OmB, norm_P, nS, sigma8, eps=1e-4):
    """Calculates the scale radius corresponding to a given linear density fluctuation.

    Uses a bisection method to find the scale radius where the smoothed variance matches the desired density fluctuation.

    Args:
        D1 (float): The linear growth factor at the redshift of interest.
        h (float): The dimensionless Hubble parameter.
        OmC (float): The cold dark matter density parameter.
        OmM (float): The matter density parameter.
        OmB (float): The baryonic matter density parameter.
        norm_P (float): The normalization factor for the power spectrum.
        nS (float): The spectral index.
        sigma8 (float): The normalization of the linear density field.
        eps (float, optional): The tolerance for convergence. Defaults to 1e-4.

    Returns:
        float: The scale radius corresponding to the linear density fluctuation.
    """
    if sigma8 < 1e-8:
        return 0

    k1 = k2 = 1.0
    while True:
        if D1 * sigmam_(1.0 / k1, 1, h, OmC, OmM, OmB, norm_P, nS, sigma8) < 1.0:
            break
        k1 *= 0.5

    while True:
        if D1 * sigmam_(1.0 / k2, 1, h, OmC, OmM, OmB, norm_P, nS, sigma8) > 1.0:
            break
        k2 *= 2.0

    while True:
        k = 0.5 * (k1 + k2)
        if D1 * sigmam_(1.0 / k, 1, h, OmC, OmM, OmB, norm_P, nS, sigma8) < 1.0:
            k1 = k
        elif D1 * sigmam_(1.0 / k, 1, h, OmC, OmM, OmB, norm_P, nS, sigma8) > 1.0:
            k2 = k
        if D1 * sigmam_(1.0 / k, 1, h, OmC, OmM, OmB, norm_P, nS, sigma8) == 1.0 or abs(k2 / k1 - 1.0) < eps * 0.1:
            break

    return 1.0 / k





class cosmology_numba:
    """
    A class to handle cosmological calculations including power spectrum and bispectrum
    based on specified cosmological parameters. It supports baryonification using
    different methods and precomputes necessary values for further calculations.

    Attributes:
        h (float): Hubble parameter.
        sigma8 (float): Amplitude of matter fluctuations.
        OmB (float): Baryon density parameter.
        OmC (float): Cold dark matter density parameter.
        nS (float): Scalar spectral index.
        w (float): Equation of state parameter for dark energy.
        OmL (float): Dark energy density parameter.
        OmM (float): Total matter density parameter (OmB + OmC).
        f_b (float): Baryon fraction.
        A_IA (float): Amplitude of intrinsic alignment.
        powerspec_model (str): Model for the power spectrum. (currently only 'baccoemu' implemented)
        powerspec_type (str): Type of the power spectrum ('linear' or 'non_linear').
        baryon_dict (dict): Dictionary of baryon parameters.
        baryon_parameters (dict): User-defined baryon parameters.
        baryonify (bool): Whether to include baryonification.
        D1 (interp1d): Interpolated growth factor as a function of redshift.
        r_sigma (interp1d): Interpolated scale radius as a function of redshift.
        n_eff (interp1d): Interpolated effective number density as a function of redshift.
        ncur (interp1d): Interpolated current density as a function of redshift.
        fk (interp1d): Interpolated growth rate factor as a function of redshift.
    """
    def __init__(
        self,
        h=0.7,
        sigma8=0.8,
        OmB=0.05,
        OmC=0.25,
        nS=0.96,
        w=-1,
        OmL=0.7,
        A_IA=0,
        zmin=0,
        zmax=3,
        zbins=256,
        
        powerspec = 'baccoemu',
        powerspec_type = 'non_linear',
        
        non_linear_powerspectrum_fname = '',
        
        baryon_dict = None,
        baryon_parameters = None,
        baryonify = False,
    ):
        """
        Initializes the cosmology_numba class with the given cosmological parameters.

        Parameters:
            h (float): Hubble parameter.
            sigma8 (float): Amplitude of matter fluctuations.
            OmB (float): Baryon density parameter.
            OmC (float): Cold dark matter density parameter.
            nS (float): Scalar spectral index.
            w (float): Equation of state parameter for dark energy.
            OmL (float): Dark energy density parameter.
            A_IA (float): Amplitude of intrinsic alignment.
            zmin (float): Minimum redshift for calculations.
            zmax (float): Maximum redshift for calculations.
            zbins (int): Number of redshift bins for precomputation.
            powerspec (str): Model for the power spectrum.
            powerspec_type (str): Type of the power spectrum ('linear' or 'non_linear').
            baryon_dict (dict): Dictionary of baryon parameters (if any).
            baryon_parameters (dict): User-defined baryon parameters (if any).
            baryonify (bool): Whether to include baryonification.
        """

        self.h = h
        self.sigma8 = sigma8
        self.OmB = OmB
        self.OmC = OmC
        self.nS = nS
        self.w = w
        self.OmL = OmL
        self.OmM = OmB + OmC 
        self.f_b = self.OmB/self.OmM 
        self.A_IA = A_IA
        
        self.powerspec_model = powerspec
        self.powerspec_type = powerspec_type
        
        
        if self.powerspec_model == None:
            warnings.warn("No powerspectrum model is set.")
        elif self.powerspec_model == 'baccoemu':
            # self.emulator=baccoemu.Matter_powerspectrum()
            self.emulator = baccoemu.Matter_powerspectrum(nonlinear_emu_path=non_linear_powerspectrum_fname, nonlinear_emu_details='details.pickle')
            
        else:
            raise ValueError(f"{self.powerspec_model} not allowed as option for powerspectrum")
        
        
        self.baryon_dict = baryon_dict
        self.baryon_parameters = baryon_parameters
        
        
        
        if baryon_dict is not None:
            self.powerspec_baryonification_type = baryon_dict['powerspec_baryonification_type']
            # self.bispectra_baryonification = cosmopower_NN(restore=True, restore_filename=self.baryon_dict['files']['bispectrum_emulator_file'])
            
            # if(self.powerspec_baryonification_type=='user'):
                # self.powerspectra_baryonification = cosmopower_NN(restore=True, restore_filename=self.baryon_dict['files']['powerspectrum_emulator_file'])
            
            self.baryon_kmin = self.baryon_dict['prior']['k'][0]
            self.baryon_kmax = self.baryon_dict['prior']['k'][1]
            self.baryon_zmin = 1/self.baryon_dict['prior']['expfactor'][1]-1
            self.baryon_zmax = 1/self.baryon_dict['prior']['expfactor'][0]-1
            
            self.check_prior()
            
        self.baryonify = baryonify
        
        if baryonify & (baryon_dict==None):
            raise ValueError("Cannot do baryonification, because baryon_dict was not given!")


        self.precomputations(zmin, zmax, zbins)

        self.cosmo = ccl.Cosmology(Omega_c =self.OmC, Omega_b = self.OmB, h = self.h, sigma8 = self.sigma8, n_s = self.nS)


    def check_prior(self):
        """
        Checks if the baryon parameters are within the prior ranges defined in baryon_dict.

        Raises:
            ValueError: If any baryon parameter or f_b is out of range.
        """
    
        if((self.f_b<self.baryon_dict['prior']['f_b'][0])
           or(self.f_b>self.baryon_dict['prior']['f_b'][1])):
            raise ValueError(f'f_b out of range ['+str(self.baryon_dict['prior']['f_b'][0])+','+str(self.baryon_dict['prior']['f_b'][1])+']')
        
        for name in self.baryon_parameters.keys():
            if((self.baryon_parameters[name]<self.baryon_dict['prior'][name][0])
               or(self.baryon_parameters[name]>self.baryon_dict['prior'][name][1])):
                raise ValueError(name+' out of range ['+str(self.baryon_dict['prior'][name][0])+','+str(self.baryon_dict['prior'][name][1])+']')
            
        print('All baryon parameters inside training range')
        
        
        
    def input_baryon_params(self, kvalues, z, type):
        """
        Prepares input parameters for baryonification based on a kvalues and redshift.

        Parameters:
            kvalues (list): List of k-values (wavenumbers). [h/Mpc]
            z (float): Redshift value.

        Returns:
            dict: Dictionary containing baryon parameters for the specified kvalues and z.
        """
        
        params = {}

        if(type=='powerspec'):
            if isinstance(kvalues, (int, float)):
                ones = np.ones(1)
                params['k'] = ones*kvalues
            elif(kvalues.ndim==1):
                ones = np.ones(len(kvalues))
                params['k'] = kvalues
            else:
                raise ValueError('dimension of kvalues is large than 1 but type is powerspec')
        elif(type=='bispec'):
            if(kvalues.ndim==1):
                ones = np.ones(1)
                params['k1'] = ones*kvalues[0]
                params['k2'] = ones*kvalues[1]
                params['k3'] = ones*kvalues[2]
                
            elif(kvalues.ndim==2):
                ones = np.ones_like(kvalues[:,0])
                params['k1'] = kvalues[0]
                params['k2'] = kvalues[1]
                params['k3'] = kvalues[2]
            else:
                raise ValueError('dimension of kvalues is neither 1 or 2.') 
        else:
            raise ValueError('type is wrong, it must be either powerspec or bispec')   
        
        params['M1_z0_cen'] = ones*self.baryon_parameters.get('M1_z0_cen', 11)
        params['M_c'] = ones*self.baryon_parameters.get('M_c', 12)
        params['M_inn'] = ones*self.baryon_parameters.get('M_inn', 11)
        params['beta'] = ones*self.baryon_parameters.get('beta', 0.0)
        params['eta'] = ones*self.baryon_parameters.get('eta', 0.0)
        params['theta_inn'] = ones*self.baryon_parameters.get('theta_inn', -1.0)
        params['theta_out'] = ones*self.baryon_parameters.get('theta_out', 0.2)
        params['expfactor'] = ones* 1/(1+z)
        params['f_b'] = ones*self.f_b
        
        return params
    
    
    def input_baryon_params_multiplez(self, kvalues, z, type):
        """
        Prepares input parameters for baryonification based on a kvalues and redshift.

        Parameters:
            kvalues (list): List of k-values (wavenumbers). [h/Mpc]
            z (float): Redshift values.

        Returns:
            dict: Dictionary containing baryon parameters for the specified kvalues and z.
        """
        
        params = {}

        params = {}
        ones = np.ones_like(z)

        if(type=='powerspec'):
            ones = np.ones_like(z)
            params['k'] = kvalues
        
        elif(type=='bispec'):    
            params['k1'] = kvalues[0]
            params['k2'] = kvalues[1]
            params['k3'] = kvalues[2]
        else:
            raise ValueError('type is wrong, it must be either powerspec or bispec')   
        
        params['M1_z0_cen'] = ones*self.baryon_parameters.get('M1_z0_cen', 11)
        params['M_c'] = ones*self.baryon_parameters.get('M_c', 12)
        params['M_inn'] = ones*self.baryon_parameters.get('M_inn', 11)
        params['beta'] = ones*self.baryon_parameters.get('beta', 0.0)
        params['eta'] = ones*self.baryon_parameters.get('eta', 0.0)
        params['theta_inn'] = ones*self.baryon_parameters.get('theta_inn', -1.0)
        params['theta_out'] = ones*self.baryon_parameters.get('theta_out', 0.2)
        params['expfactor'] = ones* 1/(1+z)
        params['f_b'] = ones*self.f_b
        
        return params
    
    def powerspec_pyccl(self, k, z):
        a = 1/(1+z)
        return self.cosmo.nonlin_power(k*self.h, a)*self.h*self.h*self.h

    def powerspec(self, k, z):
        """
        Computes the power spectrum for given wavenumbers `k` and redshift `z`, 
        with options for linear or non-linear spectra and optional baryonic corrections.

        Parameters
        ----------
        k : array-like
            Array of wavenumbers at which the power spectrum is calculated.
            
        z : float
            Redshift value at which the power spectrum is evaluated.

        Returns
        -------
        array-like
            Power spectrum values evaluated at the input `k` values, potentially modified by 
            baryonic corrections if enabled.

        Raises
        ------
        ValueError
            If an unsupported power spectrum model or type is provided, or if the required baryonic 
            parameters are missing when baryonification is requested.

        Notes
        -----
       
        Uses the Bacco emulator for cosmological power spectrum calculations 
        with parameters specified for cold dark matter, baryons, etc. It allows the computation 
        of both linear and non-linear spectra within specified redshift (`z`) and wavenumber (`k`) 
        ranges.
        
        - Linear power spectrum is valid for `0.25 < a <= 1.0` and `1e-4 < k < 4.692`.
        - Non-linear power spectrum is valid for `0.4 < a <= 1.0` and `1e-2 < k < 4.692`.

        Optional baryonic corrections can be applied if `self.baryonify` is set to `True`, using either:
        - `baccoemu_boost`: Requires specific baryonic parameters for corrections.
        - `user`: Allows user-defined baryonic correction models.

        Warnings are issued if the requested redshift or wavenumber is out of the emulator’s 
        defined range, and values are set to zero in such cases.
        
        Example
        -------
        # Assuming an instance `cosmo` with powerspec model 'baccoemu' and necessary parameters set:
        pk_values = cosmo.powerspec(k=np.array([0.1, 0.5, 1.0]), z=0.5)
        
        """
        a=1/(z+1)
        

        pk=np.zeros_like(k)

        if self.powerspec_model=='baccoemu':
            params_bacco={
                        'omega_cold'    : self.OmM,
                        'sigma8_cold'   : self.sigma8,
                        'omega_baryon'  : self.OmB,
                        'ns'            : self.nS,
                        'hubble'        : self.h,
                        'neutrino_mass' :  0.06,
                        'w0'            : self.w,
                        'wa'            :  0.0,
                        'expfactor'     :  a
                }
            
            
            # Calculate linear power spectrum
            if((a>0.25)&(a<=1.0)):
                mask = (k<50)&(k>1e-4) # Check that k is in valid range (larger range than for non-linear)
                
                if(np.sum(mask)==1): # If there is only one k value we need to pad the k-array with another value because bacco requires at least two k values
                    ktmp=np.array([k[mask][0], 1e-3]) # Padding
                    _, pktmp = self.emulator.get_linear_pk(k=(ktmp), baryonic_boost=False, **params_bacco) # Calculation for both ks
                    pk[mask] = pktmp[0] # Keeping only the wanted value 
                elif(np.sum(mask)>1): # If there are multiple ks, vectorized calculation of Pk
                    _, pk[mask] = self.emulator.get_linear_pk(k=(k[mask]), baryonic_boost=False, **params_bacco)
                else:
                    pass
            else: # Warning if we are outside the requested range
                raise ValueError("You are requesting the Bacco linear power spectrum at k's and a's where the emulator is not defined. I am setting the value to zero.")
        
            # If wanted - calculate the non-linear powerspectrum
            if(self.powerspec_type=='non_linear'):
                if((a>0.275)&(a<=1.0)):
                    mask = (k<9.803459558866939)&(k>1e-4)
                    if(np.sum(mask)==1): # If there is only one k value we need to pad the k-array with another value because bacco requires at least two k values
                        ktmp=np.array([k[mask][0], 1e-3])
                        _, pktmp = self.emulator.get_nonlinear_pk(k=(ktmp), baryonic_boost=False, **params_bacco)
                        pk[mask] = pktmp[0]
                    elif(np.sum(mask)>1): # If there are multiple ks, vectorized calculation of Pk
                        _, pk[mask] = self.emulator.get_nonlinear_pk(k=(k[mask]), baryonic_boost=False, **params_bacco)
                    else:
                        pass
                else:
                    warnings.warn("You are requesting the Bacco linear or non-linear power spectrum at k's and a's where the emulator is not defined. I am setting the value to zero.")
            elif(self.powerspec_type=='linear'):
                pass
            else:
                raise ValueError(f"powerspec_type {self.powerspec_type} is not defined, use 'non_linear' or 'linear'")
                
        else:
            raise ValueError("Powerspectrum model is not set!")
        

        powerspec_baryon_correction=np.ones_like(k)
        if(self.baryonify):
            if self.powerspec_baryonification_type=='baccoemu_boost':
                if not hasattr(self, 'baryon_parameters'):
                    raise ValueError("If you want to use the bacco baryonification you need to give the bacco baryon parameters! Use cosmo.baryonParams=baryonParams for this!")
                
                params_bacco={
                            'omega_cold'    : self.OmM,
                            'sigma8_cold'   : self.sigma8,
                            'omega_baryon'  : self.OmB,
                            'ns'            : self.nS,
                            'hubble'        : self.h,
                            'neutrino_mass' :  0.06,
                            'w0'            : self.w,
                            'wa'            :  0.0,
                            'expfactor'     :  a,
                            'theta_out'     : self.baryon_parameters.get('theta_out', 0.2),  # Default value for theta_out
                            'beta'          : self.baryon_parameters.get('beta', 0.0),  # Default value for beta
                            'theta_inn'     : self.baryon_parameters.get('theta_inn', -1.0),  # Default value for theta_inn
                            'M_c'           : self.baryon_parameters.get('M_c', 12),  # Default value for M_c
                            'M_inn'         : self.baryon_parameters.get('M_inn', 11),  # Default value for M_inn
                            'M1_z0_cen'     : self.baryon_parameters.get('M1_z0_cen', 11),  # Default value for M1_z0_cen
                            'eta'           : self.baryon_parameters.get('eta', 0.0)  # Default value for eta
                    }
                
                if((z<self.baryon_zmax) & (z>=self.baryon_zmin)): 
                    mask = (k<self.baryon_kmax)&(k>self.baryon_kmin)  
                    _, powerspec_baryon_correction[mask] = self.emulator.get_baryonic_boost(k=k[mask], **params_bacco)
                    mask = (k>self.baryon_kmax) 
                    powerspec_baryon_correction[mask] = 0.0 
                else:
                    raise ValueError("You are requesting the Bacco boost power spectrum at a's where the emulator is not defined. No baryon correction is applied.")
                    
                
            elif self.powerspec_baryonification_type=='user':
                if((z<self.baryon_zmax) & (z>=self.baryon_zmin)):
                    # print(z)
                    mask = (k<self.baryon_kmax)&(k>self.baryon_kmin)  
                    params = self.input_baryon_params(kvalues = k[mask], z=z, type='powerspec')
                    powerspec_baryon_correction[mask] = self.powerspectra_baryonification.rescaled_predictions_np(params).flatten() 
                    mask = (k>self.baryon_kmax) 
                    powerspec_baryon_correction[mask] = 0.0 
                else:
                    raise ValueError("You are requesting the emulator boost power spectrum at a's where the emulator is not defined. No baryon correction is applied.")

            elif self.powerspec_baryonification_type=='None':
                pass
            else:
                raise ValueError(f"{self.powerspec_baryonification_type} not implemented as baryonification for powerspectrum, use either user or Ps_nonlin_bary_bacco as powerspec_baryonification_type")


        return pk*powerspec_baryon_correction
        


    
    def precomputations(self, zmin, zmax, zbins, verbose=True):
        """
        Precomputes necessary values for the cosmological calculations over specified redshift range.

        Parameters:
            zmin (float): Minimum redshift.
            zmax (float): Maximum redshift.
            zbins (int): Number of redshift bins.
        """
        self.norm_P = 1  # initial setting, is overridden in next step
        self.norm_P = self.sigma8 / sigmam_(8.0, 0, self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS, self.sigma8)
        
        zarr = np.linspace(zmin, zmax, zbins)
        if verbose:
            print("Currently setting up D1")
        D1_arr = np.array([lgr_(z, self.OmM, self.OmL, self.w) for z in zarr] )/ lgr_(0.0, self.OmM, self.OmL, self.w)#).flatten()
        D1_arr = D1_arr.reshape(-1,)
        if verbose:
            print("Currently setting up r_sigma")
        r_sigma_arr = np.array([calc_r_sigma_(D1, self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS, self.sigma8) for D1 in D1_arr])

        if verbose:
            print("Currently setting up n_eff")
        D1_sigmam_2 = D1_arr * np.array([sigmam_(r, 2, self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS, self.sigma8) for r in r_sigma_arr])
        d1 = -2.0 * D1_sigmam_2**2
        n_eff_arr = -3.0 + 2.0 * D1_sigmam_2**2
        if verbose:
            print("Currenty setting up ncur")
        ncur_arr = (
            d1 * d1
            + 4.0 * np.array([sigmam_(r, 3, self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS, self.sigma8) for r in r_sigma_arr]) * D1_arr**2
        )

        if verbose:
            print("Currently setting up f_k(z)")
        fk_arr = np.array([self.calc_fk(z) for z in zarr])

        self.D1 = interp1d(zarr, D1_arr)
        self.r_sigma = interp1d(zarr, r_sigma_arr)
        self.n_eff = interp1d(zarr, n_eff_arr)
        self.ncur = interp1d(zarr, ncur_arr)
        self.fk = interp1d(zarr, fk_arr)

        if verbose:
            print("Done with precomputations")


    def bispec(self, k1, k2, k3, z):
        """
        Computes the bispectrum for a triangle configuration defined by wavenumbers `k1`, `k2`, and `k3`
        at one or multiple redshift values `z`. The calculation includes an optional baryonic correction if enabled.

        Parameters
        ----------
        k1, k2, k3 : float or ndarray
            Wavenumbers defining the triangle configuration for the bispectrum.
        z : float or ndarray
            Redshift value(s) at which the bispectrum is evaluated.

        Returns
        -------
        float or ndarray
            Bispectrum value(s) for the given wavenumbers and redshift(s), adjusted for baryonic effects if applicable.
        """
        scalar_input = np.isscalar(z)# Check if the input is scalar
        z = np.atleast_1d(z)  # Ensure z is an array
        k1 = np.atleast_1d(k1)
        k2 = np.atleast_1d(k2)
        k3 = np.atleast_1d(k3)
        D1_ = self.D1(z)
        r_sigma_ = self.r_sigma(z)
        n_eff_ = self.n_eff(z)
        bispec_baryon_correction = np.ones_like(z)
        if self.baryonify:
            mask = (
                (z < self.baryon_zmax) & (z >= self.baryon_zmin) &
                (k1 > self.baryon_kmin) & (k2 > self.baryon_kmin) & (k3 > self.baryon_kmin) &
                (k1 < self.baryon_kmax) & (k2 < self.baryon_kmax) & (k3 < self.baryon_kmax)
            )
            if np.any(mask):
                params = self.input_baryon_params(kvalues=np.stack([k1[mask], k2[mask], k3[mask]]), z=z[mask], type='bispec')
                bispec_baryon_correction[mask] = self.bispectra_baryonification.rescaled_predictions_np(params).flatten()            
            else:
                warnings.warn(f"You are requesting baryonification at ks or zs outside of the emulator range. The baryon correction is set to unity here.")


        bispec = np.zeros_like(z)
        for i in range(len(z)):
            if z[i] > 10.0:
                bispec[i] = bispec_tree_(k1[i], k2[i], k3[i], D1_[i], self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS)
            else:
                bispec[i] = bispec_(k1[i], k2[i], k3[i], z[i], D1_[i], r_sigma_[i], n_eff_[i], self.sigma8, self.nS, self.h, self.OmC, self.OmM, self.OmB, self.OmL, self.w, self.norm_P)

        bispec *= bispec_baryon_correction

        # Return a scalar if the input was scalar
        return bispec.item() if scalar_input else bispec
    
    
    def bispec_multiplez(self, k1, k2, k3, z):
        """
        Computes the bispectrum for a triangle configuration defined by wavenumbers `k1`, `k2`, and `k3`
        at a given redshift `z`. The calculation includes an optional baryonic correction if enabled.

        Parameters
        ----------
        k1, k2, k3 : float
            Wavenumbers defining the triangle configuration for the bispectrum.
            
        z : float
            Redshift at which the bispectrum is evaluated.

        Returns
        -------
        float
            Bispectrum value for the given wavenumbers and redshift, adjusted for baryonic effects if applicable.

        Notes
        -----
        The bispectrum is computed using:
        
        - The tree-level bispectrum approximation for redshifts `z > 10.0`.
        - A non-linear model for redshifts `z <= 10.0`.
        
        If `self.baryonify` is enabled and the input wavenumbers and redshift are within the 
        defined baryonic range (`baryon_kmin`, `baryon_kmax`, `baryon_zmin`, and `baryon_zmax`), 
        a baryonic correction is applied using parameters set in `self.input_baryon_params`.

        Example
        -------
        # Assuming `cosmo` is an instance with required parameters set:
        bispectrum_value = cosmo.bispec(k1=0.1, k2=0.2, k3=0.15, z=1.0)

        """

        D1_ = self.D1(z)
        r_sigma_ = self.r_sigma(z)
        n_eff_ = self.n_eff(z)
        
        if(np.any(z>self.baryon_zmax)):
            raise ValueError("You are requesting the bispectrum baryon boost at a's where the emulator is not defined.")
                    
        
        bispec_baryon_correction = np.ones_like(z)
        if(self.baryonify):
            # mask = (z<self.baryon_zmax)&(z>=self.baryon_zmin)&(k1>self.baryon_kmin)&(k2>self.baryon_kmin)&(k3>self.baryon_kmin)&(k1<self.baryon_kmax)&(k2<self.baryon_kmax)&(k3<self.baryon_kmax)
            mask = (z<self.baryon_zmax)&(z>=self.baryon_zmin)#&(k1>self.baryon_kmin)&(k2>self.baryon_kmin)&(k3>self.baryon_kmin)&(k1<self.baryon_kmax)&(k2<self.baryon_kmax)&(k3<self.baryon_kmax)
            params = self.input_baryon_params_multiplez(kvalues = np.array([k1[mask], k2[mask], k3[mask]]), z=z[mask], type='bispec')
            bispec_baryon_correction[mask] = self.bispectra_baryonification.rescaled_predictions_np(params).flatten()
                
        bispec = []
        for i in range(len(z)):
            if z[i] > 10.0:
                bispec.append(bispec_tree_(k1[i], k2[i], k3[i], D1_[i], self.h, self.OmC, self.OmM, self.OmB, self.norm_P, self.nS))

            else:
                bispec.append(bispec_(k1[i], k2[i], k3[i], z[i], D1_[i], r_sigma_[i], n_eff_[i], self.sigma8, self.nS, self.h, self.OmC, self.OmM, self.OmB, self.OmL, self.w, self.norm_P))
        
        return np.array(bispec)*bispec_baryon_correction


    
    def calc_fk(self, z):
        """
        Calculates the comoving distance to a given redshift `z` using numerical integration.

        Parameters
        ----------
        z : float
            Redshift value for which the comoving distance is calculated.

        Returns
        -------
        float
            The comoving distance to redshift `z` in units of Mpc/h.

        Notes
        -----
        This calculation integrates the inverse of the Hubble parameter from 0 to `z`, 
        using `self.Einv` for the integrand. The result is scaled by `c / H0`.

        Example
        -------
        # Assuming `cosmo` is an instance with required parameters set:
        comoving_distance = cosmo.calc_fk(z=1.5)
        """
        return integrate.quad(lambda a: self.Einv(a), 0, z)[0]*c_over_H0

    
    def Einv(self, z):
        """
        Computes the inverse of the Hubble parameter at a given redshift `z` for a cosmological model 
        with matter and dark energy densities.

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the inverse Hubble parameter.

        Returns
        -------
        float
            The inverse of the Hubble parameter at redshift `z`.

        Notes
        -----
        This function uses the matter density parameter `OmM`, the dark energy density parameter `OmL`, 
        and the dark energy equation of state parameter `w` to compute the inverse Hubble parameter.
        The calculation is given by:
        
            1 / sqrt(OmM * (1 + z)^3 + OmL * (1 + z)^(3 * (1 + w)))

        Example
        -------
        # Assuming `cosmo` is an instance with required parameters set:
        hubble_inverse = cosmo.Einv(z=0.5)
        """
        return 1 / (
            np.sqrt(self.OmM * pow(1 + z, 3) 
                    + self.OmL * pow(1 + z, 3 
                                     * (1.0 + self.w)))
        )

    def om_m_of_z(self, z):
        """
        Calculates the matter density parameter Omega_m(z) at a given redshift `z`.

        Parameters
        ----------
        z : float
            Redshift value at which Omega_m(z) is calculated.

        Returns
        -------
        float
            The matter density parameter at redshift `z`.

        Notes
        -----
        Omega_m(z) is calculated using the matter density `OmM`, the dark energy density `OmL`, 
        and the scale factor `a = 1 / (1 + z)` as follows:

            Omega_m(z) = OmM / (OmM + a * (a^2 * OmL + (1 - OmM - OmL)))

        Example
        -------
        # Assuming `cosmo` is an instance with required parameters set:
        omega_m_z = cosmo.om_m_of_z(z=0.5)

        """
        aa = 1. / (1 + z)
        return self.OmM / (self.OmM + aa * (aa * aa * self.OmL + (1. - self.OmM - self.OmL)))