import numpy as np
from scipy.integrate import nquad


class covs_3D_matter:
    """
    A class to calculate covariance matrices for power spectrum and bispectrum
    estimates in a 3D matter field, using a given cosmological model and survey volume.

    Parameters
    ----------
    cosmo : object
        A cosmology object that provides methods for computing power spectrum and bispectrum values.
    
    V : float
        The survey volume over which the covariances are calculated, in units of Mpc/h cubed.
    """
    def __init__(self, cosmo, V):
        self.cosmo=cosmo
        self.V=V





    def _Vp(self, k, delK):
        """
        Computes the volume element in Fourier space for a single wavenumber bin.

        Parameters
        ----------
        k : float
            Wavenumber for the bin.

        delK : float
            Width of the wavenumber bin.

        Returns
        -------
        float
            The Fourier volume element for the given wavenumber bin.
        """
        return 4*np.pi**2*k**2*delK

    def _Vb(self, k1, k2, k3, delK1, delK2, delK3):
        """
        Computes the volume element in Fourier space for a triplet of wavenumber bins.

        Parameters
        ----------
        k1, k2, k3 : float
            Wavenumbers defining the bin triplet.
        
        delK1, delK2, delK3 : float
            Widths of the corresponding wavenumber bins.

        Returns
        -------
        float
            The Fourier volume element for the wavenumber triplet.
        """
        return 8*np.pi**2*k1*k2*k3*delK1*delK2*delK3

    def cov_powerspec_PP_single(self, k1, k2, delK, z=0):
        """
        Computes the Gaussian covariance for the power spectrum for a single pair of wavenumbers.

        Parameters
        ----------
        k1, k2 : float
            Wavenumbers for the power spectrum covariance calculation.
        
        delK : float
            Width of the wavenumber bin.
        
        z : float, optional
            Redshift at which the covariance is evaluated (default is 0).

        Returns
        -------
        float
            Covariance of the power spectrum at the specified wavenumbers and redshift.
        
        Notes
        -----
        Returns 0 if `k1` and `k2` are not equal, as Gaussian covariance for different bins is zero.
        """
        if k1!=k2:
            return 0
        else:
            P=self.cosmo.powerspec(k1, z)
            Vp=self._Vp(k1, delK)

            return 2*(2*np.pi)**3*P**2/self.V/Vp
        
    def cov_bispec_PPP_single(self, k1, k2, k3, k4, k5, k6, delK1, delK2, delK3, z=0):
        """
        Computes the Gaussian covariance for the bispectrum for a single set of wavenumber triplets.

        Parameters
        ----------
        k1, k2, k3 : float
            Wavenumbers for the first bispectrum triplet.
        
        k4, k5, k6 : float
            Wavenumbers for the second bispectrum triplet.
        
        delK1, delK2, delK3 : float
            Widths of the wavenumber bins for the first triplet.
        
        z : float, optional
            Redshift at which the covariance is evaluated (default is 0).

        Returns
        -------
        float
            Covariance of the bispectrum at the specified wavenumber triplets and redshift.
        """

        tmp=0

        P1=self.cosmo.powerspec(k1, z)
        P2=self.cosmo.powerspec(k2, z)
        P3=self.cosmo.powerspec(k3, z)

        Vb=self._Vb(k1, k2, k3, delK1, delK2, delK3)

        if (k1==k4) & (k2==k5) & (k3==k6):
            tmp+=1
        if (k1==k4) & (k2==k6) & (k3==k5):
            tmp+=1
        if (k1==k5) & (k2==k4) & (k3==k6):
            tmp+=1
        if (k1==k5) & (k2==k6) & (k3==k4):
            tmp+=1
        if (k1==k6) & (k2==k4) & (k3==k5):
            tmp+=1
        if (k1==k6) & (k2==k5) & (k3==k4):
            tmp+=1


        return tmp*(2*np.pi)**3*P1*P2*P3/self.V/Vb
    
    def cov_powerbispec_PB_single(self, k1, k2, k3, k4, delK1, delK2, delK3, delK4, z=0):
        """
        Computes the covariance between the power spectrum and bispectrum for single configurations.

        Parameters
        ----------
        k1 : float
            Wavenumber for the power spectrum.
        
        k2, k3, k4 : float
            Wavenumbers for the bispectrum.
        
        delK1, delK2, delK3, delK4 : float
            Widths of the wavenumber bins for the respective wavenumbers.
        
        z : float, optional
            Redshift at which the covariance is evaluated (default is 0).

        Returns
        -------
        float
            Covariance between the power spectrum and bispectrum at the given configuration and redshift.
        """
        Vp=self._Vp(k1, delK1)
        P=self.cosmo.powerspec(k1, z)
        B=self.cosmo.bispec(k2, k3, k4, z)

        tmp=0

        if k1==k2:
            tmp+=1
        if k1==k3:
            tmp+=1
        if k1==k4:
            tmp+=1

        return 2*tmp*(2*np.pi)**3*P*B/self.V/Vp#


    def cov_powerspec_gauss(self, ks, delKs, z=0):
        """
        Computes the Gaussian covariance matrix for the power spectrum over multiple wavenumbers.

        Parameters
        ----------
        ks : array-like
            Array of wavenumbers at which to compute the power spectrum covariance.
        
        delKs : array-like
            Array of widths for each corresponding wavenumber bin in `ks`.
        
        z : float, optional
            Redshift at which the covariance matrix is evaluated (default is 0).

        Returns
        -------
        numpy.ndarray
            Covariance matrix of the power spectrum evaluated at the specified wavenumbers and redshift.
        
        Raises
        ------
        ValueError
            If the lengths of `ks` and `delKs` do not match.
        """
        if len(ks) != len(delKs):
            raise ValueError("Need as many delta ks as ks")

        N=len(ks)

        cov=np.zeros((N,N))

        for i, k in enumerate(ks):
            cov[i,i]=self.cov_powerspec_PP_single(k,k, delKs[i], z)

        return cov
    
    def cov_bispec_gauss(self, ktriplets, delKtriplets, z=0):
        """
        Computes the Gaussian covariance matrix for the bispectrum over multiple wavenumber triplets.

        Parameters
        ----------
        ktriplets : array-like
            Array of wavenumber triplets, each containing three wavenumbers.
        
        delKtriplets : array-like
            Array of bin widths corresponding to each triplet in `ktriplets`.
        
        z : float, optional
            Redshift at which the covariance matrix is evaluated (default is 0).

        Returns
        -------
        numpy.ndarray
            Covariance matrix of the bispectrum evaluated at the specified triplets and redshift.

        Raises
        ------
        ValueError
            If the lengths of `ktriplets` and `delKtriplets` do not match, or if the triplet elements are not in groups of three.
        """

        if len(ktriplets) != len(delKtriplets):
            raise ValueError("Need as many delta ks as ks")
        
        if len(ktriplets[0]) != 3:
            raise ValueError("Need 3 values per triplet")
        
        if len(delKtriplets[0]) != 3:
            raise ValueError("Need 3 values per triplet")
        
        N=len(ktriplets)

        cov=np.zeros((N,N))

        for i, ks1 in enumerate(ktriplets):
            for j, ks2 in enumerate(ktriplets):
                cov[i,j]=self.cov_bispec_PPP_single(ks1[0], ks1[1], ks1[2],
                                                    ks2[0], ks2[1], ks2[2],
                                                    delKtriplets[i][0], delKtriplets[i][1], delKtriplets[i][2],
                                                    z)
                
        return cov

    def cov_powerbispec_part1(self, ks, ktriplets, delKs, delKtriplets, z=0):
        """
        Computes part of the covariance matrix between the power spectrum and bispectrum 
        over multiple wavenumber configurations.

        Parameters
        ----------
        ks : array-like
            Array of wavenumbers for the power spectrum covariance.
        
        ktriplets : array-like
            Array of wavenumber triplets for the bispectrum covariance.
        
        delKs : array-like
            Bin widths corresponding to each wavenumber in `ks`.
        
        delKtriplets : array-like
            Bin widths corresponding to each triplet in `ktriplets`.
        
        z : float, optional
            Redshift at which the covariance matrix is evaluated (default is 0).

        Returns
        -------
        numpy.ndarray
            Covariance matrix between the power spectrum and bispectrum for the given configurations and redshift.
        """
        N1=len(ks)
        N2=len(ktriplets)

        cov=np.zeros((N1, N2))

        for i, k1 in enumerate(ks):
            for j, ks2 in enumerate(ktriplets):
                cov[i,j]=self.cov_powerbispec_PB_single(k1, ks2[0], ks2[1], ks2[2], delKs[i], delKtriplets[j][0], delKtriplets[j][1], delKtriplets[j, 2], z=z)

        return cov
