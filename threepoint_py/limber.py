import numpy as np
from math import factorial
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from .constants import *
import warnings

class limber:
    """
    A class for computing projections of  
    bispectra and power spectra using the Limber approximation.

    Attributes:
    ----------
    cosmo : object
        Cosmology model with functions for calculating distance, growth factor, 
        and matter power spectrum.
    Ntomo : int
        Number of tomographic bins.
    gs : list of function
        List of functions for each tomographic bin that return the lensing 
        kernel g(z) at a given redshift.
    ps : list of function
        List of functions for each tomographic bin that return the redshift 
        distribution p(z) normalized to integrate to 1.
    zmax : float
        Maximum redshift in the analysis.
    zmin : float
        Minimum redshift in the analysis.
    AIA : float, optional
        Amplitude of the intrinsic alignment model (default is None).
    eta : float
        Power-law index for intrinsic alignment evolution (default is 0.0).
    """
    def __init__(
            self,
            cosmo,
            nz_sources,
            nz_lenses,
            AIA=None,
            eta=0.0,
            galaxy_bias_params=[[1.0, 0.0, 0.0]],
            model = "polynomial"
    ):
        """
        Initializes the Limber class with cosmological parameters, redshift 
        distributions, and intrinsic alignment settings.

        Parameters:
        ----------
        cosmo : object
            An object with methods for calculating distance and growth 
            functions in a given cosmology.
        nz_sources : list
            list of [z, nz] arrays, 1 element per tomo bin
        AIA : float, optional
            Amplitude of the intrinsic alignment model (default is None).
        eta : float, optional
            Power-law index for intrinsic alignment evolution (default is 0.0).
        """
        self.cosmo=cosmo

        self.Ntomo=len(nz_sources)

        # Initialize zmin and zmax by checking minimum and maximum values across all nz_sources entries
        self.zmin = min(np.min(nz_sources[i][0]) for i in range(self.Ntomo))
        self.zmax = max(np.max(nz_sources[i][0]) for i in range(self.Ntomo))

        # Get interpolated nz_sources
        self.ps=self.calculatePs(nz_sources)
        self.pl=self.calculatePl(nz_lenses)

        # Get interpolated gs
        self.gs=self.calculateGs()

        self.AIA=AIA
        self.eta=eta
        
        self.galaxy_bias_params=galaxy_bias_params
        self.model = model

    def calculatePs(self, nz_sources):
        """
        Normalizes and interpolates the redshift distributions for each 
        tomographic bin.

        Parameters:
        ----------
        nz_sources : array
            2D array of redshift values and distributions for each tomographic 
            bin.

        Returns:
        -------
        ps : list of function
            List of interpolating functions for p(z) in each tomographic bin.
        """
        ps=[]
        for i in range(self.Ntomo):
            zs=nz_sources[i][0]
            n_redshift_bins=len(zs)
            dz=(zs[-1]-zs[0])/n_redshift_bins


            nz=nz_sources[i][1]

            norm=np.sum(nz)*dz
            nz/=norm
            ps.append(interp1d(zs, nz, fill_value=(0,0), bounds_error=False))
        return ps
    
    def calculatePl(self, nz_lenses):
        """
        Normalizes and interpolates the redshift distributions for each 
        tomographic bin.

        Parameters:
        ----------
        nz_lenses : array
            2D array of redshift values and distributions for each tomographic 
            bin.

        Returns:
        -------
        ps : list of function
            List of interpolating functions for p(z) in each tomographic bin.
        """
        pl=[]
        for i in range(self.Ntomo):
            zs=nz_lenses[i][0]
            n_redshift_bins=len(zs)
            dz=(zs[-1]-zs[0])/n_redshift_bins


            nz=nz_lenses[i][1]

            norm=np.sum(nz)*dz
            nz/=norm
            pl.append(interp1d(zs, nz, fill_value=(0,0), bounds_error=False))
        return pl
    
    def galaxy_bias_calc(self, k, tomo):
        bias = 0
        if self.model == "polynomial":
            for j in range(len(self.galaxy_bias_params[tomo])):
                bias += self.galaxy_bias_params[tomo][j]*k**j
        elif self.model == "taylor":
            for j in range(len(self.galaxy_bias_params[tomo])):
                bias +=  self.galaxy_bias_params[tomo][j]/factorial(k)
        elif self.model == "flexible":
            a = self.galaxy_bias_params[tomo][0]
            b = self.galaxy_bias_params[tomo][1]
            c = self.galaxy_bias_params[tomo][2]
            d = self.galaxy_bias_params[tomo][3]
            bias += a / (1 + ((b * k) / c)**d)
        elif self.model == "flexible_offset":
            a = self.galaxy_bias_params[tomo][0]
            c = self.galaxy_bias_params[tomo][1]
            x0 = self.galaxy_bias_params[tomo][2]
            n = self.galaxy_bias_params[tomo][3]
            bias = c + a / (1 + (k / x0)**n)
            # print(bias, a, c, x0, n, k, (k / x0)**n)
        return bias


    def calculateGs(self):
        """
        Computes the lensing efficiency functions for each tomographic bin.
        
        Warning:
        -------
        Requires self.ps to be calculated. 

        Returns:
        -------
        gs : list of function
            List of interpolating functions for g(z) in each tomographic bin.
        """
#        zs=nz_sources[0]
        if not hasattr(self, 'ps'):
            raise AttributeError("Cannot compute Gs because Ps have not yet been computed. Call calculatePs first!")

        n_redshift_bins=256 #len(zs)
        zs=np.linspace(self.zmin, self.zmax, n_redshift_bins)
        dz=(zs[-1]-zs[0])/n_redshift_bins

        f_K_array=self.cosmo.fk(zs)
        f_K_array_inv=1.0/f_K_array

        gs=[]

        for i in range(self.Ntomo):
            # nz=nz_sources[i+1]
            # norm=np.sum(nz)*dz
            # nz/=norm
            nz=self.ps[i](zs)
            g_array=np.zeros(n_redshift_bins)
            for j in range(n_redshift_bins):
                f_K_diff = f_K_array[j:] - f_K_array[j]
                nz_znow = nz[j:]

                integrand = nz_znow * f_K_diff * f_K_array_inv[j:]
                trapezoidal_sum = np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])

                g_array[j] = trapezoidal_sum * dz    
            
            g_array=np.nan_to_num(g_array) # Replace nans with zero. This happens for z=0.
            gs.append(interp1d(zs, g_array, fill_value=(0,0), bounds_error=False))
        return gs
    



    def integrand_bkappa(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Calculates the integrand for the bispectrum of convergence, b_kappa.

        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Value of the integrand for b_kappa at the given redshift.
        """
        prefactor=self.prefactor_bkappa(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        f_K=self.cosmo.fk(z)

        bispec=self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z)

        return prefactor*bispec
    

    def integrand_bggi(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Calculates the integrand for the galaxy-galaxy-intrinsic bispectrum, 
        b_ggi.
        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Value of the integrand for b_ggi at the given redshift.
        """
        prefactor=self.prefactor_bggi(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        f_K=self.cosmo.fk(z)

        bispec=self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z)

        return prefactor*bispec

            

    def integrand_bgii(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Calculates the integrand for the galaxy-intrinsic-intrinsic bispectrum, 
        b_gii.
        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Value of the integrand for b_gii at the given redshift.
        """
        prefactor=self.prefactor_bgii(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        f_K=self.cosmo.fk(z)

        bispec=self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z)

        return prefactor*bispec

    

    def integrand_biii(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Calculates the integrand for the intrinsic-intrinsic-intrinsic bispectrum, 
        b_iii.
        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Value of the integrand for b_iii at the given redshift.
        """
        prefactor=self.prefactor_biii(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        f_K=self.cosmo.fk(z)

        bispec=self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z)

        return prefactor*bispec

    
    
    
    

    def bkappa(self, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Integrates the convergence bispectrum b_kappa over redshift.

        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Integrated b_kappa value.
        """
        return integrate.quad(self.integrand_bkappa, self.zmin, self.zmax, args=(ell1, ell2, ell3, tomo1, tomo2, tomo3), epsrel=b_proj_epsrel)[0]

    def bggi(self, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Integrates the galaxy-galaxy-intrinsic bispectrum over redshift.

        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Integrated b_kappa value.
        """
        return integrate.quad(self.integrand_bggi, self.zmin, self.zmax, args=(ell1, ell2, ell3, tomo1, tomo2, tomo3), epsrel=b_proj_epsrel)[0]

    def bgii(self, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Integrates the galaxy-intrinsic-intrinsic bispectrum over redshift.

        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Integrated b_kappa value.
        """
        return integrate.quad(self.integrand_bgii, self.zmin, self.zmax, args=(ell1, ell2, ell3, tomo1, tomo2, tomo3), epsrel=b_proj_epsrel)[0]

    def biii(self, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Integrates the intrinsic-intrinsic-intrinsic bispectrum over redshift.

        Parameters:
        ----------
        z : float
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Integrated b_kappa value.
        """
        return integrate.quad(self.integrand_biii, self.zmin, self.zmax, args=(ell1, ell2, ell3, tomo1, tomo2, tomo3), epsrel=b_proj_epsrel)[0]


  
    
    def prefactor_bkappa(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        scalar_input=np.isscalar(z)
        z = np.atleast_1d(z)  # Ensure z is an array
        prefactor = np.zeros_like(z)  # Initialize the result array
        mask = z < 1e-7  # Mask for small z values
        
        if (ell1 <= 1e-10) or (ell2 <= 1e-10) or (ell3 <= 1e-10):
            return prefactor if not scalar_input else 0  # Return array or scalar
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo) or (tomo3 >= self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")
        
        # Compute the required values
        g1 = self.gs[tomo1](z)
        g2 = self.gs[tomo2](z)
        g3 = self.gs[tomo3](z)
        f_K = self.cosmo.fk(z)
        dchi_dz = (self.cosmo.Einv(z) * c_over_H0)

        pre1 = 3 / 2 / c_over_H0 / c_over_H0 * self.cosmo.OmM * (1 + z) * g1
        pre2 = 3 / 2 / c_over_H0 / c_over_H0 * self.cosmo.OmM * (1 + z) * g2
        pre3 = 3 / 2 / c_over_H0 / c_over_H0 * self.cosmo.OmM * (1 + z) * g3

        prefactor = pre1 * pre2 * pre3 / f_K * dchi_dz
        prefactor[mask] = 0  # Apply the mask for small z values

        # Return scalar if the input was scalar
        return prefactor if not scalar_input else prefactor.item()

    

    def prefactor_bggi(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        if self.AIA==None:
            raise ValueError(f"Intrinsic alignment model not defined. Set the member variable AIA of the limber object to a value!")
        
        scalar_input=np.isscalar(z)
        z=np.atleast_1d(z) # Ensure z is an array
        prefactor = np.zeros_like(z)  # Initialize the result array
        mask = z < 1e-7  # Mask for small z values


        if (ell1 <= 1e-10) or (ell2 <= 1e-10) or (ell3 <= 1e-10):
            return prefactor if not scalar_input else 0  # Return array or scalar
                
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo) or (tomo3>=self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        g1=self.gs[tomo1](z)
        g2=self.gs[tomo2](z)
        p3=self.ps[tomo3](z)
        f_K=self.cosmo.fk(z)
        
        
        pre1= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g1
        pre2= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g2
        pre3=p3/f_K

        D=self.cosmo.D1(z)
        rho=self.cosmo.om_m_of_z(z)*rho_critical
        fIA=-self.AIA*C1*rho/D

        prefactor=pre1*pre2*pre3/f_K*fIA
        prefactor[mask] = 0

        # Return scalar if the input was scalar
        return prefactor if not scalar_input else prefactor.item()

            

    def prefactor_bgii(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        if self.AIA==None:
            raise ValueError(f"Intrinsic alignment model not defined. Set the member variable AIA of the limber object to a value!")
        scalar_input=np.isscalar(z)
        z=np.atleast_1d(z) # Ensure z is an array
        prefactor = np.zeros_like(z)  # Initialize the result array
        mask = z < 1e-7  # Mask for small z values


        if (ell1 <= 1e-10) or (ell2 <= 1e-10) or (ell3 <= 1e-10):
            return prefactor if not scalar_input else 0  # Return array or scalar
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo) or (tomo3>=self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        g1=self.gs[tomo1](z)
        p2=self.ps[tomo2](z)
        p3=self.ps[tomo3](z)
        f_K=self.cosmo.fk(z)
        dchi_dz=(self.cosmo.Einv(z)*c_over_H0)

        pre1= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g1
        pre2= p2/f_K
        pre3= p3/f_K

        D=self.cosmo.D1(z)
        rho=self.cosmo.om_m_of_z(z)*rho_critical
        fIA=-self.AIA*C1*rho/D

        prefactor=pre1*pre2*pre3/f_K/dchi_dz*fIA*fIA
        prefactor[mask] = 0

        # Return scalar if the input was scalar
        return prefactor if not scalar_input else prefactor.item()
    

    def prefactor_biii(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        if self.AIA==None:
            raise ValueError(f"Intrinsic alignment model not defined. Set the member variable AIA of the limber object to a value!")
        scalar_input=np.isscalar(z)
        z=np.atleast_1d(z) # Ensure z is an array
        prefactor = np.zeros_like(z)  # Initialize the result array
        mask = z < 1e-7  # Mask for small z values


        if (ell1 <= 1e-10) or (ell2 <= 1e-10) or (ell3 <= 1e-10):
            return prefactor if not scalar_input else 0  # Return array or scalar
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo) or (tomo3>=self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        p1=self.ps[tomo1](z)
        p2=self.ps[tomo2](z)
        p3=self.ps[tomo3](z)
        f_K=self.cosmo.fk(z)
        dchi_dz=(self.cosmo.Einv(z)*c_over_H0)

        
        pre1= p1/f_K
        pre2= p2/f_K
        pre3= p3/f_K

        D=self.cosmo.D1(z)
        rho=self.cosmo.om_m_of_z(z)*rho_critical
        fIA=-self.AIA*C1*rho/D

        prefactor=pre1*pre2*pre3/f_K/dchi_dz**2*fIA*fIA*fIA
        prefactor[mask] = 0

        # Return scalar if the input was scalar
        return prefactor if not scalar_input else prefactor.item()
    
    
    

    def bispectrum_projected(self, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Integrates the projected bispectrum over redshift.
        
        Parameters:
        ----------
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Projected bispectrum value.
        """
        return integrate.quad(self.integrand_projected, self.zmin, self.zmax, args=(ell1, ell2, ell3, tomo1, tomo2, tomo3), epsrel=b_proj_epsrel)[0]
        
        
    
    
    def integrand_projected(self, z, ell1, ell2, ell3, tomo1, tomo2, tomo3):
        """
        Computes the integrand for the projected bispectrum with contributions 
        from different bispectrum terms based on intrinsic alignments.    
        
        Parameters:
        ----------
        z : float or np.array
            Redshift at which the integrand is evaluated.
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.


        Returns:
        -------
        float
            integrand for full bispectrum
        """
        prefactor = self.prefactor_bkappa(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        z=np.atleast_1d(z) # Ensure that z is array

        
        if (self.AIA!=0) and (self.AIA!=None):
            prefactor+=self.prefactor_bggi(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)
            prefactor+=self.prefactor_bggi(z, ell1, ell2, ell3, tomo3, tomo1, tomo2)
            prefactor+=self.prefactor_bggi(z, ell1, ell2, ell3, tomo2, tomo3, tomo1)

            prefactor+=self.prefactor_bgii(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)
            prefactor+=self.prefactor_bgii(z, ell1, ell2, ell3, tomo3, tomo1, tomo2)
            prefactor+=self.prefactor_bgii(z, ell1, ell2, ell3, tomo2, tomo3, tomo1)

            prefactor+=self.prefactor_biii(z, ell1, ell2, ell3, tomo1, tomo2, tomo3)

        else:
            pass
        
        prefactor=np.atleast_1d(prefactor) # Ensure prefactor is an array

        mask = prefactor>0
        
        result = np.zeros_like(z)

        f_K=self.cosmo.fk(z[mask])

        # result[mask] = prefactor[mask]*self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z[mask])

        if(np.sum(mask)>1):
            result[mask] = prefactor[mask]*self.cosmo.bispec_multiplez(ell1/f_K, ell2/f_K, ell3/f_K, z[mask])
        elif(np.sum(mask)==1):
            result[mask] = prefactor[mask]*self.cosmo.bispec(ell1/f_K, ell2/f_K, ell3/f_K, z[mask])
        else:
            pass
        
        
        return result

    def bispectrum_projected_simps(self, ell1, ell2, ell3, tomo1, tomo2, tomo3, z_points=1000):
        """
        Integrates the projected bispectrum over redshift, handling both scalar and array-based evaluations.

        Parameters:
        ----------
        ell1, ell2, ell3 : float
            Multipole moments for the three sides of the bispectrum triangle.
        tomo1, tomo2, tomo3 : int
            Indices of the tomographic bins.
        z_points : int, optional
            Number of points for integration.

        Returns:
        -------
        float
            Projected bispectrum value
        """
        z_values = np.linspace(self.zmin, self.zmax, z_points)
        dz = z_values[1] - z_values[0]

        integrand = self.integrand_projected(z_values, ell1, ell2, ell3, tomo1, tomo2, tomo3)
 
        
        return integrate.simpson(y=integrand, x=z_values, dx=dz)


    def prefactor_pgg(self, z, ell, tomo1, tomo2):
        """
        Calculates the prefactor for the galaxy-galaxy power spectrum integrand.

        Parameters:
        ----------
        z : float
            Redshift.
        ell : float
            Multipole moment.
        tomo1, tomo2 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Prefactor for the galaxy-galaxy power spectrum.
        """
        if z<1e-7:
            return 0
        
        # if ell<=1e-10:
        #     return 0
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")
        
        g1=self.gs[tomo1](z)
        g2=self.gs[tomo2](z)
        dchi_dz=(self.cosmo.Einv(z)*c_over_H0)

        pre1= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g1
        pre2= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g2
        

        return pre1*pre2*dchi_dz


    def prefactor_pgi(self, z, ell, tomo1, tomo2):
        """
        Calculates the prefactor for the galaxy-intrinsic power spectrum integrand.

        Parameters:
        ----------
        z : float
            Redshift.
        ell : float
            Multipole moment.
        tomo1, tomo2 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Prefactor for the galaxy-intrinsic power spectrum.
        """
        if self.AIA==None:
            raise ValueError(f"Intrinsic alignment model not defined. Set the member variable AIA of the limber object to a value!")
        
        if z<1e-7:
            return 0
        
        # if (ell <= 1e-10):
        #     return 0
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        g1=self.gs[tomo1](z)
        p2=self.ps[tomo2](z)
        f_K=self.cosmo.fk(z)

        
        pre1= 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g1
        pre2=p2/f_K

        D=self.cosmo.D1(z)
        rho=self.cosmo.om_m_of_z(z)*rho_critical
        fIA=-self.AIA*C1*rho/D

        return pre1*pre2*fIA
    


    def prefactor_pii(self, z, ell, tomo1, tomo2):
        """
        Calculates the prefactor for the intrinsic-intrinsic power spectrum integrand.

        Parameters:
        ----------
        z : float
            Redshift.
        ell : float
            Multipole moment.
        tomo1, tomo2 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            Prefactor for the intrinsic-intrinsic power spectrum.
        """
        if self.AIA==None:
            raise ValueError(f"Intrinsic alignment model not defined. Set the member variable AIA of the limber object to a value!")
        
        if z<1e-7:
            return 0
        
        # if (ell <= 1e-10):
        #     return 0
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        p1=self.ps[tomo1](z)
        p2=self.ps[tomo2](z)
        f_K=self.cosmo.fk(z)
        dchi_dz=(self.cosmo.Einv(z)*c_over_H0)

        pre1= p1/f_K
        pre2= p2/f_K

        D=self.cosmo.D1(z)
        rho=self.cosmo.om_m_of_z(z)*rho_critical
        fIA=-self.AIA*C1*rho/D

        return pre1*pre2/dchi_dz*fIA*fIA
    


    def integrand_projected_powerspectrum(self, z, ell, tomo1, tomo2):
        """
        Computes the integrand for the projected power spectrum with galaxy 
        and intrinsic alignment terms.

        Parameters:
        ----------
        z : float
            Redshift.
        ell : float
            Multipole moment.
        tomo1, tomo2 : int
            Indices of the tomographic bins.

        Returns:
        -------
        float
            integrand for the total power spectrum.
        """

        prefactor = self.prefactor_pgg(z, ell, tomo1, tomo2)
        
        if (self.AIA!=0) and (self.AIA!=None):
            prefactor+=self.prefactor_pgi(z, ell, tomo1, tomo2)
            prefactor+=self.prefactor_pgi(z, ell, tomo2, tomo1)
            prefactor+=self.prefactor_pii(z, ell, tomo1, tomo2)
        else:
            pass
        
        if(prefactor==0):
            return np.zeros_like(ell)
        
        f_K=self.cosmo.fk(z)
        ell_over_f_K = (ell + 0.5)/ f_K
        result = prefactor*(self.cosmo.powerspec_pyccl(ell_over_f_K, z))
        return result
    

    # OLD VERSION USING SIMPSON INTEGRATION
    # def powerspectrum_projected(self, ell, tomo1, tomo2):
    #     """
    #     Integrates the projected power spectrum over redshift

    #     Parameters:
    #     ----------
    #     z : float
    #         Redshift.
    #     ell : float
    #         Multipole moment.
    #     tomo1, tomo2 : int
    #         Indices of the tomographic bins.

    #     Returns:
    #     -------
    #     float
    #         total power spectrum.
    #     """
    #     z_points=64
    #     previous_result = None
    #     it=0
    #     # Vectorized simpson integration
    #     while True:
    #         #print(f"Limber integration on iteration {it}")
    #         z_values = np.linspace(self.zmin, self.zmax, z_points)
    #         integrand_values = np.array([
    #             self.integrand_projected_powerspectrum(z, ell, tomo1, tomo2)
    #             for z in z_values
    #         ])
    #         current_result = integrate.simpson(integrand_values, x=z_values, axis=0)
            
    #         mask=(current_result!=0)

    #         # Check for convergence if we have a previous result
    #         if previous_result is not None and np.all(np.abs(current_result[mask] - previous_result[mask])/current_result[mask] < p_proj_epsrel):
    #             break
            
    #         # Update and double the number of points for the next loop
    #         previous_result = current_result
    #         z_points *= 2
    #         it+=1

    #     return current_result


    # FASTER VERSION WITH QUAD
    # I tried parallelizing this, but could not find a way to make it work.
    def powerspectrum_projected(self, ell_array, tomo1, tomo2):
        results = []
        ell_array=np.atleast_1d(ell_array)
        for ell in ell_array:
            result=self.powerspectrum_projected_one_ell(ell, tomo1, tomo2)
            results.append(result)
        return np.array(results)


    def powerspectrum_projected_one_ell(self, ell, tomo1, tomo2):
        def integrand(z):
            return self.integrand_projected_powerspectrum(z, ell, tomo1, tomo2)
        result, _ = integrate.quad(integrand, self.zmin, self.zmax, epsrel=p_proj_epsrel)

        return result

    def prefactor_pkappa_delta(self, z, tomo1, tomo2):
    
    
        if z<1e-7:
            return 0
        
        # if (ell <= 1e-10):
        #     return 0
        
        if (tomo1 >= self.Ntomo) or (tomo2 >= self.Ntomo):
            raise ValueError(f"Only {self.Ntomo} bins defined! Check which tomo bins you want!")

        g1=self.gs[tomo1](z)
        p2=self.pl[tomo2](z)
        
        f_K=self.cosmo.fk(z)
        
        # dchi_dz=(self.cosmo.Einv(z)*c_over_H0)
        

        
        pre1 = 3/2/c_over_H0/c_over_H0*self.cosmo.OmM*(1+z)*g1
        pre2 = p2/f_K

        return pre1*pre2
        
    
    
    def integrand_projected_powerspectrum_kappa_delta(self, z, ell, tomo1, tomo2):
      

        prefactor = self.prefactor_pkappa_delta(z, tomo1, tomo2)
        
        if(prefactor==0):
            return np.zeros_like(ell)
        
        f_K=self.cosmo.fk(z)
        ell_over_f_K = (ell + 0.5) / f_K
        bias = self.galaxy_bias_calc(k=ell_over_f_K, tomo=tomo1)
        result = prefactor*(self.cosmo.powerspec_pyccl(ell_over_f_K, z))*bias
        return result
    
    
    def powerspectrum_projected_kappa_delta(self, ell_array, tomo1, tomo2):
        results = []
        ell_array=np.atleast_1d(ell_array)
        for ell in ell_array:
            result=self.powerspectrum_projected_kappa_delta_one_ell(ell, tomo1, tomo2)
            results.append(result)
        return np.array(results)


    def powerspectrum_projected_kappa_delta_one_ell(self, ell, tomo1, tomo2):
        def integrand(z):
            return self.integrand_projected_powerspectrum_kappa_delta(z, ell, tomo1, tomo2)
        result, _ = integrate.quad(integrand, self.zmin, self.zmax, epsrel=p_proj_epsrel)

        return result
    