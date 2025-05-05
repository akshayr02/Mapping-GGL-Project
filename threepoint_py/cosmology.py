import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from .cosmopower_NN import cosmopower_NN 


from .constants import *

def scalefactor_from_redshift(z:float) -> float:
    return 1./(1.+z)


def comoving_matter_density(Om_m:float) -> float:
    '''
    Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
    args:
        Om_m: Cosmological matter density (at z=0)
    '''
    return rho_critical*Om_m

class cosmology:
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
        baryon_dict = None,
        baryonify = False
    ):

        self.h = h
        self.sigma8 = sigma8
        self.OmB = OmB
        self.OmC = OmC
        self.nS = nS
        self.w = w
        self.OmL = OmL
        self.OmM = OmB + OmC
        self.A_IA = A_IA
        
        self.baryon_dict = baryon_dict

        if baryon_dict is not None:
            self.powerspectra_ratio = cosmopower_NN(restore=True, restore_filename=self.baryon_dict['powerspectrum_emulator_file'])
            self.bispectra_ratio = cosmopower_NN(restore=True, restore_filename=self.baryon_dict['bispectrum_emulator_file'])
            self.baryon_kmin = self.baryon_dict['kmin']
            self.baryon_kmax = self.baryon_dict['kmax']
            self.baryon_zmin = self.baryon_dict['zmin']
            self.baryon_zmax = self.baryon_dict['zmax']
            
        self.baryonify = baryonify
        
        if baryonify & (baryon_dict==None):
            raise ValueError("Cannot do baryonification, because baryon_dict was not given!")



        self.precomputations(zmin, zmax, zbins)

    def precomputations(self, zmin, zmax, zbins, verbose=True):

        self.norm_P = 1  # initial setting, is overridden in next step
        self.norm_P = self.sigma8 / self.sigmam(8.0, 0)

        zarr = np.linspace(zmin, zmax, zbins)
        if verbose:
            print("Currently setting up D1")
        D1_arr = np.array([self.lgr(z) for z in zarr] / self.lgr(0.0)).flatten()
        if verbose:
            print("Currently setting up r_sigma")
        r_sigma_arr = np.array([self.calc_r_sigma(D1) for D1 in D1_arr])

        if verbose:
            print("Currently setting up n_eff")
        D1_sigmam_2 = D1_arr * np.array([self.sigmam(r, 2) for r in r_sigma_arr])
        d1 = -2.0 * D1_sigmam_2**2
        n_eff_arr = -3.0 + 2.0 * D1_sigmam_2**2
        if verbose:
            print("Currenty setting up ncur")
        ncur_arr = (
            d1 * d1
            + 4.0 * np.array([self.sigmam(r, 3) for r in r_sigma_arr]) * D1_arr**2
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

    def lgr_func(self, j, la, y):
        if(j==0): return y[1]
      
        a=np.exp(la)
        g=-0.5*(5.*self.OmM+(5.-3*self.w)*self.OmL*pow(a,-3.*self.w))*y[1]-1.5*(1.-self.w)*self.OmL*pow(a,-3.*self.w)*y[0]
        g=g/(self.OmM+self.OmL*pow(a,-3.*self.w))
        if(j==1): return g
        else:
            raise ValueError("lgr_func: j not a valid value.")

    def lgr(self, z, eps=1e-4):
        """Linear growth rate. Uses Runge-Kutta method to integrate the differential equation

        Args:
            z (_type_): _description_
            eps (_type_, optional): _description_. Defaults to 1e-10.

        Returns:
            _type_: _description_
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
                    k1[j]=h*self.lgr_func(j,x,y)
                    y2[j]=y[j]+0.5*k1[j]
                    k2[j]=h*self.lgr_func(j,x+0.5*h,y2)
                    y3[j]=y[j]+0.5*k2[j]
                    k3[j]=h*self.lgr_func(j,x+0.5*h,y3)
                    y4[j]=y[j]+k3[j]
                    k4[j]=h*self.lgr_func(j,x+h,y4)
                    y[j]+=(k1[j]+k4[j])/6.+(k2[j]+k3[j])/3.
                x+=h

            if(np.abs(y[0]/yp-1.)<0.1*1e-4): break
            yp=y[0]

        return a*y[0]


    def linear_pk(self, k):
        # Eisenstein & Hu linear P(k) (no wiggles)

        k *= self.h  # unit conversion from [h/Mpc] to [1/Mpc]

        fc = self.OmC / self.OmM
        fb = self.OmB / self.OmM
        theta = 2.728 / 2.7
        pc = 0.25 * (5.0 - np.sqrt(1.0 + 24.0 * fc))

        omh2 = self.OmM * self.h * self.h
        ombh2 = self.OmB * self.h * self.h

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
            self.norm_P**2
            * (k * 2997.9 / self.h) ** (3.0 + self.nS)
            * (L / (L + C * qeff**2)) ** 2
        )
        pk = 2.0 * np.pi * np.pi / (k**3) * delk

        return self.h**3 * pk

    def window(self, x, i):
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

    def sigmam(self, r, j, eps=1e-4):
        if self.sigma8 < 1e-8:
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
                        xx += k * k * k * self.linear_pk(k) * self.window(k * r, j) ** 2
                    else:
                        xx += k * k * k * self.linear_pk(k) * self.window(k * r, j)
                if j < 3:
                    xx += 0.5 * (
                        k1 * k1 * k1 * self.linear_pk(k1) * self.window(k1 * r, j) ** 2
                        + k2
                        * k2
                        * k2
                        * self.linear_pk(k2)
                        * self.window(k2 * r, j) ** 2
                    )
                else:
                    xx += 0.5 * (
                        k1 * k1 * k1 * self.linear_pk(k1) * self.window(k1 * r, j)
                        + k2 * k2 * k2 * self.linear_pk(k2) * self.window(k2 * r, j)
                    )

                xx *= hh

                if np.abs((xx - xxp) / xx) < eps:
                    break
                xxp = xx

            if np.abs((xx - xxpp) / xx) < eps:
                break
            xxpp = xx

        if j < 3:
            return np.sqrt(xx / (2.0 * np.pi * np.pi))
        else:
            return xx / (2.0 * np.pi * np.pi)

    def calc_r_sigma(self, D1, eps=1e-4):
        if self.sigma8 < 1e-8:
            return 0

        k1 = k2 = 1.0
        while True:
            if D1 * self.sigmam(1.0 / k1, 1) < 1.0:
                break
            k1 *= 0.5

        while True:
            if D1 * self.sigmam(1.0 / k2, 1) > 1.0:
                break
            k2 *= 2.0

        while True:
            k = 0.5 * (k1 + k2)
            if D1 * self.sigmam(1.0 / k, 1) < 1.0:
                k1 = k
            elif D1 * self.sigmam(1.0 / k, 1) > 1.0:
                k2 = k
            if D1 * self.sigmam(1.0 / k, 1) == 1.0 or np.abs(k2 / k1 - 1.0) < eps * 0.1:
                break

        return 1.0 / k

    def bispec_tree(self, k1, k2, k3, D1):

        # Compute the tree-level bispectrum
        return (
            (D1**4)
            * 2.0
            * (
                self.F2_tree(k1, k2, k3) * self.linear_pk(k1) * self.linear_pk(k2)
                + self.F2_tree(k2, k3, k1) * self.linear_pk(k2) * self.linear_pk(k3)
                + self.F2_tree(k3, k1, k2) * self.linear_pk(k3) * self.linear_pk(k1)
            )
        )

    def F2_tree(self, k1, k2, k3):
        costheta12 = 0.5 * (k3 * k3 - k1 * k1 - k2 * k2) / (k1 * k2)
        return (
            (5.0 / 7.0)
            + 0.5 * costheta12 * (k1 / k2 + k2 / k1)
            + (2.0 / 7.0) * costheta12 * costheta12
        )

    def F2(self, k1, k2, k3, z, D1, r_sigma):

        q = k3 * r_sigma

        logsigma8z = np.log10(D1 * self.sigma8)
        a = 1.0 / (1.0 + z)
        omz = self.OmM / (
            self.OmM + self.OmL * a ** (-3.0 * self.w)
        )  # Omega matter at z

        dn = 10.0 ** (-0.483 + 0.892 * logsigma8z - 0.086 * omz)

        return self.F2_tree(k1, k2, k3) + dn * q

    def powerspec(self, k, z):
        """Non-linear powerspectrum (revised halofit, Takahashi+ 2012)

        Args:
            k (_type_): _description_
            z (_type_): _description_
        """
        D1_ = self.D1(z)
        r_sigma_ = self.r_sigma(z)
        n_eff_ = self.n_eff(z)
        nsqr=n_eff_*n_eff_
        ncur_=self.ncur(z)

        om_m= self.om_m_of_z(z)
        om_v = self.om_v_of_z(z)

        f1a = pow(om_m, (-0.0732))
        f2a = pow(om_m, (-0.1423))
        f3a = pow(om_m, (0.0725))
        f1b = pow(om_m, (-0.0307))
        f2b = pow(om_m, (-0.0585))
        f3b = pow(om_m, (0.0743))
        frac = om_v / (1. - om_m)
        f1 = frac * f1b + (1 - frac) * f1a
        f2 = frac * f2b + (1 - frac) * f2a
        f3 = frac * f3b + (1 - frac) * f3a
        a = 1.5222 + 2.8553 * n_eff_ + 2.3706 * nsqr + 0.9903 * n_eff_ * nsqr + 0.2250 * nsqr * nsqr - 0.6038 * ncur_ + 0.1749 * om_v * (1.0 + self.w)
        a = pow(10.0, a)
        b = pow(10.0, -0.5642 + 0.5864 * n_eff_ + 0.5716 * nsqr - 1.5474 * ncur_ + 0.2279 * om_v * (1.0 + self.w))
        c = pow(10.0, 0.3698 + 2.0404 * n_eff_ + 0.8161 * nsqr + 0.5869 * ncur_)
        gam = 0.1971 - 0.0843 * n_eff_ + 0.8460 * ncur_
        alpha = np.abs(6.0835 + 1.3373 * n_eff_ - 0.1959 * nsqr - 5.5274 * ncur_)
        beta = 2.0379 - 0.7354 * n_eff_ + 0.3157 * nsqr + 1.2490 * n_eff_ * nsqr + 0.3980 * nsqr * nsqr - 0.1682 * ncur_
        xnu = pow(10.0, 5.2105 + 3.6902 * n_eff_)

        plin = self.linear_pk(k) * D1_ * D1_ * k * k * k / (2 * np.pi**2)
        
        y = k * r_sigma_
        ysqr = y * y
        ph = a * pow(y, f1 * 3) / (1 + b * pow(y, f2) + pow(f3 * c * y, 3 - gam))
        ph = ph / (1 + xnu / ysqr)
        pq = plin * pow(1 + plin, beta) / (1 + plin * alpha) * np.exp(-y / 4.0 - ysqr / 8.0)

        delta_nl = pq + ph

        return 2 * np.pi**2 * delta_nl / k**3

    def bispec(self, k1, k2, k3, z):

        D1_ = self.D1(z)
        r_sigma_ = self.r_sigma(z)
        n_eff_ = self.n_eff(z)

        bispec_ratio = 1
        if(self.baryonify):
            if((k1>self.baryon_kmin)&(k2>self.baryon_kmin)&(k3>self.baryon_kmin)
               &(k1<self.baryon_kmax) &(k2<self.baryon_kmax)&(k3<self.baryon_kmax)
               &(z<self.baryon_zmax) & (z>=self.baryon_zmin)):  
                params = self.input_params(kvalues = np.array([k1, k2, k3]), z=z)
                bispec_ratio = self.bispectra_ratio.rescaled_predictions_np(params).flatten()[0] 



        if z > 10.0:
            return self.bispec_tree(k1, k2, k3, D1_)*bispec_ratio

        q = np.array(
            [0, k1 * r_sigma_, k2 * r_sigma_, k3 * r_sigma_]
        )  # dimensionless wavenumbers

        # sorting q[i] so that q[1] >= q[2] >= q[3]
        q[1:4] = sorted(q[1:4], reverse=True)
        r1 = q[3] / q[1]
        r2 = (q[2] + q[3] - q[1]) / q[1]

        q[1], q[2], q[3] = k1 * r_sigma_, k2 * r_sigma_, k3 * r_sigma_
        logsigma8z = np.log10(D1_ * self.sigma8)

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
        if alphan > 1.0 - (2.0 / 3.0) * self.nS:
            alphan = 1.0 - (2.0 / 3.0) * self.nS
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
            ) * D1_**2 * self.linear_pk(q[i] / r_sigma_) + 1.0 / (
                mn * q[i] ** mun + nn * q[i] ** nun
            ) / (
                1.0 + (pn * q[i]) ** -3
            )

        # 3-halo term bispectrum in Eq.(B5)
        BS3h = 2.0 * (
            self.F2(k1, k2, k3, z, D1_, r_sigma_) * PSE[1] * PSE[2]
            + self.F2(k2, k3, k1, z, D1_, r_sigma_) * PSE[2] * PSE[3]
            + self.F2(k3, k1, k2, z, D1_, r_sigma_) * PSE[3] * PSE[1]
        )
        for i in range(1, 4):
            BS3h *= 1.0 / (1.0 + en * q[i])

        return (BS1h + BS3h)*bispec_ratio

    def calc_fk(self, z):
        return integrate.quad(lambda a: self.Einv(a), 0, z)[0]*c_over_H0

    def Einv(self, z):

        return 1 / (
            np.sqrt(self.OmM * pow(1 + z, 3) 
                    + self.OmL * pow(1 + z, 3 
                                     * (1.0 + self.w)))
        )


    def om_m_of_z(self, z):
        
        aa = 1. / (1 + z)
        return self.OmM / (self.OmM + aa * (aa * aa * self.OmL + (1. - self.OmM - self.OmL)))

    def om_v_of_z(self, z):
        aa = 1. / (1 + z)
        return self.OmL * aa * aa * aa / (self.OmM + aa * (aa * aa * self.OmL + (1. - self.OmM - self.OmL)))
