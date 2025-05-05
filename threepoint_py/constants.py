import numpy as np

c_over_H0=2997.92
C1 = 5e-14 # NLA constant from Brown et al., 2002

# Physical constants
G = 6.6743e-11 # Newton constant [m^3 kg^-1 s^-2]

# Astronomy
Sun_mass = 1.9884e30 # Mass of the Sun [kg]
Mpc = 3.0857e16*1e6  # Mpc [m]


H0 = 100.                                      # Hubble parameter today in h [km/s/Mpc]
G_cosmological = G*Sun_mass/(Mpc*1e3**2)       # Gravitational constant [(Msun/h)^-1 (km/s)^2 (Mpc/h)] ~4.301e-9 (1e3**2 m -> km)
rho_critical = 3.*H0**2/(8.*np.pi*G_cosmological) # Critical density [(Msun/h) (Mpc/h)^-3] ~2.775e11 (Msun/h)/(Mpc/h)^3


Map2_epsrel=1e-2 # Integration accuracy (relative error)
Map3_epsrel=0.1 # Integration accuracy (relative error)

p_proj_epsrel=1e-2 # Integration accuracy (relative error)
b_proj_epsrel=1e-1 # Integration accuracy (relative error)
