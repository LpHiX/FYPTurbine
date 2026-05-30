import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from turbine import SupersonicStartingGoldman

gamma = 1.4
ss = SupersonicStartingGoldman(gamma)

nu_l = 0.0001
Ml = ss.mach_from_pm_rad(np.radians(nu_l), gamma)
Msl = ss.mstar_from_mach(Ml, gamma)

for nu_u in [20, 40, 60, 80]:
    Mu = ss.mach_from_pm_rad(np.radians(nu_u), gamma)
    Msu = ss.mstar_from_mach(Mu, gamma)
    nm = ss.max_inlet_pm_deg(Msl, Msu)
    print(f"nu_u: {nu_u}, nm: {nm}")
