import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4
GM1 = GAMMA - 1
GP1 = GAMMA + 1

def Mstar(M):
    return np.sqrt(GP1 / 2 * M**2 / (1 + GM1 / 2 * M**2))

def safe_asin_scalar(x):
    return np.arcsin(np.clip(x, -1, 1))

def nu_pm(M):
    """Prandtl-Meyer angle (radians)."""
    Ms = Mstar(M)
    return (np.pi / 4 * (np.sqrt(GP1 / GM1) - 1)
            + 0.5 * (np.sqrt(GP1 / GM1) * safe_asin_scalar(GM1 * Ms**2 - GAMMA)
                      + safe_asin_scalar(GP1 / Ms**2 - GAMMA)))

def M_from_nu(nu_target):
    """Invert Prandtl-Meyer function (radians in, Mach out)."""
    return brentq(lambda M: nu_pm(M) - nu_target, 1.001, 10.0)

if __name__ == '__main__':
    nu_lower_deg = 34.0
    nu_upper_deg = 49.0
    
    m_lower = M_from_nu(np.deg2rad(nu_lower_deg))
    m_upper = M_from_nu(np.deg2rad(nu_upper_deg))
    
    print(f"For gamma = {GAMMA}:")
    print(f"  nu = {nu_lower_deg} deg -> Mach = {m_lower:.5f}")
    print(f"  nu = {nu_upper_deg} deg -> Mach = {m_upper:.5f}")
