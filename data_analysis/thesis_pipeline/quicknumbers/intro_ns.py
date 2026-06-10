def get_sample_pump():
    isp = 250
    thrust = 5000
    g = 9.81
    mdot = thrust / (isp * g) * 0.5 # for OF=1
    density_keroesene = 800
    q = mdot / density_keroesene
    p = 50e5
    H = p / (density_keroesene * g)
    nq = 20
    n = nq * H**0.75 / q**0.5  # metric n_q = n*Q^0.5 / H^0.75  ->  n [rpm]
    return thrust, mdot, nq, n, p, density_keroesene


def get_sample_turbine():
    diameter = 0