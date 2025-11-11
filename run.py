import numpy as np
from scipy.optimize import root_scalar

from rocketcea.cea_obj_w_units import CEA_Obj
# from rocketcea.cea_obj import add_new_fuel
# from pyfluids import Fluid, FluidsList, Input
# import numpy as np

# class combustion_results:
#     def __init__(self, tc, cp):
#         self.tc = tc
#         self.cp = cp
    
#     def pretty_print(self):
#         print(f"Combustion Results:")
#         print(f"  {'Combustion Temperature (Tc)':<30} {self.tc:<10.4g} K")
#         print(f"  {'Combustion Heat Capacity (Cp)':<30} {self.cp:<10.4g} J/kg/K")

class GasGenerator:
    def __init__(self, ox: str, fuel: str):
        self.ox = ox
        self.fuel = fuel
        self.cea = CEA_Obj(
            oxName = self.ox,
            fuelName = self.fuel,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            # fac_CR=self.cr,
            make_debug_prints=False)
    def combustion_sim(self, OF, pc):
        Tg_c = self.cea.get_Tcomb(Pc=pc, MR=OF)
        [mw_c, gam_c] = self.cea.get_Chamber_MolWt_gamma(Pc=pc, MR=OF, eps=40)
        R_c = 8314.5 / mw_c  # J/kg-K
        Cp_c = self.cea.get_Chamber_Cp(Pc=pc, MR=OF, eps=40)
        # print(f"mw={mw_c}, R={R_c}, gam={gam_c}, Cp={Cp_c}")
        return Tg_c, Cp_c, R_c, gam_c


class Turbine:
    def __init__(self, 
                 P,       # W
                 RPM,     
                 d_mean_mm, # mm
                 T01,       # K
                 gam,
                 mdot,     # kg/s
                 cp,       # J/kg/K
                 R,         # J/kg/K
                 p01_bar,    # bar
                 beta_deg,   # degrees
                 doa,       # degree of admission
                 pamb):     # ambient pressure [Pa]
        
        # Store inputs
        self.P = P
        self.RPM = RPM
        self.d_mean_mm = d_mean_mm
        self.T01 = T01
        self.gam = gam
        self.mdot = mdot
        self.cp = cp
        self.R = R
        self.p01_bar = p01_bar
        self.beta = np.deg2rad(beta_deg)
        self.doa = doa
        self.pamb = pamb
        
        # Derived inputs
        self.d_mean = d_mean_mm / 1000
        self.p01 = p01_bar * 1e5

        # Run calculations immediately
        self._compute()

    def _compute(self):
        # Angular velocity & blade speed
        self.w = self.RPM * 2 * np.pi / 60
        self.u = self.w * self.d_mean / 2
        
        # Useful enthalpy drop
        self.deltah_useful = self.P / self.mdot
        
        # Velocity triangles
        self.c3u = self.deltah_useful / (2 * self.u) + self.u
        self.c3 = self.c3u / np.cos(self.beta)
        self.c3m = self.c3 * np.sin(self.beta)
        
        # Outlet conditions
        self.T3 = self.T01 - 0.5 * self.c3**2 / self.cp
        self.a3 = np.sqrt(self.gam * self.R * self.T3)
        self.M3 = self.c3 / self.a3
        
        # Pressure & density at blade inlet
        self.p3 = self.p01 / (1 + 0.5 * (self.gam - 1) * self.M3**2)**(self.gam/(self.gam-1))
        self.rho_3 = self.p3 / (self.R * self.T3)
        
        # Blade height
        self.Height = self.mdot / (self.doa * self.rho_3 * self.c3m * self.d_mean)
        
        # Isentropic expansion to ambient
        self.T_amb_is = self.T01 * (self.pamb/self.p01)**((self.gam - 1)/self.gam)
        self.deltah_ambis = self.cp * (self.T01 - self.T_amb_is)
        
        # Efficiency
        self.eff = self.deltah_useful / self.deltah_ambis

        # Blade-jet speed ratio
        self.blade_jet_speed_ratio = self.u / np.sqrt(2 * self.deltah_ambis)

        # Total throat area
        self.A = self.mdot / (self.p01 / np.sqrt(self.T01) * np.sqrt(self.gam / self.R) * ((2 / (self.gam + 1))**((self.gam + 1) / (2 * (self.gam - 1)))))

        # Throat to exit area ratio
        gam = self.gam
        self.eps = (( (gam + 1)/2 )**( - (gam + 1) / (2 * (gam - 1)) )) * ( 1 + 0.5 * (gam - 1) * self.M3**2 )**( (gam + 1) / (2 * (gam - 1)) ) * (1 / self.M3)

        # Nitrogen required calculations
        self.T01_n2 = 300  # K
        self.cp_n2 = 1040  # J/kg/K
        self.R_n2 = 296.8  # J/kg/K
        self.gam_n2 = 1.4

        rel_diff = 1
        p01_n2 = self.p01 * 5 # Initial guess
        iteration = 0


        def machfunc(mach, area_ratio, gam):
            if mach == 0:
                mach = 1e-7
            return area_ratio - ((1.0/mach) * ((1 + 0.5*(gam-1)*mach*mach) / ((gam + 1)/2))**((gam+1) / (2*(gam-1))))
        
        # self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[1, 5]).root
        
        while rel_diff > 5e-4:
            M3_n2 = root_scalar(machfunc, args=(self.eps, self.gam_n2), bracket=[1, 5]).root
            T3_n2 = self.T01_n2 / (1 + 0.5 * (self.gam_n2 - 1) * M3_n2**2)
            p3_n2 = p01_n2 / (1 + 0.5 * (self.gam_n2 - 1) * M3_n2**2)**(self.gam_n2/(self.gam_n2-1))
            rho3_n2 = p3_n2 / (self.R_n2 * T3_n2)
            
            c3_n2 = M3_n2 * np.sqrt(self.gam_n2 * self.R_n2 * T3_n2)

            c3u_n2 = c3_n2 * np.cos(self.beta)

            deltah_useful_n2 = 2 * self.u * (c3u_n2 - self.u)
            mdot_n2 = self.A * p01_n2 / np.sqrt(self.T01_n2) * np.sqrt(self.gam_n2 / self.R_n2) * ((2 / (self.gam_n2 + 1))**((self.gam_n2 + 1) / (2 * (self.gam_n2 - 1)))) 
            P_n2 = mdot_n2 * deltah_useful_n2

            rel_diff = abs((P_n2 - self.P) / self.P)
            # print(f"Iteration {iteration}: p01_n2 = {p01_n2/1e5:.2f} bar, p3_n2 = {p3_n2/1e5:.2f} bar, P_n2 = {P_n2:.2f} W, rel_diff = {rel_diff:.6f}")
            # print(f"it={iteration}, p01={p01_n2/1e5:.2f} bar, p3={p3_n2/1e5:.2f} bar, c3={c3_n2:.2f} m/s, M3={M3_n2:.2f}, P_n2={P_n2:.2f} W, rel_diff={rel_diff:.2e}")
            # print(f"mdot={mdot_n2*1000:.2f} g/s, deltah_useful={deltah_useful_n2:.2f} J/kg rho3={rho3_n2:.2f} kg/m3 T3={T3_n2:.2f} K")
            p01_n2 *= self.P / P_n2
            # p01_n2 *= np.sqrt(self.P / P_n2)
            iteration += 1

            
            self.p01_n2 = p01_n2
            self.M3_n2 = M3_n2
            self.T3_n2 = T3_n2
            self.p3_n2 = p3_n2
            self.rho3_n2 = rho3_n2
            self.c3_n2 = c3_n2
            self.c3u_n2 = c3u_n2
            self.deltah_useful_n2 = deltah_useful_n2
            self.mdot_n2 = mdot_n2
            self.P_n2 = P_n2
    

            if iteration > 100:
                break

        



    def pretty_print(self):
        print(f"Inputs:")
        print(f"  {'Power (P)':<30} {self.P/1000:<10.4g} kW")
        print(f"  {'RPM':<30} {self.RPM:<10.4g} RPM")
        print(f"  {'Mean Diameter (d_mean)':<30} {self.d_mean_mm:<10.4g} mm")
        # print(f"  {'GG Temperature (T01)':<30} {self.T01:<10.4g} K")
        # print(f"  {'GG Heat Capacity Ratio (gam)':<30} {self.gam:<10.4g}")
        print(f"  {'GG mdot':<30} {self.mdot*1000:<10.4g} g/s")
        # print(f"  {'GG Heat Capacity':<30} {self.cp:<10.4g} J/Kg/K")
        # print(f"  {'GG Gas Constant':<30} {self.R:<10.4g} J/kg/K")
        print(f"  {'GG Chamber Pressure':<30} {self.p01_bar:<10.4g} bar")
        print(f"  {'Nozzle Angle':<30} {np.rad2deg(self.beta):<10.4g} degree")
        print(f"  {'Degree of Admission':<30} {self.doa*100:<10.4g}%")
        print(f"Outputs:")
        print(f"  {'Rotor surface speed (u)':<30} {self.u:<10.4g} m/s")
        print(f"  {'Specific heat delta (dh_use)':<30} {self.deltah_useful:<10.4g} J/kg")
        print(f"  {'Absolute Circ. Velocity (c3u)':<30} {self.c3u:<10.4g} m/s")
        print(f"  {'Absolute Meri. Velocity (c3m)':<30} {self.c3m:<10.4g} m/s")
        print(f"  {'Absolute Velocity (c3)':<30} {self.c3:<10.4g} m/s")
        print(f"  {'Exit Temperature (T3)':<30} {self.T3:<10.4g} K")
        print(f"  {'Exit sonic velocity (a3)':<30} {self.a3:<10.4g} m/s")
        print(f"  {'Exit Pressure (p3)':<30} {self.p3/1e5:<10.4g} bar")
        print(f"  {'Exit Density (rho_3)':<30} {self.rho_3:<10.4g} kg/m3")
        print(f"  {'Required Mach Number (M3)':<30} {self.M3:<10.4g}")
        print(f"  {'Required Blade Height (H_3)':<30} {self.Height*1000:<10.4g} mm")
        print(f"  {'Isen Ambient Temp (t_amb_is)':<30} {self.T_amb_is:<10.4g} K")
        print(f"  {'Turbine Efficiency (eff)':<30} {self.eff:<10.4g}")
        print(f"  {'Blade-Jet Speed Ratio':<30} {self.blade_jet_speed_ratio:<10.4g}")   
        print(f"  {'Throat Area (A)':<30} {self.A*1e6:<10.4g} mm2")
        print(f"  {'Area Ratio (eps)':<30} {self.eps:<10.4g}")   
        print(f"Nitrogen Testing Requirements:")
        print(f"  {'N2 Pressure (p01_n2)':<30} {self.p01_n2/1e5:<10.4g} bar")
        print(f"  {'N2 Temperature (T01_n2)':<30} {self.T01_n2:<10.4g} K")
        print(f"  {'N2 Mass Flow Rate (mdot_n2)':<30} {self.mdot_n2*1000:<10.4g} g/s")
        print(f"  {'N2 Exit Mach Number (M3_n2)':<30} {self.M3_n2:<10.4g}")
        print(f"  {'N2 Exit Temperature (T3_n2)':<30} {self.T3_n2:<10.4g} K")
        print(f"  {'N2 Exit Pressure (p3_n2)':<30} {self.p3_n2/1e5:<10.4g} bar")
        print(f"  {'N2 Exit Density (rho3_n2)':<30} {self.rho3_n2:<10.4g} kg/m3")
        print(f"  {'N2 Exit Velocity (c3_n2)':<30} {self.c3_n2:<10.4g} m/s")
        print(f"  {'N2 Specific heat delta (dh_n2)':<30} {self.deltah_useful_n2:<10.4g} J/kg")
        print(f"  {'N2 Power (P_n2)':<30} {self.P_n2/1000:<10.4g} kW") 

# Example usage
if __name__ == "__main__":
    gasgen = GasGenerator(ox='N2O', fuel='Isopropanol')
    Tg_c, Cp_c, R_c, gam_c = gasgen.combustion_sim(OF=0.5, pc=20)
    print(f"Combustion Results:")
    print(f"  {'Combustion Temperature (Tc)':<30} {Tg_c:<10.4g} K")
    print(f"  {'Combustion Heat Capacity (Cp)':<30} {Cp_c:<10.4g} J/kg/K")
    print(f"  {'Combustion Gas Constant (R)':<30} {R_c:<10.4g} J/kg/K")
    print(f"  {'Combustion  gam (gam)':<30} {gam_c:<10.4g}")  

    stage = Turbine(
        P=13000,       # W
        RPM=20000,     
        d_mean_mm=100, # mm
        T01=Tg_c,       # K
        gam=gam_c,
        mdot=0.04,     # kg/s
        cp=Cp_c,       # J/kg/K
        R=R_c,         # J/kg/K
        p01_bar=20,    # bar
        beta_deg=15,   # degrees
        doa=0.2,       # degree of admission
        pamb=1e5)     # ambient pressure [Pa]
    stage.pretty_print()
    

    
