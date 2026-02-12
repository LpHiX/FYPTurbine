import numpy as np
from scipy.optimize import root_scalar
from rocketcea.cea_obj_w_units import CEA_Obj

class Turbine:
    def __init__(self, 
                 P,       # W
                 RPM,     
                 d_mean_mm, # mm
                 mdot,     # kg/s
                 beta_deg,   # degrees
                 doa,       # degree of admission
                 p_e,  # exit pressure [Pa]
                 ):     
        
        # Store inputs
        self.P = P
        self.RPM = RPM
        self.d_mean_mm = d_mean_mm
        self.mdot = mdot
        self.beta = np.deg2rad(beta_deg)
        self.doa = doa
        self.p_e = p_e
        
        
        # Derived inputs
        self.d_mean = d_mean_mm / 1000

        # Angular velocity & blade speed
        self.w = self.RPM * 2 * np.pi / 60
        self.u = self.w * self.d_mean / 2
        
        # Useful enthalpy drop
        self.deltah_useful = self.P / self.mdot
        
        # Velocity triangles
        self.c3u = self.deltah_useful / (2 * self.u) + self.u
        self.c3 = self.c3u / np.cos(self.beta)
        self.c3m = self.c3 * np.sin(self.beta)

        # Swirl velocity at outlet
        self.c4u = self.u - (self.c3u - self.u)
        self.c4 = np.sqrt(self.c4u**2 + self.c3m**2)  
        
    def from_gasgen(self, cea: CEA_Obj, OF):
        self.cea = cea
        self.OF = OF
        # Find chamber pressure and expansion ratio        
        def expansion_func(eps, p01, c3):
            return c3 - self.cea.get_SonicVelocities(Pc=p01/1e5, MR=self.OF, eps=eps)[2] * self.cea.get_MachNumber(Pc=p01/1e5, MR=self.OF, eps=eps)

        rel_diff = 1
        self.p01 = self.p_e * 5  # Initial guess for total pressure at inlet
        iteration = 0
        if False:
            def test_expansion(eps, p01):
                return self.cea.get_SonicVelocities(Pc=p01/1e5, MR=self.OF, eps=eps)[2] * self.cea.get_MachNumber(Pc=p01/1e5, MR=self.OF, eps=eps)
            # Root scalar is failing, so plot expansion_func to find root visually
            import matplotlib.pyplot as plt
            eps_values = np.linspace(0, 50, 100)
            func_values = [expansion_func(eps, self.p01, self.c3) for eps in eps_values]
            func2_values = [test_expansion(eps, self.p01) for eps in eps_values]
            plt.plot(eps_values, func_values)
            plt.plot(eps_values, func2_values, label='test_expansion')
            plt.xlabel('Expansion Ratio (eps)')
            plt.ylabel('Expansion Function Value')
            plt.title('Expansion Function vs Expansion Ratio')
            plt.grid(True)
            plt.show()

            print(f"p01 guess: {self.p01/1e5:.2f} bar, c3: {self.c3:.2f} m/s")
            # Stop code here to inspect the plot
            return

        while rel_diff > 5e-4:
            self.eps = root_scalar(expansion_func, args=(self.p01, self.c3), bracket=[1, 50]).root
            self.p3 = self.p01 / self.cea.get_PcOvPe(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
            rel_diff = abs((self.p3 - self.p_e) / self.p_e)
            self.p01 *= self.p_e / self.p3
            iteration += 1
            if iteration > 10:
                return
        # print(f"Turbine sizing converged in {iteration} iterations.")

        # Outlet conditions
        self.T01, _, self.T3 = self.cea.get_Temperatures(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
        _ , self.a_throat, self.a3 = self.cea.get_SonicVelocities(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
        self.M3 = self.cea.get_MachNumber(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
        
        # Pressure & density at blade inlet
        # self.p3 = self.p01 / (1 + 0.5 * (self.gam - 1) * self.M3**2)**(self.gam/(self.gam-1))
        _, self.rho_throat, self.rho_3 = self.cea.get_Densities(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
        
        # Blade height
        self.Height = self.mdot / (self.doa * self.rho_3 * self.c3m * self.d_mean * np.pi)
        
        # Isentropic expansion to ambient
        # self.T_amb_is = self.T01 * (self.pamb/self.p01)**((self.gam - 1)/self.gam)
        # Isentropic expansion to static
        self.deltah_ambis = 0.5 * self.c3**2
        
        # Efficiency
        self.eff = self.deltah_useful / self.deltah_ambis
        # Huzel and huang efficiency

        # Blade-jet speed ratio
        self.blade_jet_speed_ratio = self.u / np.sqrt(2 * self.deltah_ambis)

        # Specific Speed
        self.q = self.mdot / (self.rho_3)
        self.specific_speed = self.RPM * np.sqrt(self.q) / (self.P**(3/4))

        # Total throat area
        self.A_throat = self.mdot / (self.rho_throat * self.a_throat)
        self.nozzles = 4
        self.nozzle_throat_length = self.A_throat / self.nozzles / self.Height
        self.nozzle_exit_length = self.eps * self.A_throat / self.nozzles / self.Height

        # Tg_c = self.cea.get_Tcomb(Pc=pc, MR=OF)
        mw3, self.gam3 = self.cea.get_exit_MolWt_gamma(Pc=self.p01/1e5, MR=self.OF, eps=self.eps)
        self.R_3 = 8314.5 / mw3  # J/kg-K
        # Cp_3 = self.cea.get_Chamber_Cp(Pc=pc, MR=OF, eps=40)

        # Throat to exit area ratio
        # gam = self.gam
        # self.eps = (( (gam + 1)/2 )**( - (gam + 1) / (2 * (gam - 1)) )) * ( 1 + 0.5 * (gam - 1) * self.M3**2 )**( (gam + 1) / (2 * (gam - 1)) ) * (1 / self.M3)

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
            mdot_n2 = self.A_throat * p01_n2 / np.sqrt(self.T01_n2) * np.sqrt(self.gam_n2 / self.R_n2) * ((2 / (self.gam_n2 + 1))**((self.gam_n2 + 1) / (2 * (self.gam_n2 - 1)))) 
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
    def from_inert_gas(self, R, gam, T01, nozzles):
        self.R_3 = R
        self.gam3 = gam
        self.T01 = T01
        self.cp = self.gam3 * self.R_3 / (self.gam3 - 1)

        rel_diff = 1
        self.p01 = self.p_e * 5  # Initial guess for total pressure at inlet
        iteration = 0

        while rel_diff > 5e-4:
            self.T3 = self.T01 - 0.5 * self.c3**2 / self.cp
            self.a3 = np.sqrt(self.gam3 * self.R_3 * self.T3)
            self.M3 = self.c3 / self.a3
            self.p3 = self.p01 / (1 + 0.5 * (self.gam3 - 1) * self.M3**2)**(self.gam3/(self.gam3-1))
            rel_diff = abs((self.p3 - self.p_e) / self.p_e)
            self.p01 *= self.p_e / self.p3

            # That was to match exit pressure at full thrust, for this FYP I'm going to set an arbitrary constraint that at 500W, the exit pressure matches p_e.

            # Since M3 is constant for a given geometry and assuming gas properties are the same, power scales with mass flow, which according to chokes flow equations, scales with inlet pressure. So we can just scale inlet pressure to match power.

            # pressure_ratio = 1 / (1 + 0.5 * (self.gam3 - 1) * self.M3**2)**(self.gam3/(self.gam3-1))
            # power_ratio = 500 / self.P
            # self.throttled_inlet_pressure = self.p01 * power_ratio
            # self.throttled_outlet_pressure = self.throttled_inlet_pressure * pressure_ratio
            # rel_diff = abs(( - self.p_e) / self.p_e)
            # self.p01 *= self.p_e / self.throttled_outlet_pressure
            iteration += 1
            if iteration > 10:
                break
        
        self.rho_3 = self.p3 / (self.R_3 * self.T3)
        self.Height = self.mdot / (self.doa * self.rho_3 * self.c3m * self.d_mean * np.pi)
        self.deltah_ambis = 0.5 * self.c3**2
        self.eff = self.deltah_useful / self.deltah_ambis
        self.A_throat = self.mdot / (self.p01 / np.sqrt(self.T01) * np.sqrt(self.gam3 / self.R_3) * ((2 / (self.gam3 + 1))**((self.gam3 + 1) / (2 * (self.gam3 - 1)))))
        self.A3 = self.mdot / (self.rho_3 * self.c3)
        self.nozzles = nozzles
        self.nozzle_throat_length = self.A_throat / self.nozzles / self.Height
        self.eps = self.A3 / self.A_throat
        self.nozzle_exit_length = self.eps * self.A_throat / self.nozzles / self.Height
        
        self.blade_jet_speed_ratio = self.u / np.sqrt(2 * self.deltah_ambis)
    
    def partload_rpm(self, partload_rpm):
        # Assuming pump load, so power is proportional to the third power of the speed. Parameter is partload rpm because that is the limiting factor for my testing.

        speed_ratio = partload_rpm / self.RPM
        partload_power = self.P * speed_ratio**3
        
        print(f"Partload results at {partload_rpm} RPM vs {self.RPM} RPM:")
        print(f"  Power: {partload_power/1000:.2f} kW (vs {self.P/1000:.2f} kW)")
        
        if hasattr(self, 'p01_n2'):
            partload_mdot = self.mdot_n2 * speed_ratio**3
            partload_pressure = self.p01_n2 * speed_ratio**3
            partload_exit_pressure = partload_pressure / (1 + 0.5 * (self.gam_n2 - 1) * self.M3_n2**2)**(self.gam_n2/(self.gam_n2-1))

            print(f"  N2 Mass flow: {partload_mdot*1000:.2f} g/s (vs {self.mdot_n2*1000:.2f} g/s)")
            print(f"  N2 Inlet pressure: {partload_pressure/1e5:.2f} bar (vs {self.p01_n2/1e5:.2f} bar)")
            print(f"  N2 Exit pressure: {partload_exit_pressure/1e5:.2f} bar (vs {self.p3_n2/1e5:.2f} bar)")
        else:
            partload_mdot = self.mdot * speed_ratio**3
            partload_pressure = self.p01 * speed_ratio**3
            partload_exit_pressure = partload_pressure / (1 + 0.5 * (self.gam3 - 1) * self.M3**2)**(self.gam3/(self.gam3-1))

            print(f"  Mass flow: {partload_mdot*1000:.2f} g/s (vs {self.mdot*1000:.2f} g/s)")
            print(f"  Inlet pressure: {partload_pressure/1e5:.2f} bar (vs {self.p01/1e5:.2f} bar)")
            print(f"  Exit pressure: {partload_exit_pressure/1e5:.2f} bar (vs {self.p3/1e5:.2f} bar)")

        


    def pretty_print(self):
        print(f"Turbine Results:---------------------------")
        print(f"Inputs:")
        print(f"  {'Power (P)':<30} {self.P/1000:<10.4g} kW")
        print(f"  {'RPM':<30} {self.RPM:<10.4g} RPM")
        print(f"  {'Mean Diameter (d_mean)':<30} {self.d_mean_mm:<10.4g} mm")
        print(f"  {'GG mdot':<30} {self.mdot*1000:<10.4g} g/s")
        print(f"  {'Nozzle Angle':<30} {np.rad2deg(self.beta):<10.4g} degree")
        print(f"  {'Degree of Admission':<30} {self.doa*100:<10.4g}%")
        print(f"Outputs:")
        print(f"  {'GG Temperature (T01)':<30} {self.T01:<10.4g} K")
        print(f"  {'Nozzle Exit Gamma':<30} {self.gam3:<10.4g}")
        # print(f"  {'Exit Heat Capacity':<30} {self.cp:<10.4g} J/Kg/K")
        print(f"  {'Nozzle Exit Gas Constant':<30} {self.R_3:<10.4g} J/kg/K")
        print(f"  {'GG Chamber Pressure':<30} {self.p01 / 1e5:<10.4g} bar")
        # print(f"  {'GG Pressure Ratio (p01/p3)':<30} {self.p_ratio:<10.4g}")
        print(f"  {'Rotor surface speed (u)':<30} {self.u:<10.4g} m/s")
        print(f"  {'Specific heat delta (dh_use)':<30} {self.deltah_useful:<10.4g} J/kg")
        print(f"  {'Absolute Circ. Velocity (c3u)':<30} {self.c3u:<10.4g} m/s")
        print(f"  {'Absolute Meri. Velocity (c3m)':<30} {self.c3m:<10.4g} m/s")
        print(f"  {'Absolute Velocity (c3)':<30} {self.c3:<10.4g} m/s")
        print(f"  {'Swirl Velocity (c4u)':<30} {self.c4u:<10.4g} m/s")
        print(f"  {'Absolute Velocity at exit (c4)':<30} {self.c4:<10.4g} m/s")
        print(f"  {'Exit Temperature (T3)':<30} {self.T3:<10.4g} K")
        print(f"  {'Exit sonic velocity (a3)':<30} {self.a3:<10.4g} m/s")
        print(f"  {'Exit Pressure (p3)':<30} {self.p3/1e5:<10.4g} bar")
        print(f"  {'Exit Density (rho_3)':<30} {self.rho_3:<10.4g} kg/m3")
        print(f"  {'Required Mach Number (M3)':<30} {self.M3:<10.4g}")
        print(f"  {'Required Blade Height (H_3)':<30} {self.Height*1000:<10.4g} mm")
        # print(f"  {'Isen Ambient Temp (t_amb_is)':<30} {self.T_amb_is:<10.4g} K")
        print(f"  {'Turbine Efficiency (eff)':<30} {self.eff:<10.4g}")
        print(f"  {'Blade-Jet Speed Ratio':<30} {self.blade_jet_speed_ratio:<10.4g}")   
        print(f"  {'Throat Area (A)':<30} {self.A_throat*1e6:<10.4g} mm2")
        print(f"  {'Area Ratio (eps)':<30} {self.eps:<10.4g}")   
        print(f"  {'Nozzle throat length':<30} {self.nozzle_throat_length*1000:<10.4g} mm")
        print(f"  {'Nozzle exit length':<30} {self.nozzle_exit_length*1000:<10.4g} mm")
        # print(f"  {'500W Inlet Pressure':<30} {self.throttled_inlet_pressure/1e5:<10.4g} bar")
        # print(f"  {'500W Outlet Pressure':<30} {self.throttled_outlet_pressure/1e5:<10.4g} bar")
        if hasattr(self, 'p01_n2'):
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