import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
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
    
    def check_supersonic_start(self, M_rel, gamma):
        """
        Checks if the rotor passage can swallow the starting shock (Kantrowitz Limit).

        Derivation from First Principles (Metric):
        1. Assume a normal shock forms at the rotor inlet due to starting transients.
        2. Across the shock, total temperature T0 remains constant, but total pressure drops (P03 -> P0_shock).
        3. For the shock to be swallowed, the mass flow captured at the inlet (mdot_in) 
           must be able to pass through the rotor throat (At_rotor) at choked conditions (Mt=1).
        4. Mass flow function: f(M) = mdot * sqrt(R*T0) / (A * P0 * sqrt(gamma))
           f(M) = M * (1 + (gamma-1)/2 * M^2)^(-(gamma+1)/(2*(gamma-1)))
        5. At the limit (Kantrowitz Limit):
           A_in * P03 * f(M_rel) = At_rotor * P0_shock * f(1)
           (At_rotor / A_in)_min = [f(M_rel) / f(1)] * (P03 / P0_shock)

        6. Normal shock total pressure ratio (P0_shock / P03):
           pr = [((gamma+1)*M^2)/((gamma-1)*M^2 + 2)]^(gamma/(gamma-1)) * [(gamma+1)/(2*gamma*M^2 - (gamma-1))]^(1/(gamma-1))

        7. Actual geometric contraction ratio:
           For an impulse turbine, we check the limit against a 1:1 ratio (constant area passage).
           The margin tells us how much blockage (boundary layers, blade thickness) we can afford.
        """
        def f_mass(M, g):
            if M <= 0: return 0
            term = 1 + (g - 1) / 2 * M**2
            return M * term**(-(g + 1) / (2 * (g - 1)))

        def p0_ratio(M, g):
            if M <= 1.0: return 1.0
            term1 = ((g + 1) * M**2) / ((g - 1) * M**2 + 2)
            term2 = (g + 1) / (2 * g * M**2 - (g - 1))
            return (term1**(g / (g - 1))) * (term2**(1 / (g - 1)))

        f_m1 = f_mass(M_rel, gamma)
        f_1 = f_mass(1.0, gamma)
        pr = p0_ratio(M_rel, gamma)

        self.req_area_ratio = (f_m1 / f_1) / pr
        
        # Start margin against 1.0 (constant area passage)
        self.start_margin = (1.0 / self.req_area_ratio) - 1
        self.is_started = self.start_margin > 0

        return self.is_started

    def partload_rpm(self, partload_rpm):
        speed_ratio = partload_rpm / self.RPM
        partload_power = self.P * speed_ratio**3
        
        print(f"Partload results at {partload_rpm} RPM vs {self.RPM} RPM:")
        print(f"  Power: {partload_power/1000:.2f} kW (vs {self.P/1000:.2f} kW)")
        
        if hasattr(self, 'p01_n2'):
            partload_mdot = self.mdot_n2 * speed_ratio**3
            partload_pressure = self.p01_n2 * speed_ratio**3
            partload_exit_pressure = partload_pressure / (1 + 0.5 * (self.gam_n2 - 1) * self.M3_n2**2)**(self.gam_n2/(self.gam_n2-1))

            # Check starting at partload
            u_part = partload_rpm * 2 * np.pi / 60 * self.d_mean / 2
            w3u_part = self.c3u_n2 - u_part # Use n2 velocity
            w3_part = np.sqrt(w3u_part**2 + self.c3m**2)
            M_rel_part = w3_part / np.sqrt(self.gam_n2 * self.R_n2 * self.T3_n2)
            started = self.check_supersonic_start(M_rel_part, self.gam_n2)

            print(f"  N2 Mass flow: {partload_mdot*1000:.2f} g/s")
            print(f"  N2 Inlet pressure: {partload_pressure/1e5:.2f} bar")
            print(f"  N2 Exit pressure: {partload_exit_pressure/1e5:.2f} bar")
            print(f"  Relative Mach Number (Mw3): {M_rel_part:.4f}")
            print(f"  Supersonic Start Margin: {self.start_margin*100:.2f}% ({'Started' if started else 'BLOCKED'})")
        else:
            partload_mdot = self.mdot * speed_ratio**3
            partload_pressure = self.p01 * speed_ratio**3
            partload_exit_pressure = partload_pressure / (1 + 0.5 * (self.gam3 - 1) * self.M3**2)**(self.gam3/(self.gam3-1))

            # Check starting at partload
            u_part = partload_rpm * 2 * np.pi / 60 * self.d_mean / 2
            w3u_part = self.c3u - u_part
            w3_part = np.sqrt(w3u_part**2 + self.c3m**2)
            M_rel_part = w3_part / self.a3
            started = self.check_supersonic_start(M_rel_part, self.gam3)

            print(f"  Mass flow: {partload_mdot*1000:.2f} g/s")
            print(f"  Inlet pressure: {partload_pressure/1e5:.2f} bar")
            print(f"  Exit pressure: {partload_exit_pressure/1e5:.2f} bar")
            print(f"  Relative Mach Number (Mw3): {M_rel_part:.4f}")
            print(f"  Supersonic Start Margin: {self.start_margin*100:.2f}% ({'Started' if started else 'BLOCKED'})")

    def calculate_blade_stress(self, N, chord, t_max, t_shroud, w_shroud=None, r_hub=0.01, h_hub=0.015, h_tip=None, rho_mat=1150, E_mat=2.2e9, v_mat=0.3):
        """
        Calculate static bending and centrifugal stresses on the turbine blade and shroud ring.
        Assumes a face-extruded geometry: blades are extruded in the Z-axis from the face 
        of the disc at r_mean, and capped by a shroud ring.
        Uses a doubly-fixed beam model for blades (fixed at disc and shroud) under radial centrifugal UDL.
        Uses Stodola's stepwise method for a tapered disc.
        Assumes Formlabs Tough 2000 by default (rho=1150 kg/m^3, E=2.2 GPa, v=0.3).
        """
        r_mean = self.d_mean / 2
        r_outer = r_mean + chord / 2  # Disc must extend past mean radius to mount blades
        
        if h_tip is None:
            h_tip = chord
            
        if w_shroud is None:
            w_shroud = chord
        
        # 1. Doubly-fixed blade analysis
        # Centrifugal load as a UDL acting radially (N/m of axial height)
        q = rho_mat * (chord * t_max) * self.w**2 * r_mean
        
        # Root and tip bending moment (fixed-fixed beam)
        M_root = q * self.Height**2 / 12
        
        # Area moment of inertia resisting radial bending (assuming chord is approx radial)
        I_bend = t_max * chord**3 / 12
        sigma_b_pa = M_root * (chord / 2) / I_bend if I_bend > 0 else float('inf')
        sigma_b_mpa = sigma_b_pa / 1e6
        
        # Tip deflection at midspan
        delta_max = (q * self.Height**4) / (384 * E_mat * I_bend) if I_bend > 0 else float('inf')
        
        # Tip and root reaction forces (per blade)
        R_tip = q * self.Height / 2
        R_root = q * self.Height / 2
        
        # 2. Shroud ring analysis
        # Shroud's own centrifugal hoop stress
        sigma_hoop_centrifugal = rho_mat * self.w**2 * r_mean**2
        
        # Hoop stress induced by blade tip reactions pulling radially outward
        A_shroud = w_shroud * t_shroud
        sigma_hoop_blades = (N * R_tip) / (2 * np.pi * A_shroud) if A_shroud > 0 else float('inf')
        
        sigma_shroud_total_pa = sigma_hoop_centrifugal + sigma_hoop_blades
        sigma_shroud_total_mpa = sigma_shroud_total_pa / 1e6
        
        # 3. Tapered Disc Analysis (Stodola method)
        # Blade centrifugal reaction applied as an effective radial pressure at the disc's outer rim
        p_root = (N * R_root) / (2 * np.pi * r_outer * h_tip) if h_tip > 0 else 0
        
        def solve_stodola(r_inner, r_outer, h_inner, h_outer, rho, w, v, sigma_r_out_target, num_elements=50):
            C = (3 + v) / 8 * rho * w**2
            D = (1 + 3 * v) / 8 * rho * w**2
            
            radii = np.linspace(r_inner, r_outer, num_elements + 1)
            h_profile = np.linspace(h_inner, h_outer, num_elements + 1)
            h_elements = (h_profile[:-1] + h_profile[1:]) / 2
            
            def propagate(guess):
                if r_inner < 1e-9:
                    S_r = guess
                    S_t = guess
                else:
                    S_r = 0.0 # Free bore
                    S_t = guess
                    
                max_S_t = S_t
                max_S_r = S_r
                
                for i in range(num_elements):
                    r_in = radii[i]
                    r_out_elem = radii[i+1]
                    h_i = h_elements[i]
                    
                    if i > 0:
                        h_prev = h_elements[i-1]
                        S_r_new = S_r * (h_prev / h_i)
                        S_t_new = S_t + v * (S_r_new - S_r)
                        S_r, S_t = S_r_new, S_t_new
                        
                    if r_in < 1e-9:
                        A = 0
                        B = S_r
                    else:
                        B = (S_r + S_t + (C + D) * r_in**2) / 2
                        A = (r_in**2 / 2) * (S_r - S_t + (C - D) * r_in**2)
                    
                    S_r = A / r_out_elem**2 + B - C * r_out_elem**2
                    S_t = -A / r_out_elem**2 + B - D * r_out_elem**2
                    
                    max_S_t = max(max_S_t, S_t)
                    max_S_r = max(max_S_r, S_r)
                    
                S_r_final = S_r * (h_elements[-1] / h_outer)
                return S_r_final, max_S_t, max_S_r
                
            S_r_0, _, _ = propagate(0.0)
            S_r_1, _, _ = propagate(1.0)
            m = S_r_1 - S_r_0
            
            if m == 0:
                correct_guess = 0.0
            else:
                correct_guess = (sigma_r_out_target - S_r_0) / m
                
            S_r_final, max_S_t, max_S_r = propagate(correct_guess)
            return max_S_t, max_S_r

        if r_hub >= r_outer:
            r_hub = 0.0 # safety fallback
        max_sigma_t_pa, max_sigma_r_pa = solve_stodola(r_hub, r_outer, h_hub, h_tip, rho_mat, self.w, v_mat, p_root)
        
        sigma_disc_max_mpa = max(max_sigma_t_pa, max_sigma_r_pa) / 1e6
        
        print(f"Structural Analysis (N={N}, Face-Extruded):")
        print(f"  Blade UDL (q):             {q:.2f} N/m")
        print(f"  Blade Bending Stress:      {sigma_b_mpa:.2f} MPa")
        print(f"  Blade Midspan Deflection:  {delta_max*1000:.3f} mm")
        print(f"  Shroud Total Hoop Stress:  {sigma_shroud_total_mpa:.2f} MPa")
        print(f"  Tapered Disc Max Stress:   {sigma_disc_max_mpa:.2f} MPa")
        
        return sigma_b_mpa, delta_max, sigma_shroud_total_mpa, sigma_disc_max_mpa

    def sweep_blade_count(self, s_nd, b_ax_nd, c_nd, t_te_nd, w_throat_nd, mu=1.7e-5, N_min=20, N_max=60, min_print_res=0.5e-3):
        """
        Sweep blade number N and calculate physical dimensions, checking against constraints:
        1. Aspect ratio (AR >= 0.4)
        2. Manufacturing limit (t_te >= min_print_res)
        3. Kantrowitz limit (effective throat considering boundary layer displacement)
        """
        print(f"{'N':<5} | {'s (mm)':<8} | {'c (mm)':<8} | {'b_ax (mm)':<9} | {'t_te (mm)':<9} | {'AR':<6} | {'Margin%':<8} | {'Status'}")
        print("-" * 75)
        
        best_N = None
        best_loss = float('inf')
        
        if hasattr(self, 'p01_n2'):
            rho = self.rho3_n2
            w3u = self.c3u_n2 - self.u
            W_rel = np.sqrt(w3u**2 + self.c3m**2)
            M_rel = W_rel / np.sqrt(self.gam_n2 * self.R_n2 * self.T3_n2)
            gamma = self.gam_n2
        else:
            rho = self.rho_3
            w3u = self.c3u - self.u
            W_rel = np.sqrt(w3u**2 + self.c3m**2)
            M_rel = W_rel / self.a3
            gamma = self.gam3
            
        self.check_supersonic_start(M_rel, gamma)
        req_area_ratio = self.req_area_ratio
        
        valid_Ns = []
        
        for N in range(N_min, N_max + 1):
            s_phys = np.pi * self.d_mean / N
            scale = s_phys / s_nd
            
            b_ax = b_ax_nd * scale
            c = c_nd * scale
            t_te = t_te_nd * scale
            w_throat = w_throat_nd * scale
            
            AR = self.Height / c if c > 0 else 0
            
            x = c / 2  # approximate distance to throat
            Re_x = rho * W_rel * x / mu
            if Re_x > 0:
                delta_star = (0.046 * x / (Re_x**(1/5))) * (1 + 0.72 * M_rel**2)
            else:
                delta_star = 0
                
            w_eff = w_throat - 2 * delta_star
            
            effective_area_ratio = w_eff / w_throat if w_throat > 0 else 0
            start_margin = effective_area_ratio / req_area_ratio - 1 if req_area_ratio > 0 else -1
            
            status = "OK"
            if AR < 0.4:
                status = "REJECT: AR < 0.4"
            elif t_te < min_print_res:
                status = f"REJECT: t_te < {min_print_res*1000:.1f}mm"
            elif start_margin < 0:
                status = "REJECT: BLOCKED"
            else:
                valid_Ns.append(N)
                loss_proxy = N 
                if loss_proxy < best_loss:
                    best_loss = loss_proxy
                    best_N = N
                
            print(f"{N:<5} | {s_phys*1000:<8.2f} | {c*1000:<8.2f} | {b_ax*1000:<9.2f} | {t_te*1000:<9.2f} | {AR:<6.2f} | {start_margin*100:<8.2f} | {status}")
            
        print("-" * 75)
        if best_N:
            print(f"Recommended N: {best_N}")
        else:
            print("No valid N found within limits.")
        return valid_Ns, best_N

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
        print(f"  {'Nozzle Exit Gas Constant':<30} {self.R_3:<10.4g} J/kg/K")
        print(f"  {'GG Chamber Pressure':<30} {self.p01 / 1e5:<10.4g} bar")
        print(f"  {'Rotor surface speed (u)':<30} {self.u:<10.4g} m/s")
        print(f"  {'Specific heat delta (dh_use)':<30} {self.deltah_useful:<10.4g} J/kg")
        print(f"  {'Absolute Circ. Velocity (c3u)':<30} {self.c3u:<10.4g} m/s")
        print(f"  {'Absolute Meri. Velocity (c3m)':<30} {self.c3m:<10.4g} m/s")
        print(f"  {'Absolute Velocity (c3)':<30} {self.c3:<10.4g} m/s")

        # Calculate relative Mach for starting check
        w3u = self.c3u - self.u
        w3 = np.sqrt(w3u**2 + self.c3m**2)
        M_rel = w3 / self.a3
        self.check_supersonic_start(M_rel, self.gam3)

        print(f"  {'Relative Mach Number (Mw3)':<30} {M_rel:<10.4g}")
        print(f"  {'Supersonic Start Margin':<30} {self.start_margin*100:<10.4g}% ({'Started' if self.is_started else 'BLOCKED'})")

        print(f"  {'Swirl Velocity (c4u)':<30} {self.c4u:<10.4g} m/s")
        print(f"  {'Absolute Velocity at exit (c4)':<30} {self.c4:<10.4g} m/s")
        print(f"  {'Exit Temperature (T3)':<30} {self.T3:<10.4g} K")
        print(f"  {'Exit sonic velocity (a3)':<30} {self.a3:<10.4g} m/s")
        print(f"  {'Exit Pressure (p3)':<30} {self.p3/1e5:<10.4g} bar")
        print(f"  {'Exit Density (rho_3)':<30} {self.rho_3:<10.4g} kg/m3")
        print(f"  {'Required Mach Number (M3)':<30} {self.M3:<10.4g}")
        print(f"  {'Required Blade Height (H_3)':<30} {self.Height*1000:<10.4g} mm")
        print(f"  {'Turbine Efficiency (eff)':<30} {self.eff:<10.4g}")
        print(f"  {'Blade-Jet Speed Ratio':<30} {self.blade_jet_speed_ratio:<10.4g}")   
        print(f"  {'Throat Area (A)':<30} {self.A_throat*1e6:<10.4g} mm2")
        print(f"  {'Area Ratio (eps)':<30} {self.eps:<10.4g}")   
        print(f"  {'Nozzle throat length':<30} {self.nozzle_throat_length*1000:<10.4g} mm")
        print(f"  {'Nozzle exit length':<30} {self.nozzle_exit_length*1000:<10.4g} mm")
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


class SupersonicStartingGoldman:
    """
    NASA TN D-4421 (Goldman, 1968) supersonic starting analysis.

    Vortex flow (VR = const) between blade surfaces gives a 2D correction
    factor C that makes the starting limit more restrictive than 1D Kantrowitz.

    Notation follows the paper:
        M*  = V / V_cr  (critical velocity ratio)
        M*_l = lower surface (pressure side, outer radius), slower
        M*_u = upper surface (suction side, inner radius), faster
        K*  = dimensionless vortex constant, eq (23)
        Q   = vortex flow parameter, eq (34a)
        C   = 2D flow reduction factor, eq (34b)
    """

    def __init__(self, gamma):
        self.gamma = gamma
        self.gm1 = gamma - 1.0
        self.gp1 = gamma + 1.0
        self._exp = 1.0 / self.gm1
        self._Mstar_lim = np.sqrt(self.gp1 / self.gm1)

    # ── Conversions ──────────────────────────────────────────────

    @staticmethod
    def mstar_from_mach(M, gamma):
        return np.sqrt((gamma + 1) * M**2 / (2 + (gamma - 1) * M**2))

    @staticmethod
    def mach_from_mstar(Ms, gamma):
        return np.sqrt(2 * Ms**2 / ((gamma + 1) - (gamma - 1) * Ms**2))

    @staticmethod
    def prandtl_meyer_rad(M, gamma):
        gm1, gp1 = gamma - 1, gamma + 1
        return (np.sqrt(gp1 / gm1) * np.arctan(np.sqrt(gm1 / gp1 * (M**2 - 1)))
                - np.arctan(np.sqrt(M**2 - 1)))

    @staticmethod
    def mach_from_pm_rad(nu, gamma):
        gm1, gp1 = gamma - 1, gamma + 1
        def res(M):
            return (np.sqrt(gp1 / gm1) * np.arctan(np.sqrt(gm1 / gp1 * (M**2 - 1)))
                    - np.arctan(np.sqrt(M**2 - 1)) - nu)
        return root_scalar(res, bracket=[1.0001, 80], method='brentq').root

    # ── Normal shock ─────────────────────────────────────────────

    def normal_shock_p0_ratio(self, M):
        """p02/p01 across normal shock. < 1 for M > 1."""
        g, gm1, gp1 = self.gamma, self.gm1, self.gp1
        t1 = (gp1 * M**2 / (gm1 * M**2 + 2)) ** (g / gm1)
        t2 = (gp1 / (2 * g * M**2 - gm1)) ** (1.0 / gm1)
        return t1 * t2

    # ── Eq (27): K*_max ──────────────────────────────────────────

    def _solve_Kstar_max(self, Msl, Msu):
        """
        Solve eq (27) for K*_max that maximises weight flow through
        the vortex passage bounded by M*_l (outer) and M*_u (inner).

        LHS = ∫_{M*_l}^{M*_u} [1 - (K/M*_l)^2 M*^2]^{1/(γ-1)} dM*/M*
        RHS = (1 - K^2)^{1/(γ-1)} - [1 - K^2 (M*_u/M*_l)^2]^{1/(γ-1)}
        """
        e = self._exp
        r2 = (Msu / Msl) ** 2

        def residual(K):
            a = (K / Msl) ** 2

            def integ(Ms):
                v = 1.0 - a * Ms**2
                return v**e / Ms if v > 1e-15 else 0.0

            lhs, _ = quad(integ, Msl, Msu, limit=200)

            v1 = max(1.0 - K**2, 0.0)
            v2 = max(1.0 - K**2 * r2, 0.0)
            rhs = v1**e - v2**e
            return lhs - rhs

        ub = Msl / Msu * (1 - 1e-9)
        if ub < 1e-14:
            return None
        try:
            return root_scalar(residual, bracket=[1e-14, ub], method='brentq').root
        except (ValueError, RuntimeError):
            return None

    # ── Eq (34a): Q ──────────────────────────────────────────────

    def _compute_Q(self, Msl, Msu):
        """Vortex flow parameter for post-shock weight flow."""
        e = self._exp
        gp1h, gm1h = self.gp1 / 2, self.gm1 / 2

        def integ(Ms):
            v = gp1h - gm1h * Ms**2
            return v**e / Ms if v > 1e-15 else 0.0

        I, _ = quad(integ, Msl, Msu, limit=200)
        return Msl * Msu / (Msu - Msl) * I

    # ── Eq (34b): C ──────────────────────────────────────────────

    def _compute_C(self, Msl, Msu, Kmax):
        """
        2D flow reduction factor.
        Uses the analytical I_R from the eq (27) identity:
          I_L = I_R = (1-K^2)^e - [1-K^2(M*_u/M*_l)^2]^e
        so no extra quadrature is needed.
        """
        e = self._exp
        r2 = (Msu / Msl) ** 2

        v1 = max(1.0 - Kmax**2, 0.0)
        v2 = max(1.0 - Kmax**2 * r2, 0.0)
        I_R = v1**e - v2**e

        coeff = np.sqrt(self.gp1 / self.gm1) * (self.gp1 / 2) ** e
        return 1.0 - coeff * Kmax * Msu / (Msl * (Msu - Msl)) * I_R

    # ── (M*_i)_max from eqs (33) + (35) ─────────────────────────

    def max_inlet_mach(self, Msl, Msu):
        """
        Maximum inlet Mach number for supersonic starting.

        Parameters
        ----------
        Msl : float   lower-surface M* at passage throat (> 1)
        Msu : float   upper-surface M* at passage throat (> Msl)

        Returns
        -------
        float or None
        """
        if Msu <= Msl or Msl <= 0 or Msu >= self._Mstar_lim * 0.999:
            return None

        Kmax = self._solve_Kstar_max(Msl, Msu)
        if Kmax is None:
            return None

        Q = self._compute_Q(Msl, Msu)
        C = self._compute_C(Msl, Msu, Kmax)

        if C >= 1.0:
            return None

        req = Q / (1.0 - C)
        if req >= 1.0 or req <= 0:
            return None

        def res(Mi):
            return self.normal_shock_p0_ratio(Mi) - req

        if res(1.001) < 0:
            return None
        try:
            return root_scalar(res, bracket=[1.001, 80], method='brentq').root
        except (ValueError, RuntimeError):
            return None

    def max_inlet_pm_deg(self, Msl, Msu):
        """(ν_i)_max in degrees."""
        Mi = self.max_inlet_mach(Msl, Msu)
        if Mi is None:
            return None
        return np.degrees(self.prandtl_meyer_rad(Mi, self.gamma))

    # ── Practical check ──────────────────────────────────────────

    def check_starting(self, M_inlet, Msl, Msu):
        """
        Check if blade passage can swallow the starting shock.

        Returns dict with all intermediate quantities from eqs (27)-(35).
        """
        Kmax = self._solve_Kstar_max(Msl, Msu)
        Q = self._compute_Q(Msl, Msu)
        C = self._compute_C(Msl, Msu, Kmax) if Kmax is not None else None

        Mi_max = self.max_inlet_mach(Msl, Msu)
        nu_i = np.degrees(self.prandtl_meyer_rad(M_inlet, self.gamma))
        nu_i_max = np.degrees(self.prandtl_meyer_rad(Mi_max, self.gamma)) if Mi_max else None

        started = (M_inlet <= Mi_max) if Mi_max else False
        margin = (nu_i_max - nu_i) if nu_i_max else None

        return {
            'started': started,
            'M_inlet': M_inlet,
            'M_inlet_max': Mi_max,
            'nu_i_deg': nu_i,
            'nu_i_max_deg': nu_i_max,
            'margin_deg': margin,
            'Kstar_max': Kmax,
            'Q': Q,
            'C': C,
            'req_p0_ratio': Q / (1 - C) if (C is not None and C < 1) else None,
        }

    # ── 1D Kantrowitz (for comparison) ───────────────────────────

    def kantrowitz_1d_max_mach(self, area_ratio=1.0):
        """
        1D Kantrowitz starting limit for a given A_throat / A_inlet.
        Default area_ratio=1.0 is constant-area passage.
        Returns max inlet Mach for starting.
        """
        g, gm1, gp1 = self.gamma, self.gm1, self.gp1

        def f_mass(M):
            return M * (1 + gm1 / 2 * M**2) ** (-(gp1) / (2 * gm1))

        f1 = f_mass(1.0)

        def res(Mi):
            pr = self.normal_shock_p0_ratio(Mi)
            return f_mass(Mi) / f1 * (1.0 / pr) - area_ratio

        try:
            return root_scalar(res, bracket=[1.001, 80], method='brentq').root
        except (ValueError, RuntimeError):
            return None

    # ── Plotting ─────────────────────────────────────────────────

    @staticmethod
    def plot_starting_limits(gammas=None, save_path=None):
        """
        Reproduce TN D-4421 starting-limit figure.

        X-axis: ν_l (PM angle on lower blade surface at throat)
        Y-axis: (ν_i)_max (maximum inlet PM angle for starting)
        Curves: M*_u/M*_l ratios (1.0 limit = 1D Kantrowitz)
        """
        import matplotlib.pyplot as plt

        if gammas is None:
            gammas = [1.3, 1.4, 1.667]

        n = len(gammas)
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]

        ratios = [1.005, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0]
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(ratios)))

        for ax, gamma in zip(axes, gammas):
            ss = SupersonicStartingGoldman(gamma)
            Mslim = ss._Mstar_lim

            Msl_arr = np.linspace(1.005, Mslim * 0.88, 80)

            for ratio, col in zip(ratios, cmap):
                nul, nui = [], []

                for Msl in Msl_arr:
                    Msu = Msl * ratio
                    if Msu >= Mslim * 0.98:
                        continue
                    nu_max = ss.max_inlet_pm_deg(Msl, Msu)
                    if nu_max is None or nu_max <= 0:
                        continue
                    Ml = ss.mach_from_mstar(Msl, gamma)
                    if Ml <= 1:
                        continue
                    nul.append(np.degrees(ss.prandtl_meyer_rad(Ml, gamma)))
                    nui.append(nu_max)

                if len(nul) > 1:
                    lbl = '1D limit' if ratio < 1.01 else f'{ratio:.2f}'
                    ls = '--' if ratio < 1.01 else '-'
                    ax.plot(nul, nui, color=col, ls=ls, label=lbl)

            ax.set_xlabel(r'$\nu_\ell$ (deg)')
            ax.set_title(f'$\\gamma$ = {gamma}')
            ax.legend(title=r'$M^*_u / M^*_\ell$', fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        axes[0].set_ylabel(r'$(\nu_i)_{\max}$ for starting (deg)')
        fig.suptitle('Supersonic Starting Limits — NASA TN D-4421', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_starting_envelope(gamma, nu_l_deg, nu_u_deg, nu_i_actual_deg=None,
                               save_path=None):
        """
        Plot starting envelope for a specific blade design point.

        Shows where the design sits relative to the starting boundary.
        """
        import matplotlib.pyplot as plt

        ss = SupersonicStartingGoldman(gamma)

        Ml = ss.mach_from_pm_rad(np.radians(nu_l_deg), gamma)
        Mu = ss.mach_from_pm_rad(np.radians(nu_u_deg), gamma)
        Msl = ss.mstar_from_mach(Ml, gamma)
        Msu = ss.mstar_from_mach(Mu, gamma)

        ratio = Msu / Msl
        Mslim = ss._Mstar_lim

        Msl_arr = np.linspace(1.005, Mslim * 0.88, 80)
        nul, nui = [], []
        for ms in Msl_arr:
            msu = ms * ratio
            if msu >= Mslim * 0.98:
                continue
            nm = ss.max_inlet_pm_deg(ms, msu)
            if nm is None or nm <= 0:
                continue
            M = ss.mach_from_mstar(ms, gamma)
            if M <= 1:
                continue
            nul.append(np.degrees(ss.prandtl_meyer_rad(M, gamma)))
            nui.append(nm)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(nul, nui, 'b-', lw=2,
                label=f'Starting boundary ($M^*_u/M^*_\\ell$ = {ratio:.3f})')
        ax.fill_between(nul, nui, 0, alpha=0.1, color='green')
        ax.fill_between(nul, nui, max(nui) * 1.2, alpha=0.1, color='red')

        nu_i_max_design = ss.max_inlet_pm_deg(Msl, Msu)
        ax.plot(nu_l_deg, nu_i_max_design, 'rs', ms=10,
                label=f'Design point: $\\nu_{{i,max}}$ = {nu_i_max_design:.2f}°')

        if nu_i_actual_deg is not None:
            ax.axhline(nu_i_actual_deg, color='k', ls='--', lw=1,
                       label=f'Actual $\\nu_i$ = {nu_i_actual_deg:.2f}°')
            started = nu_i_actual_deg <= nu_i_max_design if nu_i_max_design else False
            status = 'STARTED' if started else 'BLOCKED'
            ax.text(0.98, 0.02, status, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', ha='right', va='bottom',
                    color='green' if started else 'red')

        ax.set_xlabel(r'$\nu_\ell$ (deg)')
        ax.set_ylabel(r'$(\nu_i)_{\max}$ (deg)')
        ax.set_title(f'Starting Envelope — $\\gamma$ = {gamma}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    gamma = 1.4

    ss = SupersonicStartingGoldman(gamma)

    # ── Sanity check: effect of M*_u/M*_l ratio on starting limit ──
    print("=== Vortex ratio effect on starting limit (gamma=1.4) ===")
    print("  (1D Kantrowitz has NO limit for constant-area passage;")
    print("   the vortex flow is what creates the restriction.)\n")

    Msl_base = 1.2
    for ratio in [1.005, 1.01, 1.05, 1.1, 1.2, 1.5]:
        Msu = Msl_base * ratio
        if Msu >= ss._Mstar_lim * 0.99:
            print(f"  ratio={ratio:.3f}  M*_u exceeds limit")
            continue
        Kmax = ss._solve_Kstar_max(Msl_base, Msu)
        Q = ss._compute_Q(Msl_base, Msu)
        C_val = ss._compute_C(Msl_base, Msu, Kmax) if Kmax else None
        Mi_max = ss.max_inlet_mach(Msl_base, Msu)
        if Mi_max:
            nu_max = np.degrees(ss.prandtl_meyer_rad(Mi_max, gamma))
            print(f"  ratio={ratio:.3f}  K*_max={Kmax:.6f}  Q={Q:.6f}  "
                  f"C={C_val:.6f}  M_i_max={Mi_max:.4f}  nu_i_max={nu_max:.2f}°")
        else:
            print(f"  ratio={ratio:.3f}  K*_max={Kmax}  Q={Q:.6f}  C={C_val}  NO STARTING SOLUTION")

    # ── Example blade check ──
    print("\n=== Example blade starting check ===")
    M_inlet = 2.0
    Msl_ex = 1.3
    Msu_ex = 1.6
    result = ss.check_starting(M_inlet, Msl_ex, Msu_ex)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.6f}")
        else:
            print(f"  {k:20s} = {v}")

    # ── Generate TN D-4421 figure ──
    print("\n=== Generating TN D-4421 starting limit plots ===")
    SupersonicStartingGoldman.plot_starting_limits(
        gammas=[1.3, 1.4, 1.667],
        save_path='tnd4421_starting_limits.png'
    )
