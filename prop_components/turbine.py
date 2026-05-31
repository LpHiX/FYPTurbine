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

        # Power-loop convergence flag (set by from_inert_gas_real)
        self.converged = False

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

        # Real loss model (Andreas Weiss) — fills phi_n, phi_r, p_v, eta_h, eff_real
        self._andreas_losses()

    def _andreas_losses(self):
        """
        Andreas Weiss supersonic loss model for a radial-inflow cantilever turbine.

            phi_n : nozzle velocity coefficient,  c3 / c3_is        (fn of M3)
            phi_r : blade  velocity coefficient,  w_exit / w_inlet  (fn of turning, Mw3)
            p_v   : ventilation / windage power loss from partial admission [W]

        Produces:
            eta_h    : hydraulic (diagram) efficiency including nozzle + blade friction
            eff_real : isentropic total-to-static efficiency, net of ventilation

        NOTE: nozzle loss cancels out of SHAFT POWER (c3 here is the actual velocity),
        so it does not appear in from_inert_gas_real's power balance. Physically phi_n
        raises the REQUIRED p01; that pressure correction is not yet applied to p01 —
        a known limitation. phi_n still correctly reduces eta_h / eff_real here.
        """
        # Relative inlet Mach and blade turning (symmetric impulse: beta_exit = beta_inlet)
        w3u = self.c3u - self.u
        w3 = np.sqrt(w3u**2 + self.c3m**2)
        self.Mw3 = w3 / self.a3
        self.deltaB_deg = np.rad2deg(2 * np.arctan2(w3u, self.c3m))

        # Nozzle velocity coefficient (supersonic, fn of nozzle exit Mach M3)
        M = self.M3
        self.phi_n = np.sqrt(
            1 - (0.0029 * M**3 - 0.0502 * M**2 + 0.2241 * M - 0.0877)
        )

        # Blade velocity coefficient (fn of relative turning deltaB and relative Mach Mw3)
        dB, Mr = self.deltaB_deg, self.Mw3
        self.phi_r = (
            0.957
            - 0.000362 * dB        - 0.0258 * Mr
            + 0.00000639 * dB**2   + 0.0674 * Mr**2
            - 0.0000000753 * dB**3 - 0.043 * Mr**3
            - 0.000238 * dB * Mr
            + 0.00000145 * dB**2 * Mr
            + 0.0000425 * dB * Mr**2
        )

        # Ventilation (partial-admission windage) power [W]
        self.p_v = (1.85 / 2) * (
            (1 - self.doa) * self.rho_3 * (self.RPM / 60)**3
            * self.d_mean**4 * 4.5 * self.Height
        )

        # Hydraulic efficiency and net isentropic total-to-static efficiency
        nu = self.u / self.c3                          # actual blade-jet speed ratio
        self.eta_h = 2 * self.phi_n**2 * nu * (np.cos(self.beta) - nu) * (1 + self.phi_r)
        self.deltah_s = self.deltah_ambis / self.phi_n**2   # isentropic available KE
        self.eff_real = self.eta_h - self.p_v / (self.mdot * self.deltah_s)

        # --- Nozzle total-pressure-loss correction on required inlet stagnation ---
        # from_inert_gas drives p3 -> p_e via the ISENTROPIC p01<->p3 map, so its
        # self.p01 is really the nozzle EXIT stagnation p03. A real nozzle loses
        # total pressure, so the true upstream p01 must be higher to reach the same
        # c3 (hence same M3, p3). Invert phi_n^2 = [1-(p3/p03)^k]/[1-(p3/p01)^k]:
        k = (self.gam3 - 1) / self.gam3
        self.p03 = self.p01                            # rename: loop output = exit stagnation
        bracket = 1 - (1 - (self.p3 / self.p03)**k) / self.phi_n**2
        if bracket > 0:
            self.p01 = self.p3 * bracket**(-1 / k)     # true inlet stagnation
        # else: loss too large to expand to p_e at this M3; leave p01 = p03 (flagged)

        # Choked throat uses inlet stagnation (loss assumed downstream of throat),
        # so the corrected p01 shrinks the throat for the same mdot.
        self.A_throat = self.mdot / (
            self.p01 / np.sqrt(self.T01) * np.sqrt(self.gam3 / self.R_3)
            * ((2 / (self.gam3 + 1))**((self.gam3 + 1) / (2 * (self.gam3 - 1))))
        )
        self.eps = self.A3 / self.A_throat
        self.nozzle_throat_length = self.A_throat / self.nozzles / self.Height
        self.nozzle_exit_length = self.eps * self.A_throat / self.nozzles / self.Height

    def from_inert_gas_real(self, R, gam, T01, nozzles, tol=1e-3, max_iter=50):
        """
        Loss-consistent sizing. The ideal velocity triangle (built in __init__ from
        self.P) under-predicts the pressure/Mach needed once blade friction (phi_r)
        and ventilation (p_v) are charged. This iterates an internal driving power
        until the ACTUAL delivered shaft power equals the requested self.P:

            P_shaft = P_internal * (1 + phi_r) / 2  -  p_v

        Every state variable (c3, M3, p01, Height, eff, eff_real, ...) is rebuilt at
        the converged internal power, so contour maps reflect the true numbers.
        """
        self.converged = False
        P_target = self.P
        P_internal = P_target          # initial guess: real == ideal
        P_shaft = P_target

        for it in range(1, max_iter + 1):
            self.power_iters = it
            # Rebuild the velocity triangle from the working internal power
            self.deltah_useful = P_internal / self.mdot
            self.c3u = self.deltah_useful / (2 * self.u) + self.u
            self.c3  = self.c3u / np.cos(self.beta)
            self.c3m = self.c3 * np.sin(self.beta)
            self.c4u = self.u - (self.c3u - self.u)
            self.c4  = np.sqrt(self.c4u**2 + self.c3m**2)

            # Solve gas path + losses for this triangle
            self.from_inert_gas(R, gam, T01, nozzles)

            # Actual shaft power after blade friction + windage
            P_shaft = P_internal * (1 + self.phi_r) / 2 - self.p_v

            if abs(P_shaft - P_target) / P_target < tol:
                self.converged = True
                break

            # Near-direct update: invert P_shaft(P_internal) holding phi_r, p_v fixed
            P_internal = (P_target + self.p_v) * 2 / (1 + self.phi_r)

        self.P_internal = P_internal
        self.P_shaft = P_shaft
        self.P = P_target              # keep reported P as the requested shaft power
        return self.converged

    def partload_rpm(self, partload_rpm, verbose=True):
        """
        Off-design behaviour of THIS turbine (FIXED hardware) driving a pump load.

        Nothing is re-sized. Blade height, nozzle throat/exit areas, area ratio
        eps, mean diameter, nozzle count and flow angle are all held at their
        design (as-built) values. With fixed nozzle geometry and a fixed
        stagnation temperature T01, the rotor-inlet velocity triangle is fixed
        too: the area ratio eps sets M3, and M3 + T01 set c3 (hence c3u, c3m).
        The ONLY kinematic change with shaft speed is the blade speed u = omega*r.

        The pump LOAD follows the affinity law:
            P_pump = P_design * (N / N_design)^3

        The gas supply needed to deliver that on the fixed hardware is solved
        from the impulse-stage power balance (identical to the one the sizer
        uses, just inverted for the supply instead of the geometry):

            dh   = (1 + phi_r(N)) * u * (c3u - u)        specific shaft work [J/kg]
            P    = mdot*dh - p_v                          with p_v ∝ mdot * N^3
                 => mdot = P_pump / (dh - p_v/mdot)       (exact, linear in mdot)
            p01  = p01_design * (mdot / mdot_design)      (choked throat, fixed A*)

        Losses are RE-EVALUATED at the part-load operating point: phi_r and the
        ventilation power p_v change with N (phi_n and the inlet triangle are
        fixed by the geometry + T01). At N == N_design with the design pump load
        this reproduces the design point exactly.

        Returns a dict of the off-design state. Geometry fields (Height,
        A_throat, eps, nozzles) are echoed unchanged to make the fixed-hardware
        assumption explicit.
        """
        if not hasattr(self, 'phi_r'):
            raise RuntimeError(
                "partload_rpm needs the real loss model. Size the turbine with "
                "from_inert_gas_real(...) (or from_inert_gas) first.")

        ratio = partload_rpm / self.RPM
        omega = partload_rpm * 2 * np.pi / 60
        u = omega * self.d_mean / 2

        # Fixed rotor-inlet triangle (geometry + T01 fixed -> unchanged from design)
        c3, c3u, c3m, a3 = self.c3, self.c3u, self.c3m, self.a3

        # Relative-flow state at the new blade speed
        w3u = c3u - u
        w3 = np.hypot(w3u, c3m)
        Mw3 = w3 / a3
        deltaB_deg = np.rad2deg(2 * np.arctan2(w3u, c3m))

        # Blade velocity coefficient re-evaluated (Andreas); phi_n fixed (M3 fixed)
        dB, Mr = deltaB_deg, Mw3
        phi_r = (
            0.957
            - 0.000362 * dB        - 0.0258 * Mr
            + 0.00000639 * dB**2   + 0.0674 * Mr**2
            - 0.0000000753 * dB**3 - 0.043 * Mr**3
            - 0.000238 * dB * Mr
            + 0.00000145 * dB**2 * Mr
            + 0.0000425 * dB * Mr**2
        )
        phi_r = min(max(phi_r, 0.0), 1.0)

        # Specific shaft work on the fixed hardware (per unit mass)
        dh = (1 + phi_r) * u * (c3u - u)
        runaway = dh <= 0.0          # u >= c3u: stage can do no net positive work

        # Pump load and the gas supply that meets it on the fixed throat.
        # p_v = (p_v_design / mdot_design) * (N/N_d)^3 * mdot   (windage ∝ mdot*N^3),
        # so mdot factors out:  P_pump = mdot*(dh - B),  B = p_v per unit mdot.
        P_pump = self.P * ratio**3
        B = (self.p_v / self.mdot) * ratio**3
        denom = dh - B
        feasible = (not runaway) and (denom > 0)

        if feasible:
            mdot = P_pump / denom
            p_v = B * mdot
            mratio = mdot / self.mdot
            p01 = self.p01 * mratio                 # choked throat, fixed A* -> p01 ∝ mdot
            p3 = self.p3 * mratio
            rho_3 = self.rho_3 * mratio
            P_shaft = mdot * dh - p_v               # == P_pump by construction
            torque = P_shaft / omega
            nu = u / c3
            eta_h = 2 * self.phi_n**2 * nu * (np.cos(self.beta) - nu) * (1 + phi_r)
            deltah_s = (0.5 * c3**2) / self.phi_n**2
            eff_real = eta_h - p_v / (mdot * deltah_s)
        else:
            mdot = p_v = p01 = p3 = rho_3 = P_shaft = torque = nu = eta_h = eff_real = float('nan')

        result = {
            'rpm': partload_rpm, 'ratio': ratio, 'u': u, 'feasible': feasible, 'runaway': runaway,
            'P_pump': P_pump, 'P_shaft': P_shaft, 'torque': torque, 'dh': dh,
            'mdot': mdot, 'p01': p01, 'p3': p3, 'rho_3': rho_3,
            'Mw3': Mw3, 'deltaB_deg': deltaB_deg, 'phi_r': phi_r, 'phi_n': self.phi_n,
            'p_v': p_v, 'eta_h': eta_h, 'eff_real': eff_real, 'blade_jet_ratio': nu,
            # fixed hardware (unchanged) — echoed to make the assumption explicit
            'Height': self.Height, 'A_throat': self.A_throat, 'eps': self.eps,
            'nozzles': self.nozzles, 'd_mean': self.d_mean, 'M3': self.M3,
        }

        if verbose:
            d = lambda a: getattr(self, a, float('nan'))
            print(f"Part-load @ {partload_rpm:.0f} RPM  (design {self.RPM:.0f}, N/N_d = {ratio:.3f})  "
                  f"[FIXED hardware, pump-affinity load]:")
            if not feasible:
                why = "u >= c3u (runaway)" if runaway else "windage exceeds available work"
                print(f"  ** INFEASIBLE: {why} — turbine cannot drive this pump load here **")
            else:
                print(f"  {'Pump load P (~N^3)':<30}{P_shaft/1000:9.3f} kW    (design {self.P/1000:.3f})")
                print(f"  {'Shaft torque':<30}{torque:9.4f} N·m")
                print(f"  {'Required gas mdot':<30}{mdot*1000:9.2f} g/s   (design {self.mdot*1000:.2f})")
                print(f"  {'Required inlet p01':<30}{p01/1e5:9.2f} bar   (design {d('p01')/1e5:.2f})")
                print(f"  {'Blade speed u':<30}{u:9.2f} m/s   (nu = u/c3 = {nu:.3f})")
                print(f"  {'Relative Mach Mw3':<30}{Mw3:9.3f}       (design {d('Mw3'):.3f})")
                print(f"  {'Blade coeff phi_r':<30}{phi_r:9.4f}      (design {d('phi_r'):.4f})")
                print(f"  {'REAL efficiency eff_real':<30}{eff_real:9.4f}      (design {d('eff_real'):.4f})")
                pv_frac = p_v / P_shaft * 100 if P_shaft > 0 else float('nan')
                print(f"  {'Ventilation p_v':<30}{p_v:9.2f} W     ({pv_frac:.1f}% of shaft power)")
            print(f"  {'(fixed) Blade height H_3':<30}{self.Height*1000:9.3f} mm")
            print(f"  {'(fixed) Throat area A*':<30}{self.A_throat*1e6:9.3f} mm²")
            print(f"  {'(fixed) Area ratio eps':<30}{self.eps:9.3f}")
            print(f"  {'(fixed) Nozzle exit Mach M3':<30}{self.M3:9.3f}")

        return result

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

        # Relative Mach at rotor inlet (starting analysis lives in SupersonicStartingGoldman)
        w3u = self.c3u - self.u
        w3 = np.sqrt(w3u**2 + self.c3m**2)
        M_rel = w3 / self.a3

        print(f"  {'Relative Mach Number (Mw3)':<30} {M_rel:<10.4g}")

        print(f"  {'Swirl Velocity (c4u)':<30} {self.c4u:<10.4g} m/s")
        print(f"  {'Absolute Velocity at exit (c4)':<30} {self.c4:<10.4g} m/s")
        print(f"  {'Exit Temperature (T3)':<30} {self.T3:<10.4g} K")
        print(f"  {'Exit sonic velocity (a3)':<30} {self.a3:<10.4g} m/s")
        print(f"  {'Exit Pressure (p3)':<30} {self.p3/1e5:<10.4g} bar")
        print(f"  {'Exit Density (rho_3)':<30} {self.rho_3:<10.4g} kg/m3")
        print(f"  {'Required Mach Number (M3)':<30} {self.M3:<10.4g}")
        print(f"  {'Required Blade Height (H_3)':<30} {self.Height*1000:<10.4g} mm")
        print(f"  {'Turbine Efficiency (eff)':<30} {self.eff:<10.4g}")
        if hasattr(self, 'eff_real'):
            print(f"  {'Real Efficiency (eff_real)':<30} {self.eff_real:<10.4g}")
            print(f"  {'Hydraulic Eff (eta_h)':<30} {self.eta_h:<10.4g}")
            print(f"  {'Nozzle Coeff (phi_n)':<30} {self.phi_n:<10.4g}")
            print(f"  {'Blade Coeff (phi_r)':<30} {self.phi_r:<10.4g}")
            print(f"  {'Blade Turning (deltaB)':<30} {self.deltaB_deg:<10.4g} deg")
            print(f"  {'Ventilation Power (p_v)':<30} {self.p_v:<10.4g} W")
            if hasattr(self, 'P_internal'):
                print(f"  {'Internal Power (P_int)':<30} {self.P_internal/1000:<10.4g} kW")
                print(f"  {'Shaft Power (P_shaft)':<30} {self.P_shaft/1000:<10.4g} kW")
                print(f"  {'Power loop converged':<30} {str(self.converged):<10}")
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
        return 1.0 - coeff * Kmax * Msu / (Msu - Msl) * I_R

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

    # ── Plotting ─────────────────────────────────────────────────

    @staticmethod
    def plot_starting_limits(save_path=None):
        """
        Reproduce TN D-4421 starting-limit figure (Figure 9-18).

        X-axis: Upper-surface Prandtl-Meyer angle, omega_u, deg
        Y-axis: Maximum inlet Prandtl-Meyer angle, omega_{in,max}, deg
        Curves: Lower-surface Prandtl-Meyer angle, omega_l, deg
        """
        import matplotlib.pyplot as plt

        gamma = 1.4
        fig, ax = plt.subplots(figsize=(8, 10))

        ss = SupersonicStartingGoldman(gamma)
        
        nu_l_degs = np.arange(0, 125, 5)
        nu_max_possible = np.degrees(np.pi / 2 * (np.sqrt((gamma + 1) / (gamma - 1)) - 1))

        # Plot the 1D Kantrowitz limit (dashed line)
        nu_1d_x = np.linspace(0, min(140, nu_max_possible - 0.1), 100)
        nu_1d_y = []
        for nu in nu_1d_x:
            if nu == 0:
                nu_1d_y.append(0.0)
                continue
            try:
                Ml = ss.mach_from_pm_rad(np.radians(nu), gamma)
            except ValueError:
                nu_1d_y.append(np.nan)
                continue
            Msl = ss.mstar_from_mach(Ml, gamma)
            Q = ss._compute_Q(Msl, Msl + 1e-6)
            def res(Mi):
                return ss.normal_shock_p0_ratio(Mi) - Q
            try:
                Mi_1d = root_scalar(res, bracket=[1.0001, 80], method='brentq').root
                nu_1d_y.append(np.degrees(ss.prandtl_meyer_rad(Mi_1d, gamma)))
            except ValueError:
                nu_1d_y.append(np.nan)
        ax.plot(nu_1d_x, nu_1d_y, 'k--', lw=1.5)

        for nu_l in nu_l_degs:
            if nu_l >= nu_max_possible:
                continue
            
            if nu_l == 0:
                Ml = 1.0001
            else:
                try:
                    Ml = ss.mach_from_pm_rad(np.radians(nu_l), gamma)
                except ValueError:
                    continue
            
            Msl = ss.mstar_from_mach(Ml, gamma)
            
            nu_u_arr = []
            nu_i_arr = []
            
            for nu_u in np.linspace(nu_l + 0.1, 140, 150):
                try:
                    Mu = ss.mach_from_pm_rad(np.radians(nu_u), gamma)
                except ValueError:
                    break
                Msu = ss.mstar_from_mach(Mu, gamma)
                
                if Msu >= ss._Mstar_lim * 0.999:
                    continue
                    
                nu_max = ss.max_inlet_pm_deg(Msl, Msu)
                if nu_max is None or nu_max <= 0:
                    continue
                    
                nu_u_arr.append(nu_u)
                nu_i_arr.append(nu_max)
                
            if len(nu_u_arr) > 0:
                ax.plot(nu_u_arr, nu_i_arr, 'k-', lw=1.2)
                # Add label at the end of the line
                ax.text(nu_u_arr[-1] + 1.5, nu_i_arr[-1], f'{int(nu_l)}', 
                        va='center', fontsize=9)

        ax.set_xlabel(r'Upper-surface Prandtl-Meyer angle, $\omega_u$, deg', fontsize=11)
        ax.set_ylabel(r'Maximum inlet Prandtl-Meyer angle, $\omega_{in,\max}$, deg', fontsize=11)
        ax.set_title(f'Maximum Prandtl-Meyer angle for supersonic starting. Specific heat ratio, {gamma}', 
                     fontsize=12, pad=20)
        
        ax.set_xlim(0, 140)
        ax.set_ylim(0, 150)
        
        # Add legend text for the curves
        ax.text(125, 145, 'Lower-surface\nPrandtl-Meyer\nangle,\n$\\omega_l$,\ndeg', 
                ha='center', va='top', fontsize=10)
        
        ax.set_xticks(np.arange(0, 141, 20))
        ax.set_yticks(np.arange(0, 141, 20))
        
        # Clean up axes to match the paper style
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='in', length=6, labelsize=10)
        
        # A dashed 1-to-1 line or similar might be useful but was not in the original, we will skip it
        # Actually there is a dashed line for the limit, let's draw a dashed line for the uppermost curve limit
        # In the original plot, the uppermost limit is a dashed line (Kantrowitz 1D limit probably)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
