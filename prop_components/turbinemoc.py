import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class SupersonicTurbineMOC:
    """
    Method of Characteristics (MOC) solver for generating supersonic turbine blade profiles.
    """
    
    def __init__(self, gamma=1.4, mach_inlet=1.7, mach_lower=1.3, mach_upper=2.2, beta_inlet_deg=75.0, dv=0.01):
        """
        Initialize the MOC solver with operating conditions.
        """
        self.gamma = gamma
        self.mach_inlet = mach_inlet
        self.mach_outlet = mach_inlet  # Assuming symmetric
        self.mach_lower = mach_lower
        self.mach_upper = mach_upper
        self.beta_inlet_deg = beta_inlet_deg
        self.beta_inlet = np.deg2rad(beta_inlet_deg)
        self.beta_outlet = -self.beta_inlet
        self.dv = dv
        
        # Will be populated after calling .generate()
        self.coords = {}
        self.results = {}
        
    @staticmethod
    def safe_asin(x):
        """Safe arcsin to avoid domain errors from float precision."""
        return np.arcsin(np.clip(x, -1.0, 1.0))

    def m_star(self, M):
        """Calculate critical velocity ratio (M*)."""
        g = self.gamma
        return np.sqrt(((g + 1) / 2 * M**2) / (1 + (g - 1) / 2 * M**2))

    def prandtl_meyer(self, M):
        """Calculate the Prandtl-Meyer function (nu) for a given Mach number."""
        g = self.gamma
        ms = self.m_star(M)
        term1 = np.pi / 4 * (np.sqrt((g + 1)/(g - 1)) - 1)
        term2 = 0.5 * (np.sqrt((g + 1)/(g - 1)) * self.safe_asin((g - 1) * ms**2 - g) + 
                       self.safe_asin((g + 1)/ms**2 - g))
        return term1 + term2

    def generate(self):
        """
        Runs the MOC iteration to generate the blade profile coordinates.
        Returns a dictionary of unrotated and rotated coordinate arrays.
        """
        nu_inlet = self.prandtl_meyer(self.mach_inlet)
        nu_outlet = self.prandtl_meyer(self.mach_outlet)
        nu_lower = self.prandtl_meyer(self.mach_lower)
        nu_upper = self.prandtl_meyer(self.mach_upper)

        alpha_lower_inlet = self.beta_inlet - (nu_inlet - nu_lower)
        alpha_lower_outlet = self.beta_outlet - (nu_outlet - nu_lower)
        alpha_upper_inlet = self.beta_inlet - (nu_upper - nu_inlet)
        alpha_upper_outlet = self.beta_outlet - (nu_upper - nu_outlet)

        g = self.gamma
        dv = self.dv

        # --- Lower Flow Simulation ---
        def u_i_lower(Rstar):
            return -np.arcsin(np.sqrt(0.5*(g+1)*Rstar**2 - 0.5*(g-1)))

        Rlk1_low = 1 / self.m_star(self.mach_lower)
        phi_k1_low = 0
        u_i_k1_low = u_i_lower(Rlk1_low)
        xlstar_k1_low, ylstar_k1_low = 0, Rlk1_low

        x_l, y_l, x_l_k, y_l_k = [], [], [], []
        xn_l, yn_l, xn_l_k, yn_l_k = [], [], [], []
        mach_l = []

        steps_lower = int(np.ceil((nu_inlet - nu_lower) / dv))
        for k in range(steps_lower, 0, -1):
            phi_k = nu_inlet - nu_lower - k * dv
            fRkstar = 2 * nu_inlet - np.pi / 2 * (np.sqrt((g + 1)/(g - 1)) - 1) - 2 * k * dv
            
            def f_low(Rstar):
                return (np.sqrt((g + 1)/(g - 1)) * self.safe_asin((g - 1) / Rstar**2 - g) + 
                        self.safe_asin((g + 1) * Rstar**2 - g) - fRkstar)
            
            Rstar = fsolve(f_low, 0.9)[0]
            xkstar = -Rstar * np.sin(phi_k)
            ykstar = Rstar * np.cos(phi_k)

            u_i_k = u_i_lower(Rstar)
            mi_k = np.tan(0.5 * (phi_k + phi_k1_low) + 0.5 * (u_i_k + u_i_k1_low))
            mbar_k = np.tan(phi_k1_low)
            xstar_l = ((ylstar_k1_low - mbar_k * xlstar_k1_low) - (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)
            ystar_l = (mi_k *(ylstar_k1_low - mbar_k * xlstar_k1_low) - mbar_k * (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)

            phi_k1_low, u_i_k1_low = phi_k, u_i_k
            xlstar_k1_low, ylstar_k1_low = xstar_l, ystar_l

            x_l.append(xstar_l * np.cos(alpha_lower_inlet) - ystar_l * np.sin(alpha_lower_inlet))
            y_l.append(xstar_l * np.sin(alpha_lower_inlet) + ystar_l * np.cos(alpha_lower_inlet))
            x_l_k.append(xkstar * np.cos(alpha_lower_inlet) - ykstar * np.sin(alpha_lower_inlet))
            y_l_k.append(ykstar * np.cos(alpha_lower_inlet) + xkstar * np.sin(alpha_lower_inlet))

            xn_l.append(xstar_l)
            yn_l.append(ystar_l)
            xn_l_k.append(xkstar)
            yn_l_k.append(ykstar)
            mach_l.append(self.convert_critical_to_mach(1.0 / Rstar, g))

        # --- Upper Flow Simulation ---
        def u_i_upper(Rstar):
            return np.arcsin(np.sqrt(0.5*(g+1)*Rstar**2 - 0.5*(g-1)))

        Rlk1_up = 1 / self.m_star(self.mach_upper)
        phi_k1_up = 0
        u_i_k1_up = u_i_upper(Rlk1_up)
        xlstar_k1_up, ylstar_k1_up = 0, Rlk1_up

        x_u, y_u, x_u_k, y_u_k = [], [], [], []
        xn_u, yn_u, xn_u_k, yn_u_k = [], [], [], []
        mach_u = []

        steps_upper = int(np.ceil((nu_upper - nu_inlet) / dv))
        for k in range(steps_upper, 0, -1):
            phi_k = nu_upper - nu_inlet - k * dv
            fRkstar = 2 * nu_inlet - np.pi / 2 * (np.sqrt((g + 1)/(g - 1)) - 1) + 2 * k * dv
            
            def f_up(Rstar):
                return (np.sqrt((g + 1)/(g - 1)) * self.safe_asin((g - 1) / Rstar**2 - g) + 
                        self.safe_asin((g + 1) * Rstar**2 - g) - fRkstar)
            
            Rstar = fsolve(f_up, 0.5)[0]
            xkstar = -Rstar * np.sin(phi_k)
            ykstar = Rstar * np.cos(phi_k)

            u_i_k = u_i_upper(Rstar)
            mi_k = np.tan(0.5 * (phi_k + phi_k1_up) + 0.5 * (u_i_k + u_i_k1_up))
            mbar_k = np.tan(phi_k1_up)
            xstar_l = ((ylstar_k1_up - mbar_k * xlstar_k1_up) - (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)
            ystar_l = (mi_k *(ylstar_k1_up - mbar_k * xlstar_k1_up) - mbar_k * (ykstar - mi_k * xkstar)) / (mi_k - mbar_k)

            phi_k1_up, u_i_k1_up = phi_k, u_i_k
            xlstar_k1_up, ylstar_k1_up = xstar_l, ystar_l

            x_u.append(xstar_l * np.cos(alpha_upper_inlet) - ystar_l * np.sin(alpha_upper_inlet))
            y_u.append(xstar_l * np.sin(alpha_upper_inlet) + ystar_l * np.cos(alpha_upper_inlet))
            x_u_k.append(xkstar * np.cos(alpha_upper_inlet) - ykstar * np.sin(alpha_upper_inlet))
            y_u_k.append(ykstar * np.cos(alpha_upper_inlet) + xkstar * np.sin(alpha_upper_inlet))

            xn_u.append(xstar_l)
            yn_u.append(ystar_l)
            xn_u_k.append(xkstar)
            yn_u_k.append(ykstar)
            mach_u.append(self.convert_critical_to_mach(1.0 / Rstar, g))

        # Store Output Data
        self.coords = {
            'lower_rot': {'x': np.array(x_l), 'y': np.array(y_l), 'x_k': np.array(x_l_k), 'y_k': np.array(y_l_k)},
            'upper_rot': {'x': np.array(x_u), 'y': np.array(y_u), 'x_k': np.array(x_u_k), 'y_k': np.array(y_u_k)},
            'lower': {'x': np.array(xn_l), 'y': np.array(yn_l), 'x_k': np.array(xn_l_k), 'y_k': np.array(yn_l_k)},
            'upper': {'x': np.array(xn_u), 'y': np.array(yn_u), 'x_k': np.array(xn_u_k), 'y_k': np.array(yn_u_k)}
        }
        self.surface_mach = {
            'lower': np.array(mach_l),
            'upper': np.array(mach_u)
        }

        self.results = {
            'nu_inlet': nu_inlet, 'nu_outlet': nu_outlet, 'nu_lower': nu_lower, 'nu_upper': nu_upper,
            'alpha_lower_inlet': alpha_lower_inlet, 'alpha_lower_outlet': alpha_lower_outlet,
            'alpha_upper_inlet': alpha_upper_inlet, 'alpha_upper_outlet': alpha_upper_outlet,
            'Rl': 1 / self.m_star(self.mach_lower),
            'Ru': 1 / self.m_star(self.mach_upper),
            'chord': -2 * x_l[-1] if len(x_l) > 0 else 0,
            'steps_lower': steps_lower,
            'steps_upper': steps_upper
        }
        return self.coords

    def print_summary(self):
        """Prints out the physical and geometric summaries of the generation."""
        if not self.results:
            self.generate()
            
        res = self.results
        print("Inputs:")
        print(f"  {'Gamma:':<20} {self.gamma:<10.4g}")
        print(f"  {'Mach Inlet:':<20} {self.mach_inlet:<10.4g}")
        print(f"  {'Mach Outlet:':<20} {self.mach_outlet:<10.4g}")
        print(f"  {'Mach Lower:':<20} {self.mach_lower:<10.4g}")
        print(f"  {'Mach Upper:':<20} {self.mach_upper:<10.4g}")
        print(f"  {'nu_inlet:':<20} {res['nu_inlet']:<10.4g} rad, {np.rad2deg(res['nu_inlet']):<10.4g} deg")
        print(f"  {'nu_lower:':<20} {res['nu_lower']:<10.4g} rad, {np.rad2deg(res['nu_lower']):<10.4g} deg")
        print(f"  {'nu_upper:':<20} {res['nu_upper']:<10.4g} rad, {np.rad2deg(res['nu_upper']):<10.4g} deg")
        print(f"  {'Beta Inlet:':<20} {self.beta_inlet:<10.4g} rad, {np.rad2deg(self.beta_inlet):<10.4g} deg")
        print(f"  {'Alpha Lower Inlet:':<20} {res['alpha_lower_inlet']:<10.4g} rad, {np.rad2deg(res['alpha_lower_inlet']):<10.4g} deg")
        print(f"  {'Alpha Upper Inlet:':<20} {res['alpha_upper_inlet']:<10.4g} rad, {np.rad2deg(res['alpha_upper_inlet']):<10.4g} deg")
        
        print("\nResults:")
        print(f"  {'R_lower:':<20} {res['Rl']:<10.4g}")
        print(f"  {'R_upper:':<20} {res['Ru']:<10.4g}")
        print(f"  {'chord:':<20} {res['chord']:<10.4g}")
        
        # Accessing the unrotated endpoints
        l_x_end = self.coords['lower']['x'][-1] if len(self.coords['lower']['x']) else 0
        l_y_end = self.coords['lower']['y'][-1] if len(self.coords['lower']['y']) else 0
        u_x_end = self.coords['upper']['x'][-1] if len(self.coords['upper']['x']) else 0
        u_y_end = self.coords['upper']['y'][-1] if len(self.coords['upper']['y']) else 0
        
        print(f"  {'lower arc x:':<20} {l_x_end:<10.4g}")
        print(f"  {'lower arc y:':<20} {l_y_end:<10.4g}")
        print(f"  {'upper arc x:':<20} {u_x_end:<10.4g}")
        print(f"  {'upper arc y:':<20} {u_y_end:<10.4g}")

    def plot_blade(self, unrotated=True, full_passage=True):
        """
        Plots the characteristic mesh.
        """
        if not self.coords:
            self.generate()

        plt.figure(figsize=(20, 15))
        
        # Toggle between plotting the unrotated (xn, yn) or the rotated (x, y) coordinate frames
        lower_key = 'lower' if unrotated else 'lower_rot'
        upper_key = 'upper' if unrotated else 'upper_rot'

        x_l, y_l = self.coords[lower_key]['x'], self.coords[lower_key]['y']
        x_l_k, y_l_k = self.coords[lower_key]['x_k'], self.coords[lower_key]['y_k']
        
        x_u, y_u = self.coords[upper_key]['x'], self.coords[upper_key]['y']
        x_u_k, y_u_k = self.coords[upper_key]['x_k'], self.coords[upper_key]['y_k']

        # Transition Lines
        plt.plot(x_l, y_l, '-')
        plt.plot(x_u, y_u, '-')
        plt.plot(x_l_k, y_l_k, 'x-', color='red')
        plt.plot(x_u_k, y_u_k, 'x-', color='red')
        
        if full_passage:
            plt.plot(-x_l, y_l, '-')
            plt.plot(-x_u, y_u, '-')
            plt.plot(-x_l_k, y_l_k, 'x-', color='red')
            plt.plot(-x_u_k, y_u_k, 'x-', color='red')

        # Characteristics
        for i in range(self.results['steps_lower']):
            plt.plot([x_l[i], x_l_k[i]], [y_l[i], y_l_k[i]], 'k-', linewidth=0.5)
            if full_passage:
                plt.plot([-x_l[i], -x_l_k[i]], [y_l[i], y_l_k[i]], 'k-', linewidth=0.5)
        for i in range(self.results['steps_upper']):
            plt.plot([x_u[i], x_u_k[i]], [y_u[i], y_u_k[i]], 'k-', linewidth=0.5)
            if full_passage:
                plt.plot([-x_u[i], -x_u_k[i]], [y_u[i], y_u_k[i]], 'k-', linewidth=0.5)

        Rl = self.results['Rl']
        Ru = self.results['Ru']
        a_l_in = self.results['alpha_lower_inlet']
        a_u_in = self.results['alpha_upper_inlet']

        # Plot circular arcs if using the unrotated scheme
        if unrotated:
            plt.plot(Rl * np.cos(np.linspace(np.pi/2 - a_l_in, np.pi/2 + a_l_in, 100)), 
                     Rl * np.sin(np.linspace(np.pi/2 - a_l_in, np.pi/2 + a_l_in, 100)), 'k-')
            plt.plot(Ru * np.cos(np.linspace(np.pi/2 - a_u_in, np.pi/2 + a_u_in, 100)), 
                     Ru * np.sin(np.linspace(np.pi/2 - a_u_in, np.pi/2 + a_u_in, 100)), 'k-')
            
            # Close the trailing edge
            if len(x_u) > 0 and len(x_l) > 0:
                plt.plot([x_u[-1], x_l[-1]], 
                         [y_u[-1], y_u[-1] + (x_l[-1] - x_u[-1]) * np.tan(self.beta_inlet - a_u_in)], 'k-')
                if full_passage:
                    plt.plot([-x_u[-1], -x_l[-1]], 
                             [y_u[-1], y_u[-1] + (x_l[-1] - x_u[-1]) * np.tan(self.beta_inlet - a_u_in)], 'k-')
        else:
            # Rotated scheme
            plt.plot(Rl * np.cos(np.linspace(np.pi/2 - a_l_in, np.pi/2 + a_l_in, 100)), 
                     Rl * np.sin(np.linspace(np.pi/2 - a_l_in, np.pi/2 + a_l_in, 100)), 'k-')
            plt.plot(Ru * np.cos(np.linspace(np.pi/2 - a_u_in, np.pi/2 + a_u_in, 100)), 
                     Ru * np.sin(np.linspace(np.pi/2 - a_u_in, np.pi/2 + a_u_in, 100)), 'k-')
            
            if len(x_u) > 0 and len(x_l) > 0:
                plt.plot([x_u[-1], x_l[-1]], 
                         [y_u[-1], y_u[-1] + (x_l[-1] - x_u[-1]) * np.tan(self.beta_inlet)], 'k-')
                if full_passage:
                    plt.plot([-x_u[-1], -x_l[-1]], 
                             [y_u[-1], y_u[-1] + (x_l[-1] - x_u[-1]) * np.tan(self.beta_inlet)], 'k-')
                    
                    # Full continuous blade outline
                    plt.plot(np.concatenate([np.flip(x_l), Rl * np.cos(np.linspace(np.pi/2 + a_l_in, np.pi/2 - a_l_in, 100)), -x_l]), 
                             np.concatenate([np.flip(y_l), Rl * np.sin(np.linspace(np.pi/2 + a_l_in, np.pi/2 - a_l_in, 100)), y_l]) - y_l[-1] + (y_u[-1] + (x_l[-1] - x_u[-1]) * np.tan(self.beta_inlet)), 'k-')

        plt.axis('equal')
        plt.title('Supersonic Turbine Blade Profile (MOC)')
        plt.show()

    def surface_mach_distribution(self, plot=True):
        """
        Compute and optionally plot the surface Mach number distribution
        vs. normalized axial chord (x/c_ax, 0 at blade inlet, 1 at blade outlet).

        Uses only the blade wall points from the MOC characteristics.
        The blade is symmetric (inlet mirrors outlet), so each half is mirrored
        about the throat plane.  The upper surface is extended to span the
        full axial chord via trailing-edge line segments at M_inlet.

        Returns
        -------
        dict with keys:
            'lower': {'x_norm': ndarray, 'mach': ndarray}
            'upper': {'x_norm': ndarray, 'mach': ndarray}
        """
        if not self.coords:
            self.generate()

        res = self.results

        # --- Blade wall points in the rotated (physical) frame ---
        # Stored order: index 0 = nearest throat, index -1 = nearest TE
        x_l = self.coords['lower_rot']['x']   # negative x values (inlet half)
        x_u = self.coords['upper_rot']['x']

        mach_l = self.surface_mach['lower']    # mach[0] ≈ M_lower, mach[-1] ≈ M_inlet
        mach_u = self.surface_mach['upper']    # mach[0] ≈ M_upper, mach[-1] ≈ M_inlet

        # The axial chord is set by the lower surface (it extends furthest)
        x_te_inlet = x_l[-1]          # most-negative x  (inlet trailing edge)
        x_te_outlet = -x_l[-1]        # most-positive x  (outlet trailing edge)
        c_ax = x_te_outlet - x_te_inlet

        # --- Lower surface (continuous: inlet TE → throat → outlet TE) ---
        # Inlet half (flip so x goes from TE toward throat):
        #   x: x_l[-1] ... x_l[0]   mach: mach_l[-1] ... mach_l[0]
        # Throat midpoint at M_lower to bridge the gap:
        x_throat_l = 0.5 * (x_l[0] + (-x_l[0]))   # = 0
        # Outlet half (mirrored):
        #   x: -x_l[0] ... -x_l[-1]   mach: mach_l[0] ... mach_l[-1]
        x_lower = np.concatenate([
            np.flip(x_l),
            np.array([x_throat_l]),
            -x_l,
        ])
        mach_lower = np.concatenate([
            np.flip(mach_l),
            np.array([self.mach_lower]),
            mach_l,
        ])

        # --- Upper surface (continuous: inlet TE → throat → outlet TE) ---
        # The upper chars don't reach the TE, so add segments at M_inlet
        # to extend to the full axial chord.
        x_throat_u = 0.5 * (x_u[0] + (-x_u[0]))   # = 0
        x_upper = np.concatenate([
            np.array([x_te_inlet]),       # inlet TE  (M_inlet)
            np.flip(x_u),                 # inlet chars, TE→throat
            np.array([x_throat_u]),        # throat midpoint (M_upper)
            -x_u,                          # outlet chars, throat→TE
            np.array([x_te_outlet]),       # outlet TE  (M_inlet)
        ])
        mach_upper = np.concatenate([
            np.array([self.mach_inlet]),
            np.flip(mach_u),
            np.array([self.mach_upper]),
            mach_u,
            np.array([self.mach_inlet]),
        ])

        # --- Normalize to [0, 1] fraction of axial chord ---
        x_lower_norm = (x_lower - x_te_inlet) / c_ax
        x_upper_norm = (x_upper - x_te_inlet) / c_ax

        out = {
            'lower': {'x_norm': x_lower_norm, 'mach': mach_lower},
            'upper': {'x_norm': x_upper_norm, 'mach': mach_upper},
        }

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_lower_norm, mach_lower, 'b-o', markersize=3, label='Lower (pressure)')
            ax.plot(x_upper_norm, mach_upper, 'r-o', markersize=3, label='Upper (suction)')
            ax.set_xlabel('x / c_ax')
            ax.set_ylabel('Surface Mach Number')
            ax.set_title('Surface Mach Number Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return out

    # --- NASA TM X-2095 Utilities ---
    
    @staticmethod
    def get_nasa_limits(mach_inlet, gamma):
        """
        Helper that uses the NASA equations below to return (M_lower_min, M_upper_max) in Mach number
        rather than critical velocity ratio.
        """
        M_i_star = SupersonicTurbineMOC.convert_mach_to_critical(mach_inlet, gamma)
        Ml_star, Mu_star = SupersonicTurbineMOC.calculate_limiting_critical_velocity(M_i_star, gamma)
        
        Ml = SupersonicTurbineMOC.convert_critical_to_mach(Ml_star, gamma)
        Mu = SupersonicTurbineMOC.convert_critical_to_mach(Mu_star, gamma)
        return Ml, Mu

    @staticmethod
    def convert_mach_to_critical(M, gamma):
        numerator = ((gamma + 1) / 2) * (M**2)
        denominator = 1 + ((gamma - 1) / 2) * (M**2)
        return math.sqrt(numerator / denominator)

    @staticmethod
    def convert_critical_to_mach(M_star, gamma):
        numerator = 2 * (M_star**2)
        denominator = (gamma + 1) - (gamma - 1) * (M_star**2)
        if denominator <= 0:
            return float('inf')
        return math.sqrt(numerator / denominator)

    @staticmethod
    def calculate_limiting_critical_velocity(M_i_star, gamma):
        """
        Calculates the limiting lower and upper surface critical velocity ratios
        (M*_l_min and M*_u_max) based on NASA TM X-2095 equations.
        """
        g_plus = gamma + 1
        g_minus = gamma - 1
        
        val_A = 1 - (g_minus / g_plus) * (M_i_star**2)
        val_B_numerator = (gamma / g_plus) * (M_i_star**2)
        val_B = 1 + 0.5 * (val_B_numerator / val_A)
        exponent = g_minus / gamma

        # Lower bound
        term_l_combined = val_A * (val_B ** exponent)
        if 1 - term_l_combined < 0:
            raise ValueError("Inputs result in imaginary Lower Surface Mach Number")
        M_l_star_min = math.sqrt((g_plus / g_minus) * (1 - term_l_combined))

        # Upper bound
        term_u_combined = val_A * ((1.0 / val_B) ** exponent)
        if 1 - term_u_combined < 0:
            raise ValueError("Inputs result in imaginary Upper Surface Mach Number")
        M_u_star_max = math.sqrt((g_plus / g_minus) * (1 - term_u_combined))

        return M_l_star_min, M_u_star_max


if __name__ == "__main__":
    # Example usage:
    moc_solver = SupersonicTurbineMOC(
        gamma=1.4, 
        mach_inlet=1.7, 
        mach_lower=1.3, 
        mach_upper=2.2, 
        beta_inlet_deg=75.0, 
        dv=0.01
    )
    
    # 1. Generate Coordinates
    coords = moc_solver.generate()
    
    # 2. Print Summary
    moc_solver.print_summary()
    
    # 3. Use NASA limit checker
    Ml_min, Mu_max = SupersonicTurbineMOC.get_nasa_limits(mach_inlet=1.7, gamma=1.4)
    print(f"\nNASA Limits:")
    print(f"  Minimum Lower Surface Mach: {Ml_min:.4f}")
    print(f"  Maximum Upper Surface Mach: {Mu_max:.4f}")
    
    # 4. Plot (unrotated frame by default, pass unrotated=False for rotated)
    moc_solver.plot_blade(unrotated=False, full_passage=True)

    # 5. Surface Mach distribution
    mach_data = moc_solver.surface_mach_distribution(plot=True)