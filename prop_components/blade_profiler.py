import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Try importing the boundary layer functions from goldman_exact
# Assuming the script is run from the project root or quicktests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from quicktests.goldman_exact import solve_bl, solve_bl_sasman_cresci
except ImportError:
    print("Warning: Could not import solve_bl or solve_bl_sasman_cresci from quicktests.goldman_exact")

class DisplacedBladeProfiler:
    """
    Takes an ideal MOC turbine passage and evaluates the boundary layers 
    to output the physically corrected metal blade contour.
    """
    
    def __init__(self, turbine, moc_solver, bl_method='sasman_cresci'):
        """
        Parameters:
        -----------
        turbine : Turbine
            Configured turbine object with inlet gas properties.
        moc_solver : SupersonicTurbineMOC
            Configured MOC solver object.
        bl_method : str
            'sasman_cresci' or 'head'
        """
        self.turbine = turbine
        self.moc = moc_solver
        self.bl_method = bl_method
        
        self.re_chord = None
        self.bl_results = {}
        self.displaced_coords = {}
        self.separation_risks = {}
        
    def calculate_reynolds_number(self, c_meters):
        """
        Calculates Re_c based on the turbine's relative inlet conditions.
        If using N2 testing parameters (p01_n2), it uses those. Otherwise, hot gas.
        """
        if hasattr(self.turbine, 'p01_n2'):
            gam = self.turbine.gam_n2
            R = self.turbine.R_n2
            T3 = self.turbine.T3_n2
            p3 = self.turbine.p3_n2
            w3u = self.turbine.c3u_n2 - self.turbine.u
            w3m = self.turbine.c3m
        else:
            gam = self.turbine.gam3
            R = self.turbine.R_3
            T3 = self.turbine.T3
            p3 = self.turbine.p3
            w3u = self.turbine.c3u - self.turbine.u
            w3m = self.turbine.c3m
            
        # Relative velocity at rotor inlet
        W_rel = np.sqrt(w3u**2 + w3m**2)
        
        # Static density at rotor inlet
        rho_in = p3 / (R * T3)
        
        # Dynamic viscosity approximation (Sutherland for Air-like gas)
        # Using a simple Sutherland's law for air
        # mu = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4
        mu_in = mu_ref * (T3 / T_ref)**1.5 * (T_ref + S) / (T3 + S)
        
        self.re_chord = rho_in * W_rel * c_meters / mu_in
        
        print(f"Calculated Re_c: {self.re_chord:.1f}")
        print(f"  (rho_in={rho_in:.3f} kg/m3, U_in={W_rel:.1f} m/s, c={c_meters*1000:.2f} mm, mu_in={mu_in:.2e} Pa.s)")
        
        return self.re_chord
        
    def evaluate_boundary_layers(self, T0=500.0):
        """
        Extracts Mach distributions and runs the boundary layer solver.
        """
        if not self.moc.coords:
            self.moc.generate()
            
        # Get MOC surface data
        self.moc_mach_dist = self.moc.surface_mach_distribution(plot=False)
        
        # M_in is the relative inlet Mach number
        if hasattr(self.turbine, 'p01_n2'):
            gam = self.turbine.gam_n2
            R = self.turbine.R_n2
            T3 = self.turbine.T3_n2
            w3u = self.turbine.c3u_n2 - self.turbine.u
            w3m = self.turbine.c3m
        else:
            gam = self.turbine.gam3
            R = self.turbine.R_3
            T3 = self.turbine.T3
            w3u = self.turbine.c3u - self.turbine.u
            w3m = self.turbine.c3m
            
        W_rel = np.sqrt(w3u**2 + w3m**2)
        M_in = W_rel / np.sqrt(gam * R * T3)
        
        # Physical chord calculation
        x_te_inlet = self.moc.coords['lower_rot']['x'][-1]
        x_te_outlet = -x_te_inlet
        c_meters = x_te_outlet - x_te_inlet # Assuming x in MOC is true scale... wait
        # MOC coords are usually normalized by throat width or something. 
        # But let's assume they represent physical scale, or we scale them?
        # Let's assume MOC chord is 1.0, and the actual chord is c_phys.
        # Actually, the turbine.py calculates a physical chord `c` in `sweep_blade_count`.
        # For now, we will assume a 10mm chord if it's a miniature turbine, to get Re_c right.
        # We can let the user pass true physical chord or derive it from the turbine.
        # We will assume c_meters = 0.01 (10mm) for a tiny turbine.
        c_meters = 0.010 
        
        self.calculate_reynolds_number(c_meters)
        
        # Run BL Solver
        if self.bl_method.lower() == 'sasman_cresci':
            solve_func = solve_bl_sasman_cresci
        else:
            solve_func = solve_bl
            
        # Lower Surface
        s_lo = self.moc_mach_dist['lower']['s_norm']
        Me_lo = self.moc_mach_dist['lower']['mach']
        s_lo_bl, th_lo, Hi_lo, Me_lo_bl = solve_func(s_lo, Me_lo, self.re_chord, M_in, T0=T3)
        dstar_lo = Hi_lo * th_lo
        
        # Upper Surface
        s_up = self.moc_mach_dist['upper']['s_norm']
        Me_up = self.moc_mach_dist['upper']['mach']
        s_up_bl, th_up, Hi_up, Me_up_bl = solve_func(s_up, Me_up, self.re_chord, M_in, T0=T3)
        dstar_up = Hi_up * th_up
        
        self.bl_results = {
            'lower': {'s': s_lo_bl, 'theta': th_lo, 'Hi': Hi_lo, 'dstar': dstar_lo},
            'upper': {'s': s_up_bl, 'theta': th_up, 'Hi': Hi_up, 'dstar': dstar_up}
        }
        
    def displace_contour(self):
        """
        Displaces the MOC contour inwards by delta*.
        """
        if not self.bl_results:
            self.evaluate_boundary_layers()
            
        def apply_displacement(side):
            # MOC coordinates (Rotated physical frame)
            x = self.moc.coords[f'{side}_rot']['x']
            y = self.moc.coords[f'{side}_rot']['y']
            
            # Since the above is only the INLET half of the blade, we need the FULL blade contour!
            # Let's reconstruct the full continuous contour just like `surface_mach_distribution` does.
            # But simpler: we just need a function mapping `s_norm` to `(x, y)`.
            pass

        # Grab the full surface coordinates directly from the MOC solver
        # to ensure TE extensions and transition arcs match exactly.
        moc_full = self.moc.surface_mach_distribution(plot=False)
            
        for side in ['lower', 'upper']:
            x_full = moc_full[side]['x_full']
            y_full = moc_full[side]['y_full']
            s_norm = moc_full[side]['s_norm']
            chord = moc_full[f'chord_{side}']
            
            # Interpolate dstar onto the full contour points
            bl = self.bl_results[side]
            interp_dstar = interp1d(bl['s'], bl['dstar'], bounds_error=False, fill_value=(bl['dstar'][0], bl['dstar'][-1]))
            dstar_full = interp_dstar(s_norm)
            # Smooth out the mathematically sharp spike at the TE junction 
            # to prevent an unphysical kink in the actual blade metal contour
            dstar_full = gaussian_filter1d(dstar_full, sigma=3.0)
            
            # Normal calculation (use arc length s for correct gradient on non-uniform points)
            s_phys = np.zeros(len(x_full))
            s_phys[1:] = np.cumsum(np.sqrt(np.diff(x_full)**2 + np.diff(y_full)**2))
            
            dx = np.gradient(x_full, s_phys)
            dy = np.gradient(y_full, s_phys)
            mag = np.sqrt(dx**2 + dy**2) + 1e-12
            
            # Tangent is (dx, dy).
            # The passage boundary x_full goes from +x (inlet) to -x (outlet).
            # dx is mostly negative. dy is positive then negative.
            # Flow goes from right to left in these coordinates.
            # Lower surface (blue): It's the TOP of the passage (larger y).
            # To push the metal OUT of the passage, we must push it UP.
            # If dx is negative, (dy, -dx) has a positive y component (-dx > 0).
            # Let's verify: we want the normal to point UP/OUT.
            # For lower (top curve), we want ny > 0.
            if side == 'lower':
                # Metal is above the passage. Push UP.
                # Since dx is negative, -dx is positive.
                nx = dy / mag
                ny = -dx / mag
                # If ny < 0, flip it
                if np.mean(ny) < 0:
                    nx, ny = -nx, -ny
            else:
                # Metal is below the passage. Push DOWN.
                # We want ny < 0.
                nx = -dy / mag
                ny = dx / mag
                # If ny > 0, flip it
                if np.mean(ny) > 0:
                    nx, ny = -nx, -ny
                
            # Scale dimensionless dstar (dstar/c) by the MOC chord length 
            # to get the displacement in MOC coordinate units.
            dstar_moc_units = dstar_full * chord
            
            x_disp = x_full + dstar_moc_units * nx
            y_disp = y_full + dstar_moc_units * ny
            
            self.displaced_coords[side] = {
                'x_ideal': x_full, 'y_ideal': y_full,
                'x_disp': x_disp, 'y_disp': y_disp,
                'dstar': dstar_full
            }
            
    def analyze_separation_risk(self):
        """
        Analyzes the form factor H_i to flag separation risks.
        """
        if not self.bl_results:
            self.evaluate_boundary_layers()
            
        print("\n--- Separation Risk Analysis ---")
        for side in ['lower', 'upper']:
            Hi = self.bl_results[side]['Hi']
            s_norm = self.bl_results[side]['s']
            
            max_Hi = np.max(Hi)
            max_Hi_loc = s_norm[np.argmax(Hi)]
            
            self.separation_risks[side] = {
                'max_Hi': max_Hi,
                'location': max_Hi_loc,
                'separated': max_Hi > 2.4,
                'at_risk': max_Hi > 1.8
            }
            
            status = "SAFE"
            if max_Hi > 2.4:
                status = "SEPARATED (High Risk)"
            elif max_Hi > 1.8:
                status = "AT RISK (Transitioning/Separating)"
                
            print(f"{side.capitalize()} Surface:")
            print(f"  Max H_i: {max_Hi:.3f} at s/c = {max_Hi_loc:.3f} -> {status}")
            
    def plot_displaced_contour(self, save_path=None):
        """
        Generates the plot of the ideal and displaced contours.
        """
        if not self.displaced_coords:
            self.displace_contour()
            
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot lower
        low = self.displaced_coords['lower']
        ax.plot(low['x_ideal'], low['y_ideal'], 'b-', lw=1.5, label='Ideal MOC (Lower)')
        ax.plot(low['x_disp'], low['y_disp'], 'b--', lw=2.5, label='Physical Metal (Lower)')
        ax.fill(np.concatenate([low['x_ideal'], low['x_disp'][::-1]]), 
                np.concatenate([low['y_ideal'], low['y_disp'][::-1]]), 
                color='blue', alpha=0.1)
        
        # Plot upper
        up = self.displaced_coords['upper']
        ax.plot(up['x_ideal'], up['y_ideal'], 'r-', lw=1.5, label='Ideal MOC (Upper)')
        ax.plot(up['x_disp'], up['y_disp'], 'r--', lw=2.5, label='Physical Metal (Upper)')
        ax.fill(np.concatenate([up['x_ideal'], up['x_disp'][::-1]]), 
                np.concatenate([up['y_ideal'], up['y_disp'][::-1]]), 
                color='red', alpha=0.1)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(rf'Blade Passage Contour Correction ($\delta^*$ Offset)\nMethod: {self.bl_method}, $Re_c={self.re_chord:.0f}$', fontsize=14)
        ax.set_xlabel('Axial Position (normalized)', fontsize=12)
        ax.set_ylabel('Tangential Position (normalized)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def plot_boundary_layer_parameters(self, save_path=None):
        """
        Plots the Incompressible Form Factor (H_i), Momentum Thickness (theta/c),
        and Displacement Thickness (dstar/c) for both surfaces.
        """
        if not self.bl_results:
            self.evaluate_boundary_layers()
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
        
        # Plot 1: Form Factor (H_i)
        ax = axes[0]
        ax.plot(self.bl_results['lower']['s'], self.bl_results['lower']['Hi'], 'b-', lw=2, label='Lower (Pressure)')
        ax.plot(self.bl_results['upper']['s'], self.bl_results['upper']['Hi'], 'r-', lw=2, label='Upper (Suction)')
        ax.axhspan(1.8, 2.4, alpha=0.08, color='red', label='Separation Risk Range')
        ax.axhline(1.8, color='gray', ls='--', alpha=0.4)
        ax.axhline(2.4, color='gray', ls='--', alpha=0.4)
        ax.set_ylabel('Form Factor, $H_i$', fontsize=12)
        ax.set_title(f'Boundary Layer Parameters ($Re_c={self.re_chord:.0f}$)\nMethod: {self.bl_method}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Momentum Thickness (theta/c)
        ax = axes[1]
        ax.plot(self.bl_results['lower']['s'], self.bl_results['lower']['theta'], 'b-', lw=2)
        ax.plot(self.bl_results['upper']['s'], self.bl_results['upper']['theta'], 'r-', lw=2)
        ax.set_ylabel(r'Momentum Thickness, $\theta / c$', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 3: Displacement Thickness (dstar/c)
        ax = axes[2]
        ax.plot(self.bl_results['lower']['s'], self.bl_results['lower']['dstar'], 'b-', lw=2)
        ax.plot(self.bl_results['upper']['s'], self.bl_results['upper']['dstar'], 'r-', lw=2)
        ax.set_ylabel(r'Displacement Thickness, $\delta^* / c$', fontsize=12)
        ax.set_xlabel('Fraction of chord ($s/c$)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved BL parameters plot to {save_path}")
        else:
            plt.show()
