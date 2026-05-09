        
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

class BarskePump:
    def __init__(self,
                 mdot_desired,  # kg/s
                 p_desired,    # Pa
                 rho,           # kg/m3
                 visc,          # Pa.s
                 RPM,
                 top_bearing,
                 bot_bearing,
                 mechanical_seal,
                 d_1=None,  # m
                 d_2=None,  # m
                 ):
        self.mdot_desired = mdot_desired
        self.p_desired = p_desired
        self.rho = rho
        self.visc = visc
        self.RPM = RPM
        self.top_bearing = top_bearing
        self.bot_bearing = bot_bearing
        self.mechanical_seal = mechanical_seal

        # Assumptions and calculations
        self.Q_desired = mdot_desired / rho
        self.coeff_p = 0.2
        self.g = 9.81
        self.H_desired = p_desired / (self.rho * self.g)  # m
        self.v_0 = 1 # Inlet velocity [m/s]

        # 1. Geometry Logic (Unified)
        self.d_0 = np.sqrt(4 * self.Q_desired / (np.pi * self.v_0))
        
        # Set d_1
        if d_1 is not None:
            self.d_1 = d_1
        else:
            self.d_1 = 1.1 * self.d_0
        self.b_1 = 1.2 * 0.25 * self.d_1

        # Set d_2 (Critical Switch)
        if d_2 is not None:
            self.d_2 = d_2
            # Calculate what pressure this FIXED hardware produces at this RPM
            u_2 = self.RPM * np.pi * self.d_2 / 60
            u_1 = self.RPM * np.pi * self.d_1 / 60
            # update p_actual based on fixed geometry
            self.p_actual = 0.5 * self.rho * ((1 + self.coeff_p) * u_2**2 - u_1**2)
            self.H_actual = self.p_actual / (self.rho * self.g)
        else:
            # Sizing mode: Calculate d_2 to MEET p_desired
            self.p_actual = p_desired
            self.H_actual = p_desired / (self.rho * self.g)
            self.d_2 = np.sqrt((1 / (1 + self.coeff_p)) * (7200 * p_desired / (self.rho * self.RPM**2 * np.pi**2) + self.d_1**2))

        self.u_2 = self.RPM * np.pi * self.d_2 / 60
        self.u_1 = self.RPM * np.pi * self.d_1 / 60

        self.p_th = 0.5 * self.rho * (2 * self.u_2**2 - self.u_1**2)
        self.power_friction = 1956 * self.rho * self.visc**0.2 * (self.RPM/1000)**2.8 * (self.d_2**4.6 + 4.6 * self.d_1**3.6 * self.b_1)
        self.mechanical_loss = self.top_bearing.power_loss(self.RPM, 0.1, 0) + self.bot_bearing.power_loss(self.RPM, 0.1, 0) + self.mechanical_seal.power_loss(self.RPM, self.p_actual)
        self.power_loss = self.power_friction + self.mechanical_loss

        self.efficiency = self.p_actual / (self.p_th + (self.power_loss) / self.Q_desired)
        self.required_power = self.mdot_desired * self.p_actual / (self.rho * self.efficiency)

        self.hydraulic_work = self.required_power - self.mechanical_loss
        self.useful_hydraulic_work = self.hydraulic_work - self.power_friction
        self.static_enthalpy = self.mdot_desired * 0.5 * (self.u_2**2 - self.u_1**2) # WHAT IS THIS?
        self.theoretical_dynamic_enthalpy = self.mdot_desired * 0.5 * (self.u_2**2) # WHAT IS THIS?
        self.dynamic_enthalpy = self.mdot_desired * 0.5 * self.coeff_p * (self.u_2**2)
        self.useful_work = self.mdot_desired * 0.5 * ((1 + self.coeff_p) * self.u_2**2 - self.u_1**2)

        self.n_q = self.RPM * self.Q_desired**0.5 * self.H_actual**-0.75
        self.s_ax = min(0.00075, self.d_2 / 100)
        self.b_2 = max(1.5 * self.b_1 * self.d_1 / self.d_2, self.s_ax)
        self.B = 2 * self.s_ax + self.b_2
        self.H = max(self.B / 2, 0.0025)
        self.n_s = 150 # Lower estimate from lobanoff
        self.NPSHR = self.H_actual * (self.n_q / self.n_s)**(4/3)

        self.pressure_efficiency = self.p_actual / self.p_th
        self.head_coeff = 2 * self.H_actual * self.g / self.u_2**2
        self.v_3 = 0.8 * self.u_2  # Assuming a diffuser velocity coefficient of 0.8
        self.d_3 = np.sqrt(self.Q_desired / (self.v_3 * np.pi)) * 2
        self.d_4 = self.d_3 * 2
        self.v_4 = self.v_3 / 4
        self.blade_number = 6
        self.w = self.RPM * 2 * np.pi / 60
        self.torque = self.required_power / self.w
        self.tip_force = self.torque / (self.d_2 / 2) / self.blade_number
        self.blade_thickness = 0.0042
        self.blade_density = 1160
        self.blade_mass = self.b_2 * self.d_2 / 2 * self.blade_thickness * self.blade_density
        self.centrifugal_force = self.w**2 * self.d_2 / 2 * self.blade_mass
        self.blade_stress = self.centrifugal_force / (self.b_2 * self.blade_thickness)

    def pretty_print(self):
        print(f"Barske Pump:---------------------------")
        print(f"Main results:")
        print(f"  {'RPM':<30} {self.RPM:.2f}")
        print(f"  {'Required Power':<30} {self.required_power:.2f} W")
        print(f"  {'Diameter at Outlet':<30} {self.d_2:.5f} m")
        print(f"  {"Mass flow rate (mdot)":<30} {self.mdot_desired:<10.4g} kg/s")
        print(f"  {"Outlet Head (H)":<30} {self.H_actual:<10.4g} m")
        print(f"  {"Rotational speed (n)":<30} {self.RPM:<10.4g} rpm")
        print(f"  {"Flow rate (Q)":<30} {self.Q_desired:<10.4g} m^3/s")
        print(f"  {"Specific speed (n_q)":<30} {self.n_q:<10.4g}")
        print(f"Detailed Results:")
        print(f"  {"Efficiency":<30} {self.efficiency:<10.4g}")
        print(f"  {"Inlet Diameter (d1)":<30} {self.d_0*1000:<10.4g} mm")
        print(f"  {"Impeller outlet diameter (d2)":<30} {self.d_2*1000:<10.4g} mm")
        print(f"  {"Impeller inlet diameter (d1)":<30} {self.d_1*1000:<10.4g} mm")
        print(f"  {"Power required (P)":<30} {self.required_power:<10.4g} W")
        print(f"  {"Blade height (b_1)":<30} {self.b_1*1000:<10.4g} mm")
        print(f"  {"Blade height (b_2)":<30} {self.b_2*1000:<10.4g} mm")
        print(f"  {"Axial Clearance (s_ax)":<30} {self.s_ax*1000:<10.4g} mm")
        print(f"  {"Radial Clearance (H)":<30} {self.H*1000:<10.4g} mm")
        print(f"  {"u_1":<30} {self.u_1:<10.4g} m/s")
        print(f"  {"u_2":<30} {self.u_2:<10.4g} m/s")
        print(f"  {"v_3":<30} {self.v_3:<10.4g} m/s")
        print(f"  {"v_4":<30} {self.v_4:<10.4g} m/s")
        print(f"  {"Mechanical Loss":<30} {self.mechanical_loss:<10.4g} W")
        print(f"  {"Friction Loss":<30} {self.power_friction:<10.4g} W")
        print(f"  {"Total Hydraulic Work":<30} {self.hydraulic_work:<10.4g} W")
        print(f"  {"Useful Hydraulic Work":<30} {self.useful_hydraulic_work:<10.4g} W")
        print(f"  {"Static Enthalpy":<30} {self.static_enthalpy:<10.4g} J/kg")
        print(f"  {"Theoretical Dynamic Enthalpy":<30} {self.theoretical_dynamic_enthalpy:<10.4g} J/kg")
        print(f"  {"Dynamic Enthalpy":<30} {self.dynamic_enthalpy:<10.4g} J/kg")
        print(f"  {"Useful Work":<30} {self.useful_work:<10.4g} J/kg")
        print(f"  {"Head Coefficient":<30} {self.head_coeff:<10.4g}")
        print(f"  {"Diffuser Diameter":<30} {self.d_3*1000:<10.4g} mm")
        print(f"  {"Diffuser Outlet Diameter":<30} {self.d_4*1000:<10.4g} mm")
        print(f"  {"Torque":<30} {self.torque:<10.4g} Nm")
        print(f"  {"Tip force":<30} {self.tip_force:<10.4g} N")
        print(f"  {"Centrifugal Force":<30} {self.centrifugal_force:<10.4g} N")
        print(f"  {"Blade Stress":<30} {self.blade_stress/1e6:<10.4g} MPa")
    
    def visualize(self):


        # Create figure with two subplots - side view and top view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle('Barske Impeller Design', fontsize=16)

        # Side view (axial section)
        ax1.set_title('Meridonal View')
        ax1.set_xlabel('Axial Distance [mm]')
        ax1.set_ylabel('Radial Distance [mm]')
        ax1.set_aspect('equal')

        # Convert to mm for plotting
        rib_height = 0.003

        d0_mm = self.d_0 * 1000
        d1_mm = self.d_1 * 1000
        d2_mm = self.d_2 * 1000
        b1_mm = self.b_1 * 1000
        b2_mm = self.b_2 * 1000
        s_ax_mm = self.s_ax * 1000
        B_mm = self.B * 1000
        H_mm = self.H * 1000
        rib_height_mm = rib_height * 1000


        # Draw an example square
        ax1.plot([-10, 10], [0, 0], color='black', linestyle='--', label='Shaft Axis')

        ax1.plot([-10, -s_ax_mm], [-d0_mm/2, -d0_mm/2], color='green', linestyle='-')
        ax1.plot([-10, -s_ax_mm], [d0_mm/2, d0_mm/2], color='green', linestyle='-')
        # Draw top blade
        ax1.add_patch(Polygon([[0, d1_mm/2], [rib_height_mm + b1_mm - b2_mm, d2_mm/2], [rib_height_mm + b1_mm, d2_mm/2], [rib_height_mm + b1_mm, d1_mm/2]], closed=True, fill=False, edgecolor='blue', linestyle='-', label='Blade (b₁)'))
        # Draw bottom blade
        ax1.add_patch(Polygon([[0, -d1_mm/2], [rib_height_mm + b1_mm - b2_mm, -d2_mm/2], [rib_height_mm + b1_mm, -d2_mm/2], [rib_height_mm + b1_mm, -d1_mm/2]], closed=True, fill=False, edgecolor='blue', linestyle='-', label='Blade (b₂)'))
        # Draw rib
        ax1.add_patch(Rectangle((b1_mm, -d1_mm/2), rib_height_mm, d1_mm, fill=False, edgecolor='purple', linestyle='-', label='Rib'))
        # Draw top casing
        ax1.add_patch(Polygon([[-s_ax_mm, d0_mm/2],[-s_ax_mm, d1_mm/2], [-s_ax_mm + rib_height_mm + b1_mm - b2_mm, d2_mm/2], [-s_ax_mm + rib_height_mm + b1_mm - b2_mm, d2_mm/2 + H_mm], [rib_height_mm + b1_mm + s_ax_mm, d2_mm/2 + H_mm], [rib_height_mm + s_ax_mm + b1_mm, d1_mm/2]], closed=False, fill=False, edgecolor='blue', linestyle='-', label='Blade (b₁)'))
        # Draw bottom casing
        ax1.add_patch(Polygon([[-s_ax_mm, -d0_mm/2],[-s_ax_mm, -d1_mm/2], [-s_ax_mm + rib_height_mm + b1_mm - b2_mm, -d2_mm/2], [-s_ax_mm + rib_height_mm + b1_mm - b2_mm, -d2_mm/2 - H_mm], [rib_height_mm + b1_mm + s_ax_mm, -d2_mm/2 - H_mm], [rib_height_mm + s_ax_mm + b1_mm, -d1_mm/2]], closed=False, fill=False, edgecolor='blue', linestyle='-', label='Blade (b₂)'))


        # Top view (radial section)
        ax2.set_title('Top View (Radial Section)')
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Y [mm]')
        ax2.set_aspect('equal')

        # Draw inlet, impeller and casing in top view
        ax2.add_patch(Circle((0, 0), d0_mm/2, fill=False, color='green', linestyle='-', label='Inlet (d₀)'))
        ax2.add_patch(Circle((0, 0), d1_mm/2, fill=False, color='blue', linestyle='-', label='Inner (d₁)'))
        ax2.add_patch(Circle((0, 0), d2_mm/2, fill=False, color='red', linestyle='-', label='Outer (d₂)'))
        ax2.add_patch(Circle((0, 0), d2_mm/2 + B_mm, fill=False, color='black', linestyle='--', label='Casing'))

        # Draw simplified blades (6 blades as typical for Barske design)
        blade_angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 equally spaced angles
        for angle in blade_angles:
            x1 = d1_mm/2 * np.cos(angle)
            y1 = d1_mm/2 * np.sin(angle)
            x2 = d2_mm/2 * np.cos(angle)
            y2 = d2_mm/2 * np.sin(angle)
            ax2.plot([x1, x2], [y1, y2], 'b-', linewidth=2)

        # Set axis limits with some padding
        max_dim = (d2_mm/2 + B_mm) * 1.2
        ax2.set_xlim(-max_dim, max_dim)
        ax2.set_ylim(-max_dim, max_dim)
        ax1.set_ylim(-max_dim, max_dim)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def find_optimal_rpm(self, rpms=None, mdots=None, plot=True):
        """Find optimal RPM(s) for one or more mass flows and optionally plot results.

        For each mdot in `mdots` this scans `rpms`, finds the RPM with maximum
        efficiency and (if `plot`) creates a vertical stack of subplots. Each row shows:
          - efficiency vs RPM (left axis)
          - hydraulic/friction and mechanical losses vs RPM (right axis)
          - outlet diameter `d_2` vs RPM on an outward right axis

        Args:
            rpms: iterable of RPMs to scan. If None uses `np.linspace(1000, 30000, 291)`.
            mdots: scalar or iterable of mass flows (kg/s). If None uses `self.mdot_desired`.
            plot: whether to show the figure.

        Returns:
            List of tuples [(mdot, opt_rpm, opt_efficiency), ...]
        """
        if rpms is None:
            rpms = np.linspace(1000, 30000, 291)

        if mdots is None:
            mdots = [self.mdot_desired]

        # ensure iterable
        if not hasattr(mdots, '__iter__') or isinstance(mdots, (str, bytes)):
            mdots = [mdots]

        results = []

        # helper to compute series for a given mdot
        def compute_series(mdot):
            eff = []
            mech = []
            friction = []
            total = []
            d2 = []
            req_power = []
            hydraulic = []
            n_q = []
            npshr = []
            for n in rpms:
                p = BarskePump(mdot, self.p_desired, self.rho, self.visc, n,
                               self.top_bearing, self.bot_bearing, self.mechanical_seal)
                eff.append(p.efficiency)
                mech.append(p.mechanical_loss)
                friction.append(p.power_friction)
                total.append(p.power_loss)
                d2.append(p.d_2)
                req_power.append(p.required_power)
                hydraulic.append(p.hydraulic_work)
                n_q.append(p.n_q)
                npshr.append(p.NPSHR)
            return (np.array(eff), np.array(mech), np.array(friction), np.array(total),
                    np.array(d2), np.array(req_power), np.array(hydraulic), np.array(n_q), np.array(npshr))

        if plot:
            n = len(mdots)
            fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), sharex=True)
            if n == 1:
                axes = np.array([axes])

        for i, mdot in enumerate(mdots):
            eff, mech, friction, total, d2, req_power, hydraulic, n_q, npshr = compute_series(mdot)

            # find optimum efficiency
            idx = int(np.nanargmax(eff))
            opt_rpm = float(rpms[idx])
            opt_eff = float(eff[idx])
            results.append((mdot, opt_rpm, opt_eff))

            if plot:
                ax_left = axes[i, 0]
                ax_right = axes[i, 1]

                # Left: efficiency (left) and losses (right twin)
                ax_left.plot(rpms, eff, color='g', lw=2, label='Efficiency')
                ax_left.set_ylabel('Efficiency', color='g')
                ax_left.tick_params(axis='y', labelcolor='g')
                ax_left.grid(True, alpha=0.3)

                ax_losses = ax_left.twinx()
                ax_losses.plot(rpms, friction, color='tab:blue', linestyle='--', label='Hydraulic friction')
                ax_losses.plot(rpms, mech, color='tab:red', linestyle='--', label='Mechanical loss')
                ax_losses.set_ylabel('Power (W)')

                # annotate optimum on left
                ax_left.axvline(opt_rpm, color='r', linestyle='--')
                ax_left.plot(opt_rpm, opt_eff, 'ro')
                ax_left.annotate(f'{opt_rpm:.0f} rpm\n{opt_eff:.3f}', xy=(opt_rpm, opt_eff), xytext=(8, -28), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

                # Left legend: combine efficiency + losses
                l1, lab1 = ax_left.get_legend_handles_labels()
                l2, lab2 = ax_losses.get_legend_handles_labels()
                ax_left.legend(l1 + l2, lab1 + lab2, loc='upper left')

                # Right: d2 (left axis), specific speed and NPSHR on twins
                ax_r = ax_right
                ax_r.plot(rpms, d2 * 1000.0, color='m', linestyle='-', label='Outlet d2 (mm)')
                ax_r.set_ylabel('d2 (mm)', color='m')
                ax_r.tick_params(axis='y', labelcolor='m')

                ax_nq = ax_r.twinx()
                ax_nq.plot(rpms, n_q, color='tab:blue', linestyle='--', label='Specific speed (n_q)')
                ax_nq.set_ylabel('Specific speed (n_q)', color='tab:blue')
                ax_nq.tick_params(axis='y', labelcolor='tab:blue')

                ax_npshr = ax_r.twinx()
                ax_npshr.spines.right.set_position(('outward', 60))
                ax_npshr.plot(rpms, npshr, color='tab:red', linestyle=':', label='NPSHR (m)')
                ax_npshr.set_ylabel('NPSHR (m)', color='tab:red')
                ax_npshr.tick_params(axis='y', labelcolor='tab:red')

                # Right legend
                lines = []
                labels = []
                for A in (ax_r, ax_nq, ax_npshr):
                    l, lab = A.get_legend_handles_labels()
                    lines += l
                    labels += lab
                ax_r.legend(lines, labels, loc='upper left')

                ax_left.set_title(f'Mdot={mdot:.4g} kg/s — optimal {opt_rpm:.0f} rpm')

        if plot:
            axes[-1, 0].set_xlabel('Rotational Speed (RPM)')
            axes[-1, 1].set_xlabel('Rotational Speed (RPM)')
            plt.tight_layout()
            plt.show()

        return results
    def plot_pump_map(self, rpms):
        fig, ax = plt.subplots(figsize=(10, 6), sharex=True)


        if rpms is None:
            rpms = np.linspace(5000, 25000, 5)

        for rpm in rpms:
            u_2 = rpm * np.pi * self.d_2 / 60
            u_1 = rpm * np.pi * self.d_1 / 60

            p = 0.5 * self.rho * ((1 + self.coeff_p) * u_2**2 - u_1**2)
            # Pump max flow rate
            throat_diameter = 0.003
            tip_velocity = u_2
            A_throat = throat_diameter * throat_diameter * np.pi / 4
            Qmax = A_throat * np.sqrt(2) * tip_velocity
            mdotmax = Qmax * 1000

            ax.plot([0, mdotmax, mdotmax], np.array([p, p, 0])/1e5, label=f'RPM: {rpm:.0f}')

        ax.set_xlabel('Mass Flow Rate (kg/s)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_title('Pump Map')
        ax.grid(True)
        ax.legend()
        plt.show()
    def plot_partload(self, rpms=None, mdots=None):
        """Compact part-load plotting.

        Left: losses (disk friction, mechanical, total) and efficiency (twin y-axis).
        Right: suction performance (NPSHR) and specific speed (n_q) overlaid for each `mdot`.

        Args:
            rpms: iterable of RPM values. If None a default range is used.
            mdots: iterable of mass flowrates (kg/s). Each mass flow produces an overlaid curve.
                   If None uses the current object's `mdot_desired` as a single curve.
        """
        if rpms is None:
            rpms = np.linspace(5000, 25000, 9)

        if mdots is None:
            mdots = [self.mdot_desired]

        # Ensure list
        mdots = list(mdots)
        nrows = len(mdots)

        # small helper to compute series for a given mdot
        def compute_series(mdot):
            friction = []
            mechanical = []
            total = []
            efficiency = []
            npshr = []
            n_q = []
            for n in rpms:
                p = BarskePump(mdot, self.p_desired, self.rho, self.visc, n,
                               self.top_bearing, self.bot_bearing, self.mechanical_seal, self.d_1, self.d_2)
                friction.append(p.power_friction)
                mechanical.append(p.mechanical_loss)
                total.append(p.power_loss)
                efficiency.append(p.efficiency)
                npshr.append(p.NPSHR)
                n_q.append(p.n_q)
            return np.array(friction), np.array(mechanical), np.array(total), np.array(efficiency), np.array(npshr), np.array(n_q)
        # compact-only mode: create a single 1x2 figure so there are no empty subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex='col')
        ax_loss = axes[0]
        ax_right = axes[1]

        # compute mdot-independent losses at a representative mdot
        rep_mdot = mdots[0]
        friction, mechanical, total, _, _, _ = compute_series(rep_mdot)

        ax_loss.plot(rpms, friction, 'b--', label='Disk Friction Loss')
        ax_loss.plot(rpms, mechanical, 'r--', label='Mechanical Loss (Seals/Bearings)')
        ax_loss.plot(rpms, total, 'k-', linewidth=2, label='Total Power Loss')
        ax_loss.set_ylabel('Power Loss (W)')
        ax_loss.set_title('Losses (independent of mdot)')
        ax_loss.grid(True, alpha=0.3)

        ax_eff = ax_loss.twinx()
        ax_eff.set_ylabel('Efficiency', color='g')
        ax_eff.tick_params(axis='y', labelcolor='g')

        # prepare color map for mdots
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / max(1, (nrows - 1))) for i in range(nrows)]

        ax_right.set_xlabel('Rotational Speed (RPM)')
        ax_right.set_title('Suction (NPSHR) and Specific Speed (n_q) for mdots')
        ax_right.grid(True, alpha=0.3)

        ax_nq = ax_right.twinx()

        # plot mdot-varied curves on the right axes and efficiency on the left twin
        for idx, mdot in enumerate(mdots):
            friction, mechanical, total, efficiency, npshr, n_q = compute_series(mdot)
            color = colors[idx]
            ax_right.plot(rpms, npshr, color=color, linestyle='-', label=f'NPSHR @ mdot={mdot:.4g} kg/s')
            ax_nq.plot(rpms, n_q, color=color, linestyle='--', label=f'n_q @ mdot={mdot:.4g} kg/s')
            ax_eff.plot(rpms, efficiency, color=color, linestyle=':', label=f'Eff @ mdot={mdot:.4g} kg/s')

        # left legend: losses + efficiency
        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_eff, labels_eff = ax_eff.get_legend_handles_labels()
        ax_loss.legend(lines_loss + lines_eff, labels_loss + labels_eff, loc='upper left')

        # right legend: NPSHR + n_q
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        lines_nq, labels_nq = ax_nq.get_legend_handles_labels()
        ax_right.legend(lines_right + lines_nq, labels_right + labels_nq, loc='upper left')

        plt.tight_layout()
        plt.show()
    