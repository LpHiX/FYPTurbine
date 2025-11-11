import numpy as np
from dataclasses import dataclass

BEARING_CONSTANTS = {
  2:   {"R1": 4.4e-7, "R2": 1.7, "S1": 2.00e-3, "S2": 100},
  3:   {"R1": 4.4e-7, "R2": 1.7, "S1": 2.00e-3, "S2": 100},
  42:  {"R1": 5.4e-7, "R2": 0.96, "S1": 3.00e-3, "S2": 40},
  43:  {"R1": 5.4e-7, "R2": 0.96, "S1": 3.00e-3, "S2": 40},
  60:  {"R1": 4.1e-7, "R2": 1.7, "S1": 3.73e-3, "S2": 14.6},
  630: {"R1": 4.1e-7, "R2": 1.7, "S1": 3.73e-3, "S2": 14.6},
  62:  {"R1": 3.9e-7, "R2": 1.7, "S1": 3.23e-3, "S2": 36.5},
  622: {"R1": 3.9e-7, "R2": 1.7, "S1": 3.23e-3, "S2": 36.5},
  63:  {"R1": 3.7e-7, "R2": 1.7, "S1": 2.84e-3, "S2": 92.8},
  623: {"R1": 3.7e-7, "R2": 1.7, "S1": 2.84e-3, "S2": 92.8},
  64:  {"R1": 3.6e-7, "R2": 1.7, "S1": 2.43e-3, "S2": 198},
  160: {"R1": 4.3e-7, "R2": 1.7, "S1": 4.63e-3, "S2": 4.25},
  161: {"R1": 4.3e-7, "R2": 1.7, "S1": 4.63e-3, "S2": 4.25},
  617: {"R1": 4.7e-7, "R2": 1.7, "S1": 6.50e-3, "S2": 0.78},
  618: {"R1": 4.7e-7, "R2": 1.7, "S1": 6.50e-3, "S2": 0.78},
  628: {"R1": 4.7e-7, "R2": 1.7, "S1": 6.50e-3, "S2": 0.78},
  637: {"R1": 4.7e-7, "R2": 1.7, "S1": 6.50e-3, "S2": 0.78},
  638: {"R1": 4.7e-7, "R2": 1.7, "S1": 6.50e-3, "S2": 0.78},
  619: {"R1": 4.3e-7, "R2": 1.7, "S1": 4.75e-3, "S2": 3.6},
  639: {"R1": 4.3e-7, "R2": 1.7, "S1": 4.75e-3, "S2": 3.6}
}

@dataclass
class Bearing:
    """
    Simple bearing model for deep groove ball bearings ONLY.
    
    Parameters
    ----------
    d : float
        Inner diameter of the bearing (millimeters).
    D : float
        Outer diameter of the bearing (millimeters).
    series : int
        Bearing series designation.
    visc : float
        Viscosity of the lubricant (mm²/s).
    C_0_kN : float
        Basic static load rating (kN).
    submerged : bool
        Whether the bearing is submerged in lubricant.
    debug : bool, optional
        Enable debug mode (default: False).

    Notes
    -----
    If the bearing is submerged, it is assumed to be fully submerged with the fluid. Otherwise, grease lubrication is assumed.
    """
    d: float
    D: float
    series: int
    visc: float
    C_0_kN: float
    submerged: bool = False
    debug: bool = False

    def __post_init__(self):
        self.initialize_constants()


    def initialize_constants(self):
        self.dm = 0.5 * (self.D + self.d)

        try:
            self.constants = BEARING_CONSTANTS[self.series]
        except KeyError:
            raise ValueError(f"Unknown bearing series: {self.series}")
        
        # Replenishment/starvation constant
        if self.submerged:
            self.K_rs = 3e-8 # low level oil bath and oil jet lube
        else:
            self.K_rs = 6e-8 # grease and oil-air lubrication
        self.K_z = 3.1 # Bearing geometric constant for deep groove ball bearings

        self.R_1 = self.constants["R1"]
        self.R_2 = self.constants["R2"]
        self.S_1 = self.constants["S1"]
        self.S_2 = self.constants["S2"]

        self._i_rw = 1 # Number of ball rows
        self._t = 2 * np.arccos((0.6 * self.dm - 1.2 * self.dm) / (0.6 * self.dm)) # Assuming submerged
        self._f_a = 0.05 * self.K_z * (self.D + self.d) / (self.D - self.d)
        self._R_s = 0.36 * self.dm ** 2 * (self._t - np.sin(self._t)) * self._f_a

        if self._t <= np.pi:
            self._f_t = np.sin(0.5 * self._t)
        else:
            self._f_t = 1

        self._K_ball = self._i_rw * self.K_z * (self.d + self.D) / (self.D - self.d) * 1e-12



        
    def geometric_constants(self, F_r_kN: float, F_a_kN: float) -> tuple:
        F_r_kN = np.abs(F_r_kN)
        F_a_kN = np.abs(F_a_kN)

        alpha_F = np.deg2rad(24.6 * (F_a_kN / self.C_0_kN)**0.24)

        if F_a_kN > 0:    
            G_rr = self.R_1 * self.dm**1.96 * (F_r_kN + self.R_2 / np.sin(alpha_F) * F_a_kN)**0.54
            G_sl = self.S_1 * self.dm**-0.145 * (F_r_kN**5 + self.S_2 * self.dm**1.5 / np.sin(alpha_F) * F_a_kN**4)**(1/3)
        else:
            G_rr = self.R_1 * self.dm**1.96 * F_r_kN**0.54
            G_sl = self.S_1 * self.dm**-0.26 * F_r_kN**(5/3)
    
        return G_rr, G_sl

    def total_moment(self, n: float, F_r_kN: float, F_a_kN: float) -> float:
        """
        Calculate the bearing's moment based on the rotational speed.

        Parameters
        ----------
        n : float
            Rotational speed in revolutions per minute (RPM).

        Returns
        -------
        float
            The calculated moment in newton-meters (Nm).
        """
        G_rr, G_sl = self.geometric_constants(F_r_kN, F_a_kN)
        # -------------------------
        # Rolling Frictional Moment 

        # Inner shear heating reduction factor
        phi_ish = 1 / (1 + 1.84e-9 * (n * self.dm)**1.28 * self.visc**0.64) 

        # Kinematic replenishment/ starvation reduction factor
        if self.submerged:
            phi_rs = 1
        else:
            phi_rs = np.exp(-self.K_rs * self.visc * n * (self.d + self.D) * np.sqrt(self.K_z / (2 * (self.D - self.d))))

        M_rr = phi_ish * phi_rs * G_rr * (self.visc * n)**0.6 / 1000 # Convert to Nm
        
        # -------------------------
        # Sliding Frictional Moment
        
        phi_bl = np.exp(-2.6e-8 * (n * self.visc)**1.4 * self.dm) # Sliding friction coefficient weighing factor

        # Movement constant
        if n == 0:
            mu_bl = 0.15
        else:
            mu_bl = 0.12
        mu_EHL = 0.1 # Highest coefficient, don't know what else to pick

        # Sliding friction coefficient
        mu_sl = phi_bl * mu_bl + (1 - phi_bl) * mu_EHL 

        M_sl = G_sl * mu_sl / 1000  # Convert to Nm

        # -------------------------
        # Drag Frictional Moment
        M_drag = 0
        V_M = 0
        M_drag_1 = 0
        M_drag_2 = 0

        if self.submerged:
            V_M = 0.0012 # Flatlines at full submersion
            M_drag_1 = 0.4 * V_M * self._K_ball * self.dm ** 5 * n**2
            M_drag_2 = 1.093e-7 * n**2 * self.dm ** 3 * (n * self.dm**2 * self._f_t / self.visc)**(-1.379) * self._R_s  
            M_drag_1 /= 1000 # Convert to Nm
            M_drag_2 /= 1000 # Convert to Nm
            M_drag = M_drag_1 + M_drag_2

        # -------------------------
        # Output
        self.M_total = M_rr + M_sl + M_drag
        if self.debug:
            print("-" * 40)
            print(f"Moments:")
            print(f"  {'Total moment':<30}: {self.M_total:<.2g}")
            print(f"  {'Rolling Moment':<30}: {M_rr:<.2g}")
            print(f"  {'Sliding Moment':<30}: {M_sl:<.2g}")
            print(f"  {'Drag Moment':<30}: {M_drag:<.2g}")
            print("-" * 40)
            print(f"Rolling Moment Parameters")
            print(f"  {'phi_ish':<30}: {phi_ish:<.2g}")
            print(f"  {'phi_rs':<30}: {phi_rs:<.2g}")
            print(f"  {'G_rr':<30}: {G_rr:<.2g}")
            print(f"Sliding Moment Parameters")
            print(f"  {'phi_bl':<30}: {phi_bl:<.2g}")
            print(f"  {'mu_bl':<30}: {mu_bl:<.2g}")
            print(f"  {'mu_EHL':<30}: {mu_EHL:<.2g}")
            print(f"  {'mu_sl':<30}: {mu_sl:<.2g}")
            print(f"  {'G_sl':<30}: {G_sl:<.2g}")
            if self.submerged:
                print(f"Drag Moment Parameters")
                print(f"  {'M_drag_1':<30}: {M_drag_1:<.2g}")
                print(f"  {'M_drag_2':<30}: {M_drag_2:<.2g}")
                print(f"  {'V_M':<30}: {V_M:<.2g}")
                print(f"  {'K_ball':<30}: {self._K_ball:<.2g}")
                print(f"  {'i_rw':<30}: {self._i_rw:<.2g}")
                print(f"  {'t':<30}: {self._t:<.2g}")
                print(f"  {'f_t':<30}: {self._f_t:<.2g}")
                print(f"  {'f_a':<30}: {self._f_a:<.2g}")
                print(f"  {'R_s':<30}: {self._R_s:<.2g}")
            print(f"Constants:")
            print(f"  {'K_z':<30}: {self.K_z:<.2g}")
            print(f"  {'K_rs':<30}: {self.K_rs:<.2g}")
            print(f"  {'R_1':<30}: {self.R_1:<.2g}")
            print(f"  {'R_2':<30}: {self.R_2:<.2g}")
            print(f"  {'S_1':<30}: {self.S_1:<.2g}")
            print(f"  {'S_2':<30}: {self.S_2:<.2g}")
        return self.M_total
    
    def power_loss(self, n: float, F_r_kN:float, F_a_kN: float) -> float:
        """
        Estimate steady-state frictional power loss in the bearing.

        Parameters
        ----------
        n : float
            Rotational speed in revolutions per minute (RPM).
        F_r_kN : float
            Radial load in kilonewtons (kN).
        F_a_kN : float
            Axial load in kilonewtons (kN).

        Returns
        -------
        float
            Power loss in watts (W).
        """
        M = self.total_moment(n, F_r_kN, F_a_kN)
        P = M * np.pi * n / 30

        if self.debug:
            print("-" * 40)
            print(f"  {"Total Power":<30}: {P:<.5g} W")
        return P

if __name__ == "__main__":
    # Example usage
    bearing_61900 = Bearing(d=10.0, D=22.0, series=619, visc=1.0, C_0_kN=1.27, submerged=True, debug=False)
    print(f"Moment(Nm): {bearing_61900.total_moment(n=30000, F_r_kN=0.5, F_a_kN=0.0):<5g}, Power(W):{bearing_61900.power_loss(n=30000, F_r_kN=0.5, F_a_kN=0):<5g}")
    print(f"Moment(Nm): {bearing_61900.total_moment(n=20000, F_r_kN=1.0, F_a_kN=0.5):<5g}, Power(W):{bearing_61900.power_loss(n=20000, F_r_kN=1.0, F_a_kN=0.5):<5g}")
    print(f"Moment(Nm): {bearing_61900.total_moment(n=10000, F_r_kN=2.0, F_a_kN=1.0):<5g}, Power(W):{bearing_61900.power_loss(n=10000, F_r_kN=2.0, F_a_kN=1.0):<5g}")
    
    bearing_61900.submerged = False
    bearing_61900.visc = 10.0
    bearing_61900.initialize_constants()

    print(f"Moment(Nm): {bearing_61900.total_moment(n=30000, F_r_kN=0.5, F_a_kN=0.0):<5g}, Power(W):{bearing_61900.power_loss(n=30000, F_r_kN=0.5, F_a_kN=0):<5g}")
    print(f"Moment(Nm): {bearing_61900.total_moment(n=20000, F_r_kN=1.0, F_a_kN=0.5):<5g}, Power(W):{bearing_61900.power_loss(n=20000, F_r_kN=1.0, F_a_kN=0.5):<5g}")
    print(f"Moment(Nm): {bearing_61900.total_moment(n=10000, F_r_kN=2.0, F_a_kN=1.0):<5g}, Power(W):{bearing_61900.power_loss(n=10000, F_r_kN=2.0, F_a_kN=1.0):<5g}")
    

    bearing_61900.debug = True
    bearing_61900.submerged = True
    bearing_61900.visc = 1.0

    bearing_61900.power_loss(n=30000, F_r_kN=1.0, F_a_kN=1.0)

