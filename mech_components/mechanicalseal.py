import numpy as np
from dataclasses import dataclass

"""
mechanicalseal.py

Utilities for representing and computing simple mechanical-seal quantities.

Notes
-----
This module's dataclass accepts input dimensions in millimetres (fields named *_mm).
Internally all length-based attributes are converted to meters (SI) in `__post_init__`.
"""

@dataclass
class MechanicalSeal:
    """
    Simple mechanical seal model.

    The dataclass fields ending with `_mm` are input dimensions in millimetres.
    Internally the class converts those to metres and provides computed attributes
    useful for friction/power estimates.

    Parameters
    ----------
    OD_mm : float
        Outer diameter of the seal (millimetres).
    ID_mm : float
        Inner diameter of the seal (millimetres).
    BD_mm : float
        Closing diameter (millimetres), usually shaft diameter if OD pressurized.
    F_sp : float
        Spring force (newtons).
    f : float, optional
        Friction coefficient (dimensionless), by default 0.07
    k : float, optional
        Pressure profile coefficient (dimensionless), by default 0.5
    debug : bool, optional
        Enable debug mode (default: False).

    Attributes
    ----------
    OD, ID, BD : float
        Diameters in metres (converted from input mm).
    a_0 : float
        Contact area in square metres, computed as pi/4 * (OD^2 - ID^2).
    b : float
        Seal balance parameter (dimensionless).
    P_sp : float
        Spring pressure in pascals (N/m^2).

    Notes
    -----
    - Inputs are expected in mm for geometric fields and N for forces.
    - Internal attributes for lengths are in metres (SI).
    """
    OD_mm: float
    ID_mm: float
    BD_mm: float
    F_sp: float
    f: float = 0.07
    k: float = 0.5
    debug: bool = False

    def __post_init__(self) -> None:
        # Convert dimensions from mm to m for internal use
        self.OD = self.OD_mm / 1000
        self.ID = self.ID_mm / 1000
        self.BD = self.BD_mm / 1000

        # Contact area
        self.a_0 = (np.pi / 4) * (self.OD**2 - self.ID**2)
        # Seal balance
        self.b = (self.OD**2 - self.BD**2) / (self.OD**2 - self.ID**2)
        # Spring pressure
        self.P_sp = self.F_sp / self.a_0
        # Mean radius
        self.r_m = (self.OD + self.ID) / 4


    def torque(self, n: float, P_delta: float) -> float:
        """
        Estimate torque due to friction in the seal.

        Parameters
        ----------
        n : float
            Rotational speed in revolutions per minute (RPM).
        P_delta : float
            Pressure difference across the seal in pascals (Pa).

        Returns
        -------
        float
            Torque in newton-meters (Nm).
        
        Notes
        -----
        If rotational speed is 0, starting torque is assumed and according to Karassik it would be around 3-5x running torque.
        """
        P_f = P_delta * (self.b - self.k) + self.P_sp

        if self.debug:
            print(f"Percentage of spring force: {self.P_sp / P_f * 100:.2f}%")
        T = P_f * self.a_0 * self.r_m * self.f
        if n == 0:
            # Starting torque is typically higher, assume 5x running torque
            T *= 5
        return T

    def power_loss(self, n: float, P_delta: float) -> float:
        """
        Estimate steady-state frictional power loss in the seal.

        Parameters
        ----------
        n : float
            Rotational speed in revolutions per minute (RPM).
        P_delta : float
            Pressure difference across the seal in pascals (Pa).

        Returns
        -------
        float
            Power loss in watts (W).
        """
        T = self.torque(n, P_delta)
        return T * np.pi * n / 30
    

if __name__ == "__main__":
    # Example usage
    mechseal_59U19 = MechanicalSeal(OD_mm=28.7, ID_mm=20.9, BD_mm=19.0, F_sp=50, debug=True)

    print(mechseal_59U19.power_loss(n=30000, P_delta=1000000))
    print(mechseal_59U19.torque(n=0, P_delta=0))
    print(mechseal_59U19.torque(n=30000, P_delta=1000000))

    print(mechseal_59U19.power_loss(n=20000, P_delta=1000000))
    print(mechseal_59U19.torque(n=20000, P_delta=1000000))