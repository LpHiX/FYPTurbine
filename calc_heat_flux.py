import sys
import os

# Add the active directory to path so we can import the module
sys.path.append(os.path.abspath("C:/Users/Martin/Active/FYPTurbine"))

from mech_components.bearing import Bearing

def calculate_prototype_heat():
    # 61900 Bearing Parameters
    # C_0_kN = 1.27 (from SKF/Standard tables for 61900)
    # visc = 20 mm2/s (Standard ISO VG 32 or 46 grease at ~40-50C)
    bearing = Bearing(
        d=10.0, 
        D=22.0, 
        series=619, 
        visc=20.0, 
        C_0_kN=1.27, 
        submerged=False, 
        debug=True
    )

    # Loads
    # Assume 50N radial (unbalance/shaft weight) and 10N axial (residual thrust)
    n = 17000
    F_r_kN = 0.05
    F_a_kN = 0.01

    print(f"--- BEARING THERMAL ANALYSIS AT {n} RPM ---")
    power_loss = bearing.power_loss(n, F_r_kN, F_a_kN)
    
    print("\n--- RESULTS ---")
    print(f"Heat Flux into Housing: {power_loss:.2f} Watts")
    
    # Simple thermal check
    # PLA Thermal Conductivity ~0.13 W/mK (extremely low, like wood/wool)
    # This heat will stay in the bearing race and the immediate PLA interface.
    print(f"\nNOTE: {power_loss:.2f}W is roughly equivalent to a 10-watt soldering iron tip.")
    print("In a PLA housing, this will reach 60°C (Tg) very quickly.")

if __name__ == "__main__":
    calculate_prototype_heat()
