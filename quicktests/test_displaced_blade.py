import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prop_components.turbine import Turbine
from prop_components.turbinemoc import SupersonicTurbineMOC
from prop_components.blade_profiler import DisplacedBladeProfiler

def main():
    print("Initializing test turbine...")
    # Setup test turbine properties
    turbine = Turbine(
        P=2600,            # 2.6 kW
        RPM=20000,         # 20,000 RPM
        d_mean_mm=100,     # 100 mm mean diameter
        mdot=0.034,        # 0.034 kg/s mass flow
        beta_deg=15.0,     # Nozzle exit angle (15 deg)
        doa=0.1,           # 10% degree of admission
        p_e=1.1e5          # Turbine exit pressure
    )
    
    # Initialize from cold gas (N2 testing parameters from notebook)
    turbine.from_inert_gas(R=296.8, gam=1.4, T01=300, nozzles=3)
    turbine.pretty_print()
    
    # Using matched Mach numbers from the velocity triangles (Mw3 = 1.39)
    moc = SupersonicTurbineMOC(
        gamma=1.4,
        mach_inlet=1.7, 
        mach_lower=1.3,
        mach_upper=2,
        beta_inlet_deg=75.0,  # Calculated from w3u/w3m
        dv=0.01
    )
    
    # Initialize the profiler
    print("\nInitializing DisplacedBladeProfiler (Sasman-Cresci)...")
    profiler = DisplacedBladeProfiler(turbine, moc, bl_method='sasman_cresci')
    
    # Evaluate boundary layers and run displacement
    profiler.evaluate_boundary_layers()
    profiler.displace_contour()
    
    # Separation Risk
    profiler.analyze_separation_risk()
    
    # Plotting
    save_path_contour = os.path.join(os.path.dirname(__file__), 'displaced_blade_contour.png')
    profiler.plot_displaced_contour(save_path=save_path_contour)
    
    save_path_bl = os.path.join(os.path.dirname(__file__), 'displaced_blade_bl_params.png')
    profiler.plot_boundary_layer_parameters(save_path=save_path_bl)

if __name__ == "__main__":
    main()
