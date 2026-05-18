import ross as rs
import numpy as np

print("Initializing Rotordynamics Model...")

# Materials
aluminum = rs.Material(name="Aluminum", rho=2700, E=69e9, G_s=26e9)

# Shaft setup: 10mm solid steel shaft as per new plan
L = 0.095
i_d = 0.0
o_d = 0.010 # 10mm OD steel shaft
N_elem = 6

shaft_elements = [
    rs.ShaftElement(
        L=L/N_elem,
        idl=i_d,
        odl=o_d,
        material=aluminum,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    for _ in range(N_elem)
]

# Disks
# Impeller (SLA Plastic approximated by mass and inertia)
Id_gmm = 700 
Id = Id_gmm * 1e-9  
Ip_gmm = 1276 
Ip = Ip_gmm * 1e-9  
mass_kg = 8.728 * 1e-3  
impeller = rs.DiskElement(n=0, m=mass_kg, Ip=Ip, Id=Id, tag="Impeller")

# Turbine
turbine = rs.DiskElement(n=6, m=15.232e-3, Ip=7800e-9, Id=4000e-9, tag="Turbine")

# Coupler (10-10mm Aluminum Beam Coupler L25 D19 approximated as overhung mass at n=6)
# Mass ~15g
coupler_mass = 15e-3
coupler_Ip = coupler_mass * (0.019/2)**2 / 2
coupler_Id = coupler_mass * (3 * (0.019/2)**2 + 0.025**2) / 12
coupler = rs.DiskElement(n=6, m=coupler_mass, Ip=coupler_Ip, Id=coupler_Id, tag="Coupler")

# Bearings
bearing0 = rs.BallBearingElement(
    n=1,
    n_balls=9,
    d_balls=0.003,
    fs=20,
    alpha=0
)
bearing1 = rs.BallBearingElement(
    n=5,
    n_balls=9,
    d_balls=0.003,
    fs=20,
    alpha=0
)
bearings = [bearing0, bearing1]

# Assemble rotor
rotor = rs.Rotor(
    shaft_elements=shaft_elements, 
    disk_elements=[impeller, turbine, coupler], 
    bearing_elements=bearings
)

# Run modal analysis at operating speed
speed_rpm = 20000
speed_rads = speed_rpm * 2 * np.pi / 60
print(f"Running Modal Analysis at {speed_rpm} RPM ({speed_rads:.2f} rad/s)...")
modal = rotor.run_modal(speed=speed_rads, num_modes=12)

print("\n--- ROTORDYNAMICS ANALYSIS RESULTS ---")
print("Natural Frequencies:")
for i, wn in enumerate(modal.wn[:6]):
    print(f"Mode {i+1}: {wn:.2f} rad/s ({wn * 60 / (2 * np.pi):.2f} RPM)")

critical_speeds = [wn for wn in modal.wn if wn > 0]
if critical_speeds:
    first_crit = critical_speeds[0]
    margin = (first_crit - speed_rads) / speed_rads * 100
    print(f"\nMargin to first critical speed: {margin:.1f}%")
    if margin > 20:
        print("CONCLUSION: SURVIVAL LIKELY. First critical speed is safely above operating speed.")
    elif margin < 0:
        print("CONCLUSION: DANGER. Operating speed is ABOVE the first critical speed.")
    else:
        print("CONCLUSION: MARGINAL. First critical speed is too close to operating speed (<20% margin).")
else:
    print("No critical speeds found (rigid body modes or heavily damped).")

try:
    campbell = rotor.run_campbell(speed_range=np.linspace(0, 3000, 20))
    fig = campbell.plot()
    fig.write_html("C:/Users/Martin/Active/FYPTurbine/campbell_diagram.html")
    print("Campbell diagram saved to C:/Users/Martin/Active/FYPTurbine/campbell_diagram.html")
except Exception as e:
    print(f"Could not save Campbell diagram plot: {e}")
