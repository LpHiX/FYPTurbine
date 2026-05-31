import numpy as np
import matplotlib.pyplot as plt

# Assumptions
p_max_barg = 6.5
p_atm_bar = 1.01325
gamma = 1.4
R = 287.05  # J/(kg.K)
T0 = 293.15  # 20 C in Kelvin
density_fad = 1.204  # kg/m^3 at 20 C, 1 atm

# 8mm ID tube
d_tube = 0.008 # m
A_tube = np.pi / 4 * d_tube**2

# We will plot from 0 to max choked flow
# Theoretical choked flow mass rate:
# m_dot = A * P0 * sqrt(gamma / (R*T0)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
choked_term = np.sqrt(gamma / (R * T0)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
m_dot_max_kg_s = A_tube * (p_max_barg + p_atm_bar) * 1e5 * choked_term
m_dot_max_g_s = m_dot_max_kg_s * 1000

# Convert to FAD volume flow
v_dot_max_fad_m3_s = m_dot_max_kg_s / density_fad
v_dot_max_fad_dm3_s = v_dot_max_fad_m3_s * 1000

print(f"Estimated Max Choked Mass Flow: {m_dot_max_g_s:.2f} g/s")
print(f"Estimated Max FAD Volume Flow: {v_dot_max_fad_dm3_s:.2f} dm^3/s")

# Let's create an array of volume flows (FAD dm^3/s)
v_dot_dm3_s = np.linspace(0, v_dot_max_fad_dm3_s * 1.05, 100)
m_dot_g_s = v_dot_dm3_s * density_fad  # linear relationship since it's FAD

# Estimate pressure drop. Assuming a simple quadratic head loss model where P drops to 0 barg at max flow
# P_supply(Q) = P_max - k * Q^2
# P_max - k * (v_dot_max_fad_dm3_s)^2 = 0 (assuming all pressure is lost to friction + dynamic pressure at max flow)
k_drop = p_max_barg / (v_dot_max_fad_dm3_s**2)

pressure_barg = p_max_barg - k_drop * (v_dot_dm3_s**2)
pressure_barg[pressure_barg < 0] = 0

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Air Flow (Standard dm$^3$/s)')
ax1.set_ylabel('Supply Pressure (barg)', color=color1)
line1, = ax1.plot(v_dot_dm3_s, pressure_barg, color=color1, label='Pressure (barg)', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, p_max_barg * 1.1)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color2 = 'tab:red'
ax2.set_ylabel('Mass Flow (g/s)', color=color2)
line2, = ax2.plot(v_dot_dm3_s, m_dot_g_s, color=color2, linestyle='--', label='Mass Flow (g/s)', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, max(m_dot_g_s) * 1.1)

# Added title and legends
plt.title(f"Estimated Pre-Regulator Shop Air Supply (8mm ID Tube)\nMax static: {p_max_barg} barg")
fig.tight_layout()

# Legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left')

plt.savefig("shop_air_estimate.png", dpi=300)
print("Plot saved as shop_air_estimate.png")
