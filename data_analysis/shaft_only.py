import polars as pl
import h5py
import numpy as np
shaftonly = h5py.File(r"D:\Projects\propbackend_logs\2026-05-01\mainloop\test_20260501_135954_mainloop.h5", "r")
time = shaftonly["channels"]["adc_rpm_mv"]["time"][:]
mask = (time >= 875) & (time <= 1034)
time = time[mask]
rpm = shaftonly["channels"]["adc_rpm_mv"]["data"][:][mask]
torque = shaftonly["channels"]["adc_torque_mv"]["data"][:][mask]
in_pt = shaftonly["channels"]["adc_pt_in_mv"]["data"][:][mask]
motor_throttle = shaftonly["channels"]["servos_motor_angle"]["data"][:][mask]
from matplotlib import pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
ax[0].plot(time, rpm, color="#1f77b4"); ax[0].set_ylabel("RPM")
ax[1].plot(time, torque, color="#d62728"); ax[1].set_ylabel("Torque (Nm)")
ax[2].plot(time, in_pt, color="#2ca02c"); ax[2].set_ylabel("Inlet pt (bar(g))"); ax[2].set_xlabel("Time (s)")
ax[3].plot(time, motor_throttle, color="#ff7f0e"); ax[3].set_ylabel("Motor Throttle"); ax[3].set_xlabel("Time (s)")
for a in ax:
    a.grid(alpha=0.25)
fig.tight_layout()
plt.show()