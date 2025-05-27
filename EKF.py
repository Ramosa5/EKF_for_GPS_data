import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

# WGS84 constants for conversion
a = 6378137.0  # semi-major axis
f = 1 / 298.257223563  # flattening
b = a * (1 - f)  # semi-minor axis
e_sq = f * (2 - f)  # eccentricity squared

def llh_to_ecef(lat, lon, alt):
    """Convert lat, lon (deg), alt (m) to ECEF coordinates (meters)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_sq) + alt) * np.sin(lat_rad)
    return np.array([x, y, z])

def ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref):
    """Convert ECEF xyz to local ENU coordinates relative to reference."""
    ref_xyz = llh_to_ecef(lat_ref, lon_ref, alt_ref)
    lat_rad = np.radians(lat_ref)
    lon_rad = np.radians(lon_ref)
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ])
    diff = np.array([x, y, z]) - ref_xyz
    enu = R @ diff
    return enu

def llh_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    x, y, z = llh_to_ecef(lat, lon, alt)
    enu = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)
    return enu

class KalmanFilter:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((6, 1))  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1.0
        q = 1e-3
        self.Q = np.eye(6) * q
        r = 5.0
        self.R = np.eye(3) * r

    def f(self, x, a):
        dt = self.dt
        pos = x[:3]
        vel = x[3:]
        new_pos = pos + vel * dt + 0.5 * a.reshape((3, 1)) * dt**2
        new_vel = vel + a.reshape((3, 1)) * dt
        return np.vstack((new_pos, new_vel))

    def F_jacobian(self):
        dt = self.dt
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def h(self, x):
        return x[:3]

    def H_jacobian(self):
        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        return H

    def predict(self, a):
        self.x = self.f(self.x, a)
        F = self.F_jacobian()
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.reshape(z, (3, 1))
        y = z - self.h(self.x)
        H = self.H_jacobian()
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def get_state(self):
        return self.x.flatten()

def load_sample(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')
    return list(map(float, lines))

def process_folder(data_folder, timestamp_file, dt_default=0.1):
    with open(timestamp_file, 'r') as f:
        timestamps = list(map(int, f.read().strip().split('\n')))

    files = sorted(glob(os.path.join(data_folder, '*.txt')))
    assert len(files) == len(timestamps), "Files and timestamps count mismatch"

    first_sample = load_sample(files[0])
    lat_ref, lon_ref, alt_ref = first_sample[0], first_sample[1], first_sample[2]

    kf = KalmanFilter(dt=dt_default)
    filtered_states = []

    prev_vel_enu = None
    prev_time = timestamps[0]

    for i, file in enumerate(files):
        sample = load_sample(file)
        time = timestamps[i]

        lat, lon, alt = sample[0], sample[1], sample[2]
        pos_enu = llh_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref)

        v_ned = np.array([sample[7], sample[8], sample[9]])
        v_enu = np.array([v_ned[1], v_ned[0], -v_ned[2]])

        dt = (time - prev_time) * 1e-9
        if dt <= 0:
            dt = dt_default

        kf.dt = dt

        if i == 0:
            kf.x[:3, 0] = pos_enu
            kf.x[3:, 0] = v_enu
            prev_vel_enu = v_enu
        else:
            a = (v_enu - prev_vel_enu) / dt
            prev_vel_enu = v_enu
            kf.predict(a)
            kf.update(pos_enu)

        prev_time = time
        filtered_states.append(kf.get_state())

    return np.array(filtered_states)

# Example usage:
data_folder = r'dataset\gps_imu\data'
timestamp_file = r"dataset\gps_imu\data_timestamp.txt"
filtered_results = process_folder(data_folder, timestamp_file)

east = filtered_results[:, 0]
north = filtered_results[:, 1]
up = filtered_results[:, 2]

# Plot filtered 2D trajectory
plt.figure(figsize=(10, 8))
plt.plot(east, north, label='Kalman Filter Position')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Filtered 2D Trajectory (East-North Plane)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# Plot filtered altitude over time
plt.figure(figsize=(10, 4))
plt.plot(up, label='Altitude (Up)')
plt.xlabel('Sample index')
plt.ylabel('Altitude (m)')
plt.title('Filtered Altitude over Time')
plt.legend()
plt.grid(True)
plt.show()

# Load raw GPS positions
raw_positions = []
raw_up = []
files = sorted(glob(os.path.join(data_folder, '*.txt')))
first_sample = load_sample(files[0])
lat_ref, lon_ref, alt_ref = first_sample[0], first_sample[1], first_sample[2]

for f in files:
    sample = load_sample(f)
    lat, lon, alt = sample[0], sample[1], sample[2]
    pos_enu = llh_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref)
    raw_positions.append(pos_enu)
    raw_up.append(pos_enu[2])

raw_positions = np.array(raw_positions)
raw_up = np.array(raw_up)

# Plot: Raw vs Filtered Trajectory
plt.figure(figsize=(10, 8))
plt.plot(raw_positions[:, 0], raw_positions[:, 1], label='Raw GPS Position', alpha=0.6)
plt.plot(east, north, label='Filtered Position')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.title('Raw GPS vs Filtered Trajectory')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

# NEW: Raw vs Filtered Altitude
plt.figure(figsize=(10, 4))
plt.plot(raw_up, label='Raw Altitude (Up)', alpha=0.6)
plt.plot(up, label='Filtered Altitude (Up)')
plt.xlabel('Sample Index')
plt.ylabel('Altitude (m)')
plt.title('Raw vs Filtered Altitude')
plt.legend()
plt.grid(True)
plt.show()
