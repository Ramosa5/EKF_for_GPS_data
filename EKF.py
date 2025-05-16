import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("LeesburgToIndy.csv", delimiter=';', skiprows=1)

# Extract and rename relevant columns
df = df[[
    'Time since start in ms ',
    'LINEAR ACCELERATION X (m/s²)',
    'LINEAR ACCELERATION Y (m/s²)',
    'LOCATION Latitude : ',
    'LOCATION Longitude : ',
    'LOCATION Speed ( Kmh)'
]].copy()
df.columns = ['time_ms', 'acc_x', 'acc_y', 'lat', 'lon', 'speed_kmh']

# Clean data
df.dropna(subset=['lat', 'lon'], inplace=True)
df['time_s'] = df['time_ms'] / 1000.0
df['speed_mps'] = df['speed_kmh'] * (1000 / 3600)

# Convert lat/lon to UTM coordinates (meters)
transformer = Transformer.from_crs("epsg:4326", "epsg:32618", always_xy=True)
df['x'], df['y'] = transformer.transform(df['lon'].values, df['lat'].values)

# Prepare data
dt_list = df['time_s'].diff().fillna(0).values
acc_x = df['acc_x'].values
acc_y = df['acc_y'].values
gps_x = df['x'].values
gps_y = df['y'].values

# Initial state
x = np.array([gps_x[0], gps_y[0], 0, 0])
P = np.eye(4)
Q = np.eye(4) * 0.1
R = np.eye(2) * 5
I = np.eye(4)

# EKF functions
def f(x, u, dt):
    px, py, vx, vy = x
    ax, ay = u
    return np.array([
        px + vx * dt + 0.5 * ax * dt**2,
        py + vy * dt + 0.5 * ay * dt**2,
        vx + ax * dt,
        vy + ay * dt
    ])

def F_jacobian(dt):
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# Run EKF
estimates = []
for i in range(len(df)):
    dt = dt_list[i]
    u = np.array([acc_x[i], acc_y[i]])

    # Predict
    x_pred = f(x, u, dt)
    F = F_jacobian(dt)
    P_pred = F @ P @ F.T + Q

    # Update
    z = np.array([gps_x[i], gps_y[i]])
    y_k = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y_k
    P = (I - K @ H) @ P_pred

    estimates.append(x.copy())

# Result DataFrame
ekf_result = pd.DataFrame(estimates, columns=['ekf_x', 'ekf_y', 'ekf_vx', 'ekf_vy'])
ekf_result['gps_x'] = gps_x
ekf_result['gps_y'] = gps_y
ekf_result['time_s'] = df['time_s'].values

# --- Visualization ---

# Plot 1: Trajectory
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['y'], label='GPS Raw', linestyle='--', alpha=0.6)
plt.plot(ekf_result['ekf_x'], ekf_result['ekf_y'], label='EKF Estimated', linewidth=2)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('GPS vs EKF Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot 2: Velocity Magnitude Over Time
velocity = np.sqrt(ekf_result['ekf_vx']**2 + ekf_result['ekf_vy']**2)

plt.figure(figsize=(10, 4))
plt.plot(ekf_result['time_s'], velocity, label='EKF Speed (m/s)', color='tab:green')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Estimated Speed Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
