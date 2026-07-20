"""
IMU noise model stripped from gnss-ins-sim via OysterSim.
https://github.com/Aceinna/gnss-ins-sim
https://github.com/prgumd/Oystersim
"""
import numpy as np
import math
from scipy.spatial.transform import Rotation

D2R = math.pi / 180
GRAVITY = np.array([0.0, 0.0, 9.81])

# mid-accuracy IMU params (from gnss-ins-sim, similar to IMU381)
GYRO_ERR = {
    'b': np.array([0.0, 0.0, 0.0]) * D2R,
    'b_drift': np.array([3.5, 3.5, 3.5]) * D2R / 3600.0,
    'b_corr': np.array([100.0, 100.0, 100.0]),
    'arw': np.array([0.25, 0.25, 0.25]) * D2R / 60.0,
}
ACCEL_ERR = {
    'b': np.array([0.0, 0.0, 0.0]),
    'b_drift': np.array([5.0e-5, 5.0e-5, 5.0e-5]),
    'b_corr': np.array([100.0, 100.0, 100.0]),
    'vrw': np.array([0.03, 0.03, 0.03]) / 60.0,
}


def bias_drift(corr_time, drift, n, fs):
    # first-order Gauss-Markov bias drift model
    out = np.zeros((n, 3))
    for ax in range(3):
        if not math.isinf(corr_time[ax]):
            a = 1 - 1 / fs / corr_time[ax]
            b = drift[ax] * np.sqrt(1.0 - np.exp(-2 / (fs * corr_time[ax])))
            noise = np.random.randn(n)
            for j in range(1, n):
                out[j, ax] = a * out[j-1, ax] + b * noise[j-1]
        else:
            out[:, ax] = drift[ax] * np.random.randn(n)
    return out


def add_gyro_noise(ref_w, fs, gyro_err=None):
    if gyro_err is None:
        gyro_err = GYRO_ERR
    dt = 1.0 / fs
    n = ref_w.shape[0]
    drift = bias_drift(gyro_err['b_corr'], gyro_err['b_drift'], n, fs)
    noise = np.random.randn(n, 3)
    for ax in range(3):
        noise[:, ax] *= gyro_err['arw'][ax] / math.sqrt(dt)
    return ref_w + gyro_err['b'] + drift + noise


def add_accel_noise(ref_a, fs, accel_err=None):
    if accel_err is None:
        accel_err = ACCEL_ERR
    dt = 1.0 / fs
    n = ref_a.shape[0]
    drift = bias_drift(accel_err['b_corr'], accel_err['b_drift'], n, fs)
    noise = np.random.randn(n, 3)
    for ax in range(3):
        noise[:, ax] *= accel_err['vrw'][ax] / math.sqrt(dt)
    return ref_a + accel_err['b'] + drift + noise


def simulate_imu(traj, freq=1000):
    pos = traj['positions']
    quats = traj['quaternions']
    omega_body = traj['angular_velocities']
    dt = 1.0 / freq
    n = len(pos)

    # true acceleration in world frame from velocity
    acc_world = np.gradient(traj['velocities'], dt, axis=0)

    # transform to body frame: accel_body = R^T * (a_world + g)
    true_accel = np.zeros((n, 3))
    for i in range(n):
        R = Rotation.from_quat(quats[i], scalar_first=True)
        true_accel[i] = R.inv().apply(acc_world[i] + GRAVITY)

    gyro = add_gyro_noise(omega_body, freq)
    accel = add_accel_noise(true_accel, freq)

    return dict(gyro=gyro, accel=accel, timestamps=traj['timestamps'])


if __name__ == '__main__':
    from trajectory_generator import generate_trajectory
    traj = generate_trajectory(duration=10.0, seed=42)
    imu = simulate_imu(traj)
    print(f"IMU samples: {len(imu['gyro'])}")
    print(f"Gyro range: [{imu['gyro'].min():.2f}, {imu['gyro'].max():.2f}] rad/s")
    print(f"Accel range: [{imu['accel'].min():.2f}, {imu['accel'].max():.2f}] m/s^2")
    print(f"Accel norm mean: {np.mean(np.linalg.norm(imu['accel'], axis=1)):.2f}")
