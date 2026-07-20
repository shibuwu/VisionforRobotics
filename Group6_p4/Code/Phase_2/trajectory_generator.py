import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.ndimage import uniform_filter1d


def make_waypoints(n_wp=8, xy_range=3.0, z_range=(5.0, 10.0), rng=None):
    rng = rng or np.random.default_rng()
    x = rng.uniform(-xy_range, xy_range, n_wp)
    y = rng.uniform(-xy_range, xy_range, n_wp)
    z = rng.uniform(z_range[0], z_range[1], n_wp)
    return np.stack([x, y, z], axis=1)


def make_orientations(pos, dt, max_rp=45.0, rng=None):
    rng = rng or np.random.default_rng()
    n = len(pos)
    max_rp_rad = np.deg2rad(max_rp)

    # yaw from velocity direction, heavily smoothed
    vel = np.gradient(pos, dt, axis=0)
    yaw = np.arctan2(vel[:, 1], vel[:, 0])
    yaw = np.unwrap(yaw)
    yaw = uniform_filter1d(yaw, size=2000)

    # roll/pitch: many control points with smaller amplitude, clamped to ±max_rp
    n_ctrl = max(20, n // 500)
    ctrl_t = np.linspace(0, n - 1, n_ctrl)
    t_all = np.arange(n, dtype=float)

    amp = max_rp_rad * 0.3
    roll_ctrl = rng.uniform(-amp, amp, n_ctrl)
    pitch_ctrl = rng.uniform(-amp, amp, n_ctrl)
    roll_ctrl[0] = roll_ctrl[-1] = 0.0
    pitch_ctrl[0] = pitch_ctrl[-1] = 0.0

    roll = np.clip(CubicSpline(ctrl_t, roll_ctrl, bc_type='clamped')(t_all),
                   -max_rp_rad, max_rp_rad)
    pitch = np.clip(CubicSpline(ctrl_t, pitch_ctrl, bc_type='clamped')(t_all),
                    -max_rp_rad, max_rp_rad)

    # blender camera default already looks down (-Z), so no base rotation needed
    quats = np.zeros((n, 4))
    for i in range(n):
        R = Rotation.from_euler('ZYX', [yaw[i], pitch[i], roll[i]])
        quats[i] = R.as_quat(scalar_first=True)
    return quats


def generate_trajectory(duration=10.0, freq=1000, n_wp=8,
                        xy_range=3.0, z_range=(5.0, 10.0),
                        max_rp=45.0, seed=None):
    rng = np.random.default_rng(seed)
    n = int(duration * freq)
    dt = 1.0 / freq

    wp = make_waypoints(n_wp, xy_range, z_range, rng)
    wp_t = np.linspace(0, duration, n_wp)
    t = np.linspace(0, duration, n, endpoint=False)

    splines = [CubicSpline(wp_t, wp[:, i], bc_type='clamped') for i in range(3)]
    pos = np.stack([s(t) for s in splines], axis=1)
    vel = np.stack([s(t, 1) for s in splines], axis=1)
    quats = make_orientations(pos, dt, max_rp, rng)

    # angular velocity from finite diff on quaternions
    omega = np.zeros((n, 3))
    for i in range(1, n):
        R_prev = Rotation.from_quat(quats[i-1], scalar_first=True)
        R_curr = Rotation.from_quat(quats[i], scalar_first=True)
        omega[i] = (R_prev.inv() * R_curr).as_rotvec() / dt
    for ax in range(3):
        omega[:, ax] = uniform_filter1d(omega[:, ax], size=21)

    return dict(timestamps=t, positions=pos, quaternions=quats,
                velocities=vel, angular_velocities=omega)


def relative_poses(quats, pos):
    n = len(pos)
    rel_t = np.zeros((n-1, 3))
    rel_q = np.zeros((n-1, 4))
    for i in range(n-1):
        R_prev = Rotation.from_quat(quats[i], scalar_first=True)
        R_curr = Rotation.from_quat(quats[i+1], scalar_first=True)
        rel_q[i] = (R_prev.inv() * R_curr).as_quat(scalar_first=True)
        rel_t[i] = R_prev.inv().apply(pos[i+1] - pos[i])
    return rel_t, rel_q


if __name__ == '__main__':
    traj = generate_trajectory(duration=10.0, seed=42)
    pos = traj['positions']
    vel = traj['velocities']
    omega = traj['angular_velocities']
    print(f"Samples: {len(pos)}")
    print(f"Max speed: {np.linalg.norm(vel, axis=1).max():.2f} m/s")
    print(f"Max angular rate: {np.rad2deg(np.linalg.norm(omega, axis=1).max()):.1f} deg/s")

    cam_idx = np.arange(0, len(pos), 10)
    rt, rq = relative_poses(traj['quaternions'][cam_idx], pos[cam_idx])
    print(f"Camera frames: {len(cam_idx)}, rel poses: {len(rt)}")
    print(f"Mean rel translation: {np.linalg.norm(rt, axis=1).mean():.4f} m")
