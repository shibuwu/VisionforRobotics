import os
import time
import numpy as np
from multiprocessing import Process, Queue


class Viewer(object):
    def __init__(self, save_dir='vio_output', save_frames=True, frame_every=2):
        os.makedirs(save_dir, exist_ok=True)
        if save_frames:
            os.makedirs(os.path.join(save_dir, 'frames'), exist_ok=True)

        self.image_queue = Queue()
        self.pose_queue  = Queue()

        # trajectory.tum: timestamp tx ty tz qx qy qz qw
        self.traj_path = os.path.join(save_dir, 'trajectory.tum')
        self._traj_file = open(self.traj_path, 'w')
        self._traj_file.write('# timestamp tx ty tz qx qy qz qw\n')

        self.view_proc = Process(
            target=_view_loop,
            args=(self.pose_queue, self.image_queue, save_dir, save_frames, frame_every),
            daemon=True,
        )
        self.view_proc.start()

    def update_pose(self, pose, timestamp=None):
        if pose is None:
            return
        try:
            m = pose.matrix()
        except AttributeError:
            m = np.asarray(pose)

        t = m[:3, 3]
        R = m[:3, :3]
        q_jpl = _rotmat_to_quat_jpl(R)
        q_ham = _jpl_to_hamilton(q_jpl)
        ts = timestamp if timestamp is not None else time.time()
        self._traj_file.write(
            f'{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} '
            f'{q_ham[0]:.6f} {q_ham[1]:.6f} {q_ham[2]:.6f} {q_ham[3]:.6f}\n'
        )
        self._traj_file.flush()

        try:
            self.pose_queue.put_nowait(m)
        except Exception:
            pass

    def update_image(self, image):
        if image is None:
            return
        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        if self.image_queue.qsize() < 3:
            try:
                self.image_queue.put_nowait(image)
            except Exception:
                pass


def _rotmat_to_quat_jpl(R):
    # JPL quaternion [x, y, z, w], same convention as utils.to_quaternion
    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            q = [t, R[0, 1] + R[1, 0], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]]
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            q = [R[0, 1] + R[1, 0], t, R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            q = [R[0, 2] + R[2, 0], R[2, 1] + R[1, 2], t, R[0, 1] - R[1, 0]]
        else:
            t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
            q = [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], t]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _jpl_to_hamilton(q_jpl):
    # q_hamilton = conj(q_jpl). TUM/evo expect Hamilton [qx qy qz qw].
    return np.array([-q_jpl[0], -q_jpl[1], -q_jpl[2], q_jpl[3]])


def _view_loop(pose_queue, image_queue, save_dir, save_frames, frame_every):
    import matplotlib
    if 'DISPLAY' in os.environ:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            matplotlib.use('Agg')
    else:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    interactive = matplotlib.get_backend() != 'Agg'
    if interactive:
        plt.ion()

    fig = plt.figure(figsize=(13, 6))
    ax3d  = fig.add_subplot(1, 2, 1, projection='3d')
    aximg = fig.add_subplot(1, 2, 2)

    ax3d.set_title('Estimated trajectory (cam0 in world)')
    ax3d.set_xlabel('x [m]'); ax3d.set_ylabel('y [m]'); ax3d.set_zlabel('z [m]')
    aximg.set_title('cam0')
    aximg.axis('off')

    traj = []
    line_handle, = ax3d.plot([], [], [], '-', color='black', linewidth=1.2)
    head_handle, = ax3d.plot([], [], [], 'o', color='blue', markersize=6)
    img_handle = None
    cur_image = None

    pose_count = 0
    frame_idx  = 0
    last_redraw = time.time()

    try:
        while True:
            got_pose = False
            new_poses = []
            while True:
                try:
                    new_poses.append(pose_queue.get_nowait())
                    got_pose = True
                except Exception:
                    break
            for m in new_poses:
                traj.append(m[:3, 3])
                pose_count += 1

            while True:
                try:
                    cur_image = image_queue.get_nowait()
                except Exception:
                    break

            now = time.time()
            need_redraw = got_pose or (cur_image is not None and now - last_redraw > 0.1)

            if need_redraw and len(traj) > 0:
                arr = np.asarray(traj)
                line_handle.set_data(arr[:, 0], arr[:, 1])
                line_handle.set_3d_properties(arr[:, 2])
                head_handle.set_data([arr[-1, 0]], [arr[-1, 1]])
                head_handle.set_3d_properties([arr[-1, 2]])

                pad = 1.0
                xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
                ax3d.set_xlim(xs.min() - pad, xs.max() + pad)
                ax3d.set_ylim(ys.min() - pad, ys.max() + pad)
                ax3d.set_zlim(zs.min() - pad, zs.max() + pad)

                if cur_image is not None:
                    if img_handle is None:
                        img_handle = aximg.imshow(cur_image, cmap='gray' if cur_image.ndim == 2 else None)
                    else:
                        img_handle.set_data(cur_image)

                if interactive:
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                if save_frames and got_pose and pose_count % frame_every == 0:
                    fig.savefig(
                        os.path.join(save_dir, 'frames', f'frame_{frame_idx:06d}.png'),
                        dpi=80,
                    )
                    frame_idx += 1

                last_redraw = now

            if interactive:
                if not plt.fignum_exists(fig.number):
                    break
                plt.pause(0.01)
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    if len(traj) > 0:
        fig.savefig(os.path.join(save_dir, 'final_trajectory.png'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    import numpy as np
    class FakePose:
        def __init__(self, m): self._m = m
        def matrix(self): return self._m

    v = Viewer(save_frames=False)
    for i in range(500):
        t_ = i * 0.02
        pos = np.array([np.cos(t_), np.sin(t_), t_ * 0.1])
        m = np.eye(4); m[:3, 3] = pos
        v.update_pose(FakePose(m), timestamp=t_)
        time.sleep(0.01)
    time.sleep(2)