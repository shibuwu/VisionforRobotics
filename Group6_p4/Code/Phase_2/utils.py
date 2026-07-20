import numpy as np

def wxyz_to_xyzw(q):
    q = np.asarray(q)
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)

def xyzw_to_wxyz(q):
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)

def quat_canonical(q):
    q = np.asarray(q)
    sign = np.where(q[..., :1] < 0, -1.0, 1.0)
    return q * sign

def quat_normalize(q, eps=1e-12):
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)

def quat_multiply_wxyz(q1, q2):
    (w1, x1, y1, z1) = (q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3])
    (w2, x2, y2, z2) = (q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3])
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)

def quat_inverse_wxyz(q):
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1
    return out

def quat_rotate_vec_wxyz(q, v):
    v_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_multiply_wxyz(quat_multiply_wxyz(q, v_quat), quat_inverse_wxyz(q))[..., 1:]
