import numpy as np
import os
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import estimate_fundamental_matrix
from GetInliersRANSAC import get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_F
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import nonlinear_triangulation
from LinearPnP import linear_pnp, pnp_reprojection_error
from PnPRANSAC import pnp_ransac
from NonlinearPnP import nonlinear_pnp
from BuildVisibilityMatrix import build_visibility_matrix
from BundleAdjustment import bundle_adjustment

np.random.seed(42)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_LOCAL = os.path.join(THIS_DIR, 'P2Data')
DATA_DIR_PARENT = os.path.join(THIS_DIR, '..', 'P2Data')
DATA_DIR = DATA_DIR_LOCAL if os.path.exists(DATA_DIR_LOCAL) else DATA_DIR_PARENT

K = np.loadtxt(os.path.join(DATA_DIR, 'calibration.txt'))

def parse_matches(filepath):
    features = []
    with open(filepath, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            n_matches = int(parts[0])
            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            u_src, v_src = float(parts[4]), float(parts[5])
            matches = {}
            idx = 6
            for _ in range(n_matches - 1):
                img_id = int(parts[idx])
                u_m = float(parts[idx + 1])
                v_m = float(parts[idx + 2])
                matches[img_id] = (u_m, v_m)
                idx += 3
            features.append({
                'rgb': (r, g, b),
                'u': u_src, 'v': v_src,
                'matches': matches
            })
    return features

all_features = {}
for i in range(1, 5):
    all_features[i] = parse_matches(os.path.join(DATA_DIR, f'matching{i}.txt'))

def build_correspondences(all_features):
    corr = {}
    for src_id, features in all_features.items():
        for feat in features:
            for dst_id, (u_dst, v_dst) in feat['matches'].items():
                pair = (src_id, dst_id)
                if pair not in corr:
                    corr[pair] = {'pts1': [], 'pts2': [], 'rgb': []}
                corr[pair]['pts1'].append([feat['u'], feat['v']])
                corr[pair]['pts2'].append([u_dst, v_dst])
                corr[pair]['rgb'].append(feat['rgb'])
    for pair in corr:
        corr[pair]['pts1'] = np.array(corr[pair]['pts1'])
        corr[pair]['pts2'] = np.array(corr[pair]['pts2'])
        corr[pair]['rgb'] = np.array(corr[pair]['rgb'])
    return corr

correspondences = build_correspondences(all_features)


def compute_reproj_error(P, X, pts):
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    proj = (P @ X_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.mean(np.sqrt(np.sum((pts - proj)**2, axis=1)))

def project_points(P, X):
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    proj = (P @ X_h.T).T
    return proj[:, :2] / proj[:, 2:3]

def in_image_mask(proj, width, height):
    finite = np.isfinite(proj).all(axis=1)
    inside_x = (proj[:, 0] >= 0) & (proj[:, 0] < width)
    inside_y = (proj[:, 1] >= 0) & (proj[:, 1] < height)
    return finite & inside_x & inside_y

def plot_ransac_matches(img1_path, img2_path, all_pts1, all_pts2, inlier_idx, out_path):
    img1 = plt.imread(img1_path)
    img2 = plt.imread(img2_path)
    h1, w1 = img1.shape[:2]
    # side-by-side canvas
    canvas = np.concatenate([img1, img2], axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(canvas)
    inlier_set = set(inlier_idx)
    for i in range(all_pts1.shape[0]):
        x1, y1 = all_pts1[i]
        x2, y2 = all_pts2[i, 0] + w1, all_pts2[i, 1]
        color = 'lime' if i in inlier_set else 'red'
        ax.plot([x1, x2], [y1, y2], c=color, linewidth=0.4, alpha=0.6)
    ax.set_axis_off()
    ax.set_title('RANSAC feature matching')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_pose_disambiguation(poses, K, pts1, pts2, out_path):
    C1 = np.zeros(3)
    R1 = np.eye(3)
    colors = ['#b56551', '#ff00ff', '#7a3db8', '#3455db']

    fig, ax = plt.subplots(figsize=(7, 7))
    all_xz = []
    for i, (C2, R2) in enumerate(poses):
        X = linear_triangulation(K, C1, R1, C2, R2, pts1, pts2)
        finite = np.isfinite(X).all(axis=1)
        X_plot = X[finite]
        if X_plot.shape[0] > 0:
            ax.scatter(X_plot[:, 0], X_plot[:, 2], c=colors[i], s=5, alpha=0.9, linewidths=0)
            all_xz.append(X_plot[:, [0, 2]])

    # Set axis limits from percentiles so outliers don't blow up the scale
    if all_xz:
        combined = np.vstack(all_xz)
        x_lo, x_hi = np.percentile(combined[:, 0], [5, 95])
        z_lo, z_hi = np.percentile(combined[:, 1], [5, 95])
        pad_x = 0.15 * max(x_hi - x_lo, 1)
        pad_z = 0.15 * max(z_hi - z_lo, 1)
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(z_lo - pad_z, z_hi + pad_z)

    ax.set_title('initial triangulation')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def draw_camera_marker(ax, C, R, color, size=0.8):
    direction = R[2, :]
    dx, dz = direction[0], direction[2]
    norm = np.sqrt(dx**2 + dz**2)
    if norm > 1e-6:
        dx, dz = dx / norm, dz / norm
    px, pz = -dz, dx
    tip = [C[0] + dx * size, C[2] + dz * size]
    left = [C[0] - dx * size * 0.3 + px * size * 0.5, C[2] - dz * size * 0.3 + pz * size * 0.5]
    right = [C[0] - dx * size * 0.3 - px * size * 0.5, C[2] - dz * size * 0.3 - pz * size * 0.5]
    tri = plt.Polygon([tip, left, right], closed=True, facecolor=color, edgecolor='k', linewidth=0.5, zorder=5)
    ax.add_patch(tri)

def plot_linear_vs_nonlinear(X_linear, X_nonlinear, C1, R1, C2, R2, out_path):
    finite = np.isfinite(X_nonlinear).all(axis=1) & np.isfinite(X_linear).all(axis=1)
    Xl = X_linear[finite]
    Xn = X_nonlinear[finite]

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.scatter(Xl[:, 0], Xl[:, 2], c='r', s=12, alpha=0.7, linewidths=0, label='linear')
    ax.scatter(Xn[:, 0], Xn[:, 2], c='b', s=6, alpha=0.8, linewidths=0, label='nonlinear')

    draw_camera_marker(ax, C1, R1, 'g', size=0.8)
    draw_camera_marker(ax, C2, R2, 'm', size=0.8)

    combined = np.vstack([Xl, Xn])
    x_lo, x_hi = np.percentile(combined[:, 0], [5, 95])
    z_lo, z_hi = np.percentile(combined[:, 2], [5, 95])
    x_lo = min(x_lo, C1[0], C2[0]) - 1
    z_lo = min(z_lo, C1[2], C2[2]) - 2
    pad_x = 0.15 * max(x_hi - x_lo, 1)
    pad_z = 0.15 * max(z_hi - z_lo, 1)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(z_lo - pad_z, z_hi + pad_z)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.legend(loc='upper left')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_reprojection_comparison(image1_path, image2_path, pts1, pts2, X_linear, X_nonlinear, P1, P2, out_path):
    img1 = plt.imread(image1_path)
    img2 = plt.imread(image2_path)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    proj1_lin = project_points(P1, X_linear)
    proj2_lin = project_points(P2, X_linear)
    proj1_nonlin = project_points(P1, X_nonlinear)
    proj2_nonlin = project_points(P2, X_nonlinear)

    m1_lin = in_image_mask(proj1_lin, w1, h1)
    m2_lin = in_image_mask(proj2_lin, w2, h2)
    m1_non = in_image_mask(proj1_nonlin, w1, h1)
    m2_non = in_image_mask(proj2_nonlin, w2, h2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    entries = [
        (axes[0, 0], img1, pts1, proj1_lin, m1_lin, 'lin reproj vis, frame 0'),
        (axes[0, 1], img2, pts2, proj2_lin, m2_lin, 'lin reproj vis, frame 1'),
        (axes[1, 0], img1, pts1, proj1_nonlin, m1_non, 'nonlin reproj vis, frame 0'),
        (axes[1, 1], img2, pts2, proj2_nonlin, m2_non, 'nonlin reproj vis, frame 1'),
    ]
    for ax, img, pts, proj, mask, title in entries:
        ax.imshow(img)
        ax.scatter(proj[mask, 0], proj[mask, 1], c='r', s=8, marker='.', linewidths=0, alpha=0.9)
        ax.scatter(pts[:, 0], pts[:, 1], c='lime', s=12, marker='.', linewidths=0, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_pnp_comparison(pnp_results, K, out_path):
    C1 = np.zeros(3)
    R1 = np.eye(3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    target_order = [3, 4, 5]
    result_by_target = {r['target']: r for r in pnp_results}

    for ax, target in zip(axes, target_order):
        res = result_by_target.get(target)
        if res is None:
            ax.text(0.5, 0.5, f'(1,{target})\ninsufficient matches',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue

        pts1 = res['pts1']
        pts_target = res['pts_target']
        C_lin, R_lin = res['linear_pose']
        C_non, R_non = res['nonlinear_pose']

        max_pts = 30
        if pts1.shape[0] > max_pts:
            sub_idx = np.random.choice(pts1.shape[0], max_pts, replace=False)
            pts1 = pts1[sub_idx]
            pts_target = pts_target[sub_idx]

        X_lin = linear_triangulation(K, C1, R1, C_lin, R_lin, pts1, pts_target)
        X_non = linear_triangulation(K, C1, R1, C_non, R_non, pts1, pts_target)

        finite = np.isfinite(X_lin).all(axis=1) & np.isfinite(X_non).all(axis=1)
        depth_ok = (X_non[:, 2] > 0) & (X_lin[:, 2] > 0)
        good = finite & depth_ok
        X_lin = X_lin[good]
        X_non = X_non[good]

        ax.scatter(
            X_non[:, 0], X_non[:, 2],
            c='royalblue', s=10, alpha=0.85, linewidths=0, zorder=2, label='nonlinear'
        )
        ax.scatter(
            X_lin[:, 0], X_lin[:, 2],
            c='red', marker='x', s=28, linewidths=1.0, zorder=3, label='linear'
        )

        all_pts = np.vstack([X_lin, X_non])
        if all_pts.shape[0] > 0:
            x_lo, x_hi = np.percentile(all_pts[:, 0], [3, 97])
            z_lo, z_hi = np.percentile(all_pts[:, 2], [3, 97])
            pad_x = 0.1 * max(x_hi - x_lo, 1)
            pad_z = 0.1 * max(z_hi - z_lo, 1)
            ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
            ax.set_ylim(z_lo - pad_z, z_hi + pad_z)

        ax.set_title(f'(1,{target})')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_bundle_adjustment_comparison(X_before, X_after, cam_centers_after, cam_rotations_after, out_path):
    finite = np.isfinite(X_before).all(axis=1) & np.isfinite(X_after).all(axis=1)
    Xb = X_before[finite]
    Xa = X_after[finite]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(Xb[:, 0], Xb[:, 2], c='blue', s=6, alpha=0.6, linewidths=0, label='before bund adj', zorder=2)
    ax.scatter(Xa[:, 0], Xa[:, 2], c='red', s=4, alpha=0.7, linewidths=0, label='after bund adj', zorder=3)

    cam_colors = ['green', 'red', 'blue', 'purple', 'orange']
    for i in range(len(cam_centers_after)):
        C = np.asarray(cam_centers_after[i])
        R = np.asarray(cam_rotations_after[i])
        draw_camera_marker(ax, C, R, cam_colors[i % len(cam_colors)], size=0.9)
        ax.annotate(str(i + 1), (C[0], C[2]), fontsize=7, fontweight='bold',
                    ha='center', va='bottom', zorder=6)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-5, 25)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.legend(loc='upper right')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def compute_observation_reproj_error(K, camera_centers, camera_rotations, points3d, obs_cam_idx, obs_pt_idx, obs_uv):
    if obs_uv.shape[0] == 0:
        return 0.0
    errs = []
    for cam_i, pt_i, uv in zip(obs_cam_idx, obs_pt_idx, obs_uv):
        C = camera_centers[int(cam_i)]
        R = camera_rotations[int(cam_i)]
        X = points3d[int(pt_i)]
        X_cam = R @ (X - C)
        proj = K @ X_cam
        uv_hat = proj[:2] / proj[2]
        errs.append(np.linalg.norm(uv_hat - uv))
    return float(np.mean(errs))


def build_tracks_from_correspondences(inlier_corr):
    coord_to_track = {}  
    tracks = []

    def _key(img_id, u, v):
        return (img_id, round(u, 5), round(v, 5))

    for (src, dst) in sorted(inlier_corr.keys()):
        pts1 = inlier_corr[(src, dst)]['pts1']
        pts2 = inlier_corr[(src, dst)]['pts2']
        for i in range(pts1.shape[0]):
            u1, v1 = pts1[i]
            u2, v2 = pts2[i]
            key1 = _key(src, u1, v1)
            key2 = _key(dst, u2, v2)

            tid1 = coord_to_track.get(key1)
            tid2 = coord_to_track.get(key2)

            if tid1 is not None and tracks[tid1] is None:
                tid1 = None
            if tid2 is not None and tracks[tid2] is None:
                tid2 = None

            if tid1 is not None and tid2 is not None:
                if tid1 == tid2:
                    continue
                for img_id, uv in tracks[tid2].items():
                    if img_id not in tracks[tid1]:
                        tracks[tid1][img_id] = uv
                    coord_to_track[_key(img_id, uv[0], uv[1])] = tid1
                tracks[tid2] = None
            elif tid1 is not None:
                tracks[tid1][dst] = (u2, v2)
                coord_to_track[key2] = tid1
            elif tid2 is not None:
                tracks[tid2][src] = (u1, v1)
                coord_to_track[key1] = tid2
            else:
                tid = len(tracks)
                tracks.append({src: (u1, v1), dst: (u2, v2)})
                coord_to_track[key1] = tid
                coord_to_track[key2] = tid

    clean_tracks = []
    new_coord_to_track = {}
    for t in tracks:
        if t is None:
            continue
        new_tid = len(clean_tracks)
        clean_tracks.append(t)
        for img_id, uv in t.items():
            new_coord_to_track[_key(img_id, uv[0], uv[1])] = new_tid
    return clean_tracks, new_coord_to_track



OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'IntermediateOutputImages')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: RANSAC pre-filtering all image pairs")
print("=" * 60)

inlier_correspondences = {}
inlier_indices = {}
for pair in sorted(correspondences.keys()):
    pts_a = correspondences[pair]['pts1']
    pts_b = correspondences[pair]['pts2']
    if pts_a.shape[0] < 8:
        continue
    F_pair, inlier_idx = get_inliers_ransac(pts_a, pts_b, n_iters=5000, threshold=0.05)
    inlier_correspondences[pair] = {
        'pts1': pts_a[inlier_idx],
        'pts2': pts_b[inlier_idx],
        'rgb': correspondences[pair]['rgb'][inlier_idx],
    }
    inlier_indices[pair] = inlier_idx
    print(f"  Pair {pair}: {len(inlier_idx)} inliers / {len(pts_a)} total")

fig5_path = os.path.join(OUT_DIR, 'figure5_ransac_matches.png')
plot_ransac_matches(
    os.path.join(DATA_DIR, '1.png'), os.path.join(DATA_DIR, '2.png'),
    correspondences[(1, 2)]['pts1'], correspondences[(1, 2)]['pts2'],
    inlier_indices[(1, 2)], fig5_path
)
print(f"Saved {fig5_path}")


tracks, coord_to_track = build_tracks_from_correspondences(inlier_correspondences)
track_X = [None] * len(tracks)  # 3D point per track (None = not yet triangulated)
print(f"\nBuilt {len(tracks)} unified tracks across all images")


print("\n" + "=" * 60)
print("Steps 3-5: Initialize from pair (1, 2)")
print("=" * 60)

pair12 = inlier_correspondences[(1, 2)]
pts1_in = pair12['pts1']
pts2_in = pair12['pts2']
print(f"RANSAC (1,2): {pts1_in.shape[0]} inliers")

F_best = estimate_fundamental_matrix(pts1_in, pts2_in)
E = essential_matrix_from_F(F_best, K)

poses = extract_camera_pose(E)

fig6_path = os.path.join(OUT_DIR, 'figure6_pose_disambiguation.png')
plot_pose_disambiguation(poses, K, pts1_in, pts2_in, fig6_path)
print(f"Saved {fig6_path}")

C1 = np.zeros(3)
R1 = np.eye(3)
(C2, R2), X_linear, valid = disambiguate_camera_pose(poses, K, pts1_in, pts2_in)
pts1_valid = pts1_in[valid]
pts2_valid = pts2_in[valid]
print(f"Cheirality: {X_linear.shape[0]} points in front of both cameras")
print(f"  Camera 2 center: {C2}, baseline: {np.linalg.norm(C2):.4f}")
print(f"  Camera 2 R det: {np.linalg.det(R2):.4f}")
print(f"  3D points X range: [{X_linear[:,0].min():.2f}, {X_linear[:,0].max():.2f}]")
print(f"  3D points Z range: [{X_linear[:,2].min():.2f}, {X_linear[:,2].max():.2f}]")
print(f"  3D points Z median: {np.median(X_linear[:,2]):.2f}")

P1 = K @ R1 @ np.hstack([np.eye(3), -C1.reshape(3, 1)])
P2 = K @ R2 @ np.hstack([np.eye(3), -C2.reshape(3, 1)])
print(f"Linear reproj error - cam1: {compute_reproj_error(P1, X_linear, pts1_valid):.4f}, "
      f"cam2: {compute_reproj_error(P2, X_linear, pts2_valid):.4f}")

X_nonlinear = nonlinear_triangulation(K, C1, R1, C2, R2, pts1_valid, pts2_valid, X_linear)
print(f"Nonlinear reproj error - cam1: {compute_reproj_error(P1, X_nonlinear, pts1_valid):.4f}, "
      f"cam2: {compute_reproj_error(P2, X_nonlinear, pts2_valid):.4f}")

fig7_path = os.path.join(OUT_DIR, 'figure7_linear_vs_nonlinear.png')
plot_linear_vs_nonlinear(X_linear, X_nonlinear, C1, R1, C2, R2, fig7_path)
print(f"Saved {fig7_path}")

fig8_path = os.path.join(OUT_DIR, 'figure8_reprojection_comparison.png')
img1_path = os.path.join(DATA_DIR, '1.png')
img2_path = os.path.join(DATA_DIR, '2.png')
plot_reprojection_comparison(img1_path, img2_path, pts1_in, pts2_in,
                             X_linear, X_nonlinear, P1, P2, fig8_path)
print(f"Saved {fig8_path}")

def _tkey(img_id, u, v):
    return (img_id, round(u, 5), round(v, 5))

populated = 0
for i in range(pts1_valid.shape[0]):
    key = _tkey(1, pts1_valid[i, 0], pts1_valid[i, 1])
    tid = coord_to_track.get(key)
    if tid is not None and track_X[tid] is None and np.isfinite(X_nonlinear[i]).all():
        track_X[tid] = X_nonlinear[i].copy()
        populated += 1
print(f"Populated {populated} tracks with initial 3D points")

camera_poses = {1: (C1.copy(), R1.copy()), 2: (C2.copy(), R2.copy())}


pnp_results = []

for new_img_id in [3, 4, 5]:
    print(f"\n{'=' * 60}")
    print(f"Registering camera {new_img_id}")
    print(f"{'=' * 60}")

    X_pnp = []
    x_pnp = []
    pts1_fig9 = []
    pts_target_fig9 = []
    for tid, track in enumerate(tracks):
        if track_X[tid] is None:
            continue
        if new_img_id not in track:
            continue
        X_pnp.append(track_X[tid])
        x_pnp.append(track[new_img_id])
        if 1 in track:
            pts1_fig9.append(track[1])
            pts_target_fig9.append(track[new_img_id])
    X_pnp = np.array(X_pnp) if X_pnp else np.empty((0, 3))
    x_pnp = np.array(x_pnp) if x_pnp else np.empty((0, 2))
    pts1_fig9 = np.array(pts1_fig9) if pts1_fig9 else np.empty((0, 2))
    pts_target_fig9 = np.array(pts_target_fig9) if pts_target_fig9 else np.empty((0, 2))
    print(f"  PnP: {X_pnp.shape[0]} 2D-3D correspondences")

    if X_pnp.shape[0] < 6:
        print(f"  Skipping image {new_img_id}: insufficient correspondences")
        continue

    try:
        C_lin, R_lin, inliers_pnp = pnp_ransac(
            X_pnp, x_pnp, K, n_iters=3000, threshold=15.0
        )
        ransac_msg = f"RANSAC inliers={inliers_pnp.shape[0]}"
    except RuntimeError:
        C_lin, R_lin = linear_pnp(X_pnp, x_pnp, K)
        inliers_pnp = np.arange(X_pnp.shape[0])
        ransac_msg = "RANSAC failed, using all points"

    C_new, R_new = nonlinear_pnp(
        X_pnp[inliers_pnp], x_pnp[inliers_pnp], K, C_lin, R_lin
    )

    err_lin = np.mean(pnp_reprojection_error(X_pnp, x_pnp, K, C_lin, R_lin))
    err_non = np.mean(pnp_reprojection_error(X_pnp, x_pnp, K, C_new, R_new))
    print(f"  {ransac_msg}, "
          f"linear err={err_lin:.3f}px, nonlinear err={err_non:.3f}px")

    pnp_results.append({
        'target': new_img_id,
        'pts1': pts1_fig9,
        'pts_target': pts_target_fig9,
        'linear_pose': (C_lin, R_lin),
        'nonlinear_pose': (C_new, R_new),
    })

    camera_poses[new_img_id] = (C_new.copy(), R_new.copy())

    C_ref, R_ref = camera_poses[1]
    new_point_count = 0

    tids_to_tri = []
    pts_ref = []
    pts_new = []
    for tid, track in enumerate(tracks):
        if track_X[tid] is not None:
            continue
        if 1 in track and new_img_id in track:
            tids_to_tri.append(tid)
            pts_ref.append(track[1])
            pts_new.append(track[new_img_id])

    if tids_to_tri:
        pts_ref = np.array(pts_ref)
        pts_new = np.array(pts_new)

        X_lin_new = linear_triangulation(K, C_ref, R_ref, C_new, R_new, pts_ref, pts_new)
        X_nlin_new = nonlinear_triangulation(K, C_ref, R_ref, C_new, R_new, pts_ref, pts_new, X_lin_new)

        cond_ref = R_ref[2, :] @ (X_nlin_new - C_ref).T > 0
        cond_new = R_new[2, :] @ (X_nlin_new - C_new).T > 0
        finite = np.isfinite(X_nlin_new).all(axis=1)
        good = cond_ref & cond_new & finite

        for j, tid in enumerate(tids_to_tri):
            if good[j]:
                track_X[tid] = X_nlin_new[j].copy()
                new_point_count += 1

    print(f"  Triangulated {new_point_count} new 3D points")
    total_3d = sum(1 for x in track_X if x is not None)
    print(f"  Total 3D points in cloud: {total_3d}")

    registered_cams = sorted(camera_poses.keys())
    cam_id_to_idx = {cid: i for i, cid in enumerate(registered_cams)}

    ba_tids = []
    ba_X = []
    ba_obs_cam = []
    ba_obs_pt = []
    ba_obs_uv = []

    for tid, track in enumerate(tracks):
        if track_X[tid] is None:
            continue
        if not np.isfinite(track_X[tid]).all():
            continue
        pt_idx = len(ba_tids)
        ba_tids.append(tid)
        ba_X.append(track_X[tid])
        for cid in registered_cams:
            if cid in track:
                ba_obs_cam.append(cam_id_to_idx[cid])
                ba_obs_pt.append(pt_idx)
                ba_obs_uv.append(track[cid])

    ba_X = np.array(ba_X, dtype=float)
    ba_obs_cam = np.array(ba_obs_cam, dtype=int)
    ba_obs_pt = np.array(ba_obs_pt, dtype=int)
    ba_obs_uv = np.array(ba_obs_uv, dtype=float)

    cam_centers = [camera_poses[cid][0] for cid in registered_cams]
    cam_rotations = [camera_poses[cid][1] for cid in registered_cams]

    n_cams = len(registered_cams)
    V = build_visibility_matrix(n_cams, ba_X.shape[0], ba_obs_cam, ba_obs_pt)

    print(f"  BA: {ba_X.shape[0]} points, {ba_obs_uv.shape[0]} observations, "
          f"{n_cams} cameras ({registered_cams})")

    X_before_ba = ba_X.copy()
    cams_before_ba = [c.copy() for c in cam_centers]

    err_before = compute_observation_reproj_error(
        K, cam_centers, cam_rotations, ba_X, ba_obs_cam, ba_obs_pt, ba_obs_uv
    )

    C_ba, R_ba, X_ba, ba_result = bundle_adjustment(
        K, cam_centers, cam_rotations, ba_X,
        ba_obs_cam, ba_obs_pt, ba_obs_uv,
        visibility_matrix=V,
        fixed_camera_indices=(0,),
        max_nfev=50000, loss="linear", f_scale=1.0,
    )

    err_after = compute_observation_reproj_error(
        K, C_ba, R_ba, X_ba, ba_obs_cam, ba_obs_pt, ba_obs_uv
    )
    print(f"  BA: nfev={ba_result.nfev}, success={ba_result.success}, "
          f"reproj before={err_before:.3f}px, after={err_after:.3f}px")
    print(f"  BA: message={ba_result.message}, optimality={ba_result.optimality:.6e}, "
          f"cost={ba_result.cost:.2f}")


    for i, cid in enumerate(registered_cams):
        camera_poses[cid] = (np.array(C_ba[i]), np.array(R_ba[i]))
    for j, tid in enumerate(ba_tids):
        track_X[tid] = X_ba[j].copy()

    if new_img_id == 3:
        fig10_path = os.path.join(OUT_DIR, 'figure10_bundle_adjustment.png')
        plot_bundle_adjustment_comparison(
            X_before_ba, X_ba, list(C_ba), list(R_ba), fig10_path
        )
        print(f"  Saved {fig10_path}")

    if new_img_id == 5:
        fig11_path = os.path.join(OUT_DIR, 'figure11_bundle_adjustment.png')
        plot_bundle_adjustment_comparison(
            X_before_ba, X_ba, list(C_ba), list(R_ba), fig11_path
        )
        print(f"  Saved {fig11_path}")

fig9_path = os.path.join(OUT_DIR, 'figure9_pnp_linear_vs_nonlinear.png')
plot_pnp_comparison(pnp_results, K, fig9_path)
print(f"\nSaved {fig9_path}")

print("\n" + "=" * 60)
print("SfM Pipeline Complete")
print("=" * 60)
total_points = sum(1 for x in track_X if x is not None)
print(f"Total 3D points: {total_points}")
print(f"Registered cameras: {sorted(camera_poses.keys())}")
for cid in sorted(camera_poses.keys()):
    C, R = camera_poses[cid]
    print(f"  Camera {cid}: C = [{C[0]:.4f}, {C[1]:.4f}, {C[2]:.4f}]")
