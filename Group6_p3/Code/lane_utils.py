import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, splev, splprep


class Lane:
    def __init__(self, points=None, invalid_value=-2.0, metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(
            points[:, 1], points[:, 0], k=min(3, len(points) - 1)
        )
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01
        self.metadata = metadata or {}

    def __repr__(self):
        return "[Lane]\n" + str(self.points) + "\n[/Lane]"

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)
        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration


def interp(points, n=5):
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=5, k=min(3, len(points) - 1))
    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    output = np.array(splev(u, tck)).T
    return output


def sample_lane(points, sample_ys, img_w):
    """
    Sample lane points on the horizontal grids.
    Points must be sorted in descending y order.
    Duplicate y values are removed to ensure strict ordering.
    """
    points = np.array([points[0::2], points[1::2]]).transpose(1, 0)

    _, unique_idx = np.unique(-points[:, 1], return_index=True)
    points = points[unique_idx]

    # Sort descending by y
    sort_idx = np.argsort(-points[:, 1])
    points = points[sort_idx]

    # Final strict-order check — skip lane if still not valid
    if len(points) < 2:
        return np.zeros(0), np.zeros(0)
    if not np.all(points[1:, 1] < points[:-1, 1]):
        # silently skip invalid lane instead of raising
        return np.zeros(0), np.zeros(0)

    x, y = points[:, 0], points[:, 1]

    interp_fn = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
    domain_min_y = y.min()
    domain_max_y = y.max()
    mask_inside_domain = (sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)
    sample_ys_inside_domain = sample_ys[mask_inside_domain]
    if len(sample_ys_inside_domain) == 0:
        return np.zeros(0), np.zeros(0)
    interp_xs = interp_fn(sample_ys_inside_domain)

    two_closest_points = points[:2]
    extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
    extrap_ys = sample_ys[sample_ys > domain_max_y]
    extrap_xs = np.polyval(extrap, extrap_ys)
    all_xs = np.hstack((extrap_xs, interp_xs))

    inside_mask = (all_xs >= 0) & (all_xs < img_w)
    xs_inside_image = all_xs[inside_mask]
    xs_outside_image = all_xs[~inside_mask]
    return xs_outside_image, xs_inside_image
