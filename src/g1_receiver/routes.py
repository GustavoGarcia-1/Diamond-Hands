"""Route commands in a declared local field frame; no simulator state access."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Route:
    name: str
    waypoints: tuple[tuple[float, float], ...]
    nominal_speed_mps: float = 0.6
    corridor_m: float = 0.5

    def __post_init__(self):
        if len(self.waypoints) < 2:
            raise ValueError('Route needs at least two waypoints')
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in self.waypoints):
            raise ValueError('Waypoints must be finite xy pairs')
        if not math.isfinite(self.nominal_speed_mps) or self.nominal_speed_mps <= 0:
            raise ValueError('Speed must be positive and finite')
        if not math.isfinite(self.corridor_m) or self.corridor_m <= 0:
            raise ValueError('Corridor must be positive and finite')
        if any(math.dist(a, b) < 1e-6 for a, b in zip(self.waypoints, self.waypoints[1:])):
            raise ValueError('Duplicate adjacent waypoints')

    @property
    def length_m(self):
        return sum(math.dist(a, b) for a, b in zip(self.waypoints, self.waypoints[1:]))

    def project(self, estimated_xy, min_progress_m=0.):
        """Return distance to path and arc-length progress from a measured estimate."""
        if len(estimated_xy) != 2 or not all(math.isfinite(v) for v in estimated_xy):
            raise ValueError('Estimated position must be a finite xy pair')
        if not math.isfinite(min_progress_m) or not 0 <= min_progress_m <= self.length_m:
            raise ValueError('Invalid minimum route progress')
        best = (float('inf'), self.length_m)
        traveled = 0.
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            if traveled + length < min_progress_m:
                traveled += length
                continue
            lower = max(0., (min_progress_m-traveled)/length)
            t = max(lower, min(1., ((estimated_xy[0]-a[0])*dx + (estimated_xy[1]-a[1])*dy)/length**2))
            d = math.dist(estimated_xy, (a[0]+t*dx, a[1]+t*dy))
            if d < best[0]:
                best = (d, traveled+t*length)
            traveled += length
        return best

    def point_at(self, progress_m):
        if not math.isfinite(progress_m):
            raise ValueError('Progress must be finite')
        remaining = max(0., progress_m)
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            length = math.dist(a, b)
            if remaining <= length:
                t = remaining/length
                return (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
            remaining -= length
        return self.waypoints[-1]

    def velocity_command(self, estimated_xy, lookahead_m=0.35, min_progress_m=0.):
        """Bounded baseline path follower. Requires estimated, not privileged, pose."""
        if not math.isfinite(lookahead_m) or lookahead_m <= 0:
            raise ValueError('Lookahead must be positive and finite')
        _, progress = self.project(estimated_xy, min_progress_m)
        target = self.point_at(progress+lookahead_m)
        delta = (target[0]-estimated_xy[0], target[1]-estimated_xy[1])
        distance = math.hypot(*delta)
        speed = min(self.nominal_speed_mps, distance*2.)
        return tuple(speed*d/max(distance, 1e-9) for d in delta)


def make_route(name='slant', scale=1., speed_mps=0.6, mirror=False):
    """Provisional short indoor commands, not validated locomotion capabilities."""
    shapes = {
        'slant': ((0., 0.), (1., 0.), (2.5, 1.)),
        'hitch': ((0., 0.), (2., 0.), (1.5, 0.)),
        'out': ((0., 0.), (1.5, 0.), (1.5, 1.5)),
        'post': ((0., 0.), (2., 0.), (4., 1.5)),
        'go': ((0., 0.), (4., 0.)),
    }
    if name not in shapes or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Unknown route or invalid scale')
    return Route(name, tuple((x*scale, y*scale*(-1 if mirror else 1)) for x, y in shapes[name]), speed_mps)


@dataclass
class RouteFollower:
    """Stateful segment follower; overlapping return legs cannot skip the turn."""
    route: Route
    lookahead_m: float = .35
    arrival_tolerance_m: float = .05
    next_waypoint: int = 1

    def command(self, estimated_xy):
        self.route.project(estimated_xy)  # validate finite input
        if self.next_waypoint >= len(self.route.waypoints):
            return (0., 0.)
        a, b = self.route.waypoints[self.next_waypoint-1:self.next_waypoint+1]
        length = math.dist(a, b)
        tolerance = min(self.arrival_tolerance_m, length*.1)
        if math.dist(estimated_xy, b) <= tolerance:
            self.next_waypoint += 1
            return self.command(estimated_xy)
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0., min(1., ((estimated_xy[0]-a[0])*dx+(estimated_xy[1]-a[1])*dy)/length**2))
        target_t = min(1., t+self.lookahead_m/length)
        delta = (a[0]+target_t*dx-estimated_xy[0], a[1]+target_t*dy-estimated_xy[1])
        norm = math.hypot(*delta)
        speed = min(self.route.nominal_speed_mps, 2*norm)
        return tuple(speed*d/max(norm, 1e-9) for d in delta)
