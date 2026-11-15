import math
from dataclasses import replace
import pytest
from g1_receiver.routes import Route, RouteFollower, make_route
from g1_receiver.evaluation import EvaluationFrame, evaluate_episode, wilson_interval


def test_route_projection_and_bounded_stop():
    route = make_route('out', speed_mps=.5)
    assert route.project((1.5, .8)) == pytest.approx((0., 2.3))
    assert math.hypot(*route.velocity_command((0., 0.))) <= .5
    assert route.velocity_command(route.waypoints[-1]) == (0., 0.)
    assert make_route('out', mirror=True).waypoints[-1][1] == -1.5


def valid_episode():
    return [EvaluationFrame(i*.05, (min(i/40, 1.), 0.), True, False, i == 0, False, i//15) for i in range(61)]


def test_hitch_progress_disambiguates_return_leg():
    route = make_route('hitch')
    assert route.project((1.8, 0.), min_progress_m=2.) == pytest.approx((0., 2.2))
    assert route.velocity_command((1.8, 0.), min_progress_m=2.)[0] < 0


def test_hitch_follower_visits_turn_before_returning():
    route = make_route('hitch', scale=.45, speed_mps=.3)
    follower = RouteFollower(route)
    xy = (0., 0.)
    max_x = 0.
    for _ in range(1000):
        velocity = follower.command(xy)
        xy = tuple(p+v*.02 for p, v in zip(xy, velocity))
        max_x = max(max_x, xy[0])
    assert max_x >= .85
    assert math.dist(xy, route.waypoints[-1]) < .05
    assert follower.next_waypoint == len(route.waypoints)


def test_contact_retention_route_and_steps_required():
    route = Route('test', ((0., 0.), (1., 0.)))
    frames = valid_episode()
    assert evaluate_episode(frames, route)['success']
    assert not evaluate_episode([replace(f, ball_robot_contact=False) for f in frames], route)['success']
    assert not evaluate_episode([replace(f, completed_steps=0) for f in frames], route)['success']
    assert not evaluate_episode([replace(f, receiver_xy=(0., 0.)) for f in frames], route)['success']
    assert not evaluate_episode([replace(f, receiver_xy=(1., 0.)) for f in frames], route)['success']
    assert not evaluate_episode([replace(f, invalid_state_write=True) for f in frames], route)['success']
    frames[30] = replace(frames[30], ball_ground_contact=True)
    assert not evaluate_episode(frames, route)['success']


def test_fall_or_sparse_logs_cannot_pass():
    route = Route('test', ((0., 0.), (1., 0.)))
    frames = valid_episode()
    frames[-1] = replace(frames[-1], fallen=True)
    assert not evaluate_episode(frames, route)['success']
    assert not evaluate_episode(valid_episode()[::10], route)['success']
    with pytest.raises(ValueError):
        evaluate_episode(list(reversed(frames)), route)
    frames = valid_episode()
    frames[-1] = replace(frames[-1], ball_ground_contact=True)
    assert not evaluate_episode(frames, route)['success']
    with pytest.raises(ValueError):
        evaluate_episode([replace(f, completed_steps=float('nan')) for f in frames], route)


def test_wilson_reports_uncertainty_at_extremes():
    lo, hi = wilson_interval(100, 100)
    assert .96 < lo < 1. and hi == pytest.approx(1.)
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0.) and 0 < hi < .04
