"""Repeat the route-only walk audit with a tighter vision-guided final approach."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import audit_receiver_all_routes as audit
from g1_receiver.routes import RouteFollower

TERMINAL_ARRIVAL_CAP_M = .025
MINIMUM_NONZERO_DRIVE_MPS = .055


class TerminalCapFollower(RouteFollower):
    """Keep the stock turn regions and tighten only the terminal approach."""

    def __init__(self, route, *, terminal_cap_m: float):
        super().__init__(route)
        self.terminal_cap_m = terminal_cap_m

    def command(self, estimated_xy):
        if self.next_waypoint == len(self.route.waypoints) - 1:
            self.arrival_tolerance_m = self.terminal_cap_m
        return super().command(estimated_xy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    plan_path = audit.ROOT / "reports" / audit.PLAN_NAME
    plan = audit.read_json(plan_path)
    cases = [(name, False) for name in plan["routes"]] + [(name, True) for name in plan["mirrored_routes"]]
    factory = partial(TerminalCapFollower, terminal_cap_m=TERMINAL_ARRIVAL_CAP_M)
    plant = audit.SensorPlant(audit.DEFAULT_RUNTIME_ROOT)
    try:
        results = {}
        # Keep the stock runner and exact evaluator untouched. Substitute only
        # its local follower factory during these development rollouts.
        with patch.object(audit, "RouteFollower", factory):
            for name, mirror in cases:
                route = audit.make_route(name, scale=plan["scale"], speed_mps=plan["command_speed_mps"], mirror=mirror)
                case_id = f"{name}_mirror" if mirror else name
                result = audit.run_route(plant, route, plan, MINIMUM_NONZERO_DRIVE_MPS)
                results[case_id] = result
                print(case_id, result["passed"], round(result["endpoint_error_m"], 4), flush=True)
    finally:
        plant.close()

    report = {
        "evidence_class": "no_throw_rendered_vision_wbc_terminal_cap_development",
        "plan_sha256": audit.digest(plan_path),
        "correction_script_sha256": audit.digest(Path(__file__)),
        "audit_script_sha256": audit.digest(Path(audit.__file__)),
        "source_sha256": audit._source_snapshot(None, audit.DEFAULT_RUNTIME_ROOT),
        "controller": {
            "terminal_arrival_cap_m": TERMINAL_ARRIVAL_CAP_M,
            "minimum_nonzero_command_mps": MINIMUM_NONZERO_DRIVE_MPS,
            "existing_turn_arrival_cap_m": RouteFollower.__dataclass_fields__["arrival_tolerance_m"].default,
        },
        "configuration": plan,
        "passed": sum(row["passed"] for row in results.values()),
        "rollouts": len(results),
        "results": results,
        "limitations": [
            "The terminal approach was developed on these route fixtures. Results are development evidence.",
            "The physical thresholds, route shapes, nominal speed, horizon and camera are shared with the stock comparison.",
            "Bare-wrist walking with no thrown ball. The loaded hand configuration needs its own evaluation.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
