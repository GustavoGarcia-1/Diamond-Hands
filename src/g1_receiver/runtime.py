"""Bundled public G1 assets and the read-only WBC observation contract."""
from pathlib import Path

from ._vendor.g1_runtime.pool.contracts import RobotObservation, SkillGoal

DEFAULT_RUNTIME_ROOT = Path(__file__).resolve().parent / "_vendor" / "g1_runtime"
DEFAULT_ROBOT_XML = DEFAULT_RUNTIME_ROOT / "assets" / "robot" / "g1_body.xml"

__all__ = ["DEFAULT_ROBOT_XML", "DEFAULT_RUNTIME_ROOT", "RobotObservation", "SkillGoal"]
