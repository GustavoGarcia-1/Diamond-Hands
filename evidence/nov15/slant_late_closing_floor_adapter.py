"""Isolated Slant physical rollout. Reports are independent evaluation only."""
from pathlib import Path
import json
import tempfile
import numpy as np


def run_training_rollout(config, audit, output=None):
    """Run frozen aimed Slant with fourteen arm actions and fixed native grip.

    Each process runs one rollout at a time. The environment must never read the
    returned physical report for observations, reward, termination or selection.
    """
    import model_comparison
    import native_runner
    from encoder_grip_compliance import EncoderGripCompliance
    from g1_receiver.lower_cup_grip import NativeCupGrip, legacy_targets
    from g1_receiver.hand_route_plant import HandRoutePlant

    temporary = tempfile.TemporaryDirectory(prefix='slant-precontact-') if output is None else None
    directory = Path(temporary.name)/'rollout' if temporary else Path(output)
    directory.parent.mkdir(parents=True, exist_ok=True)
    original_observe = HandRoutePlant.observe
    original_step = HandRoutePlant.step
    original = NativeCupGrip.targets
    instances = []
    pending = {}
    observation_diagnostics = []
    observations = []
    native_rows = []
    floor_rows = []
    latched_floor = None
    closing_floor_enabled = bool(config.get('late_closing_floor_enabled', False))
    profile_settings = json.loads(Path(config['profile']).read_text())['play']
    preparation_goal = legacy_targets(float(profile_settings['closure']))
    preparation_delay = float(profile_settings['grip_actuation_delay_s'])
    fingers = np.array([0,1,2,3,7,8,9,10])
    thumbs = np.array([4,5,6,11,12,13])

    def nominal(self, now, closure, release_at, gate, encoders):
        raw = original(self, now, closure, release_at, gate, encoders)
        if not hasattr(self, '_fixed_training_compliance'):
            self._fixed_training_compliance = EncoderGripCompliance(self.calibration.ranges)
            instances.append(self._fixed_training_compliance)
        prepared = raw.copy()
        if gate is not None and now >= gate-1e-8 and now < gate+preparation_delay-1e-8:
            prepared[fingers] = preparation_goal[fingers]
        assert np.array_equal(prepared[thumbs],raw[thumbs])
        pending.clear()
        pending.update(regulator=self._fixed_training_compliance, now=now,
                       gate=gate, encoders=np.asarray(encoders).copy(), nominal=prepared.copy(), raw=raw.copy())
        return prepared

    class FixedNativeBoundary:
        def bind(self, *args, **kwargs):
            return audit.bind(*args, **kwargs)

        def __call__(self, *args):
            adjusted, native = audit(*args)
            if 5.8 <= float(args[0].timestamp) <= 6.5:
                pose, frame, survey = args[1:4]
                ball = None if frame is None else audit.sensors.detector.detect(frame, survey)
                observation_diagnostics.append(dict(time_s=float(args[0].timestamp), enabled=bool(args[7]),
                    capture_s=None if frame is None else frame.timestamp_s,
                    pose_s=None if pose is None else pose.timestamp_s,
                    pose_z=None if pose is None else pose.z_m,
                    pose_z_std=None if pose is None else pose.z_std_m,
                    pose_covariance=None if pose is None else np.asarray(pose.covariance).tolist(),
                    ball_detected=ball is not None, sensor_valid=bool(audit.sensors.valid),
                    sensor_sigma_m=np.asarray(audit.sensors.sigma).tolist(),
                    intercept_valid=bool(audit.intercept_valid)))
            nominal_request = np.asarray(args[5])
            if not np.array_equal(native, nominal_request):
                raise ValueError('Arm-only audit altered fixed native requests')
            if not np.array_equal(nominal_request, pending['nominal']):
                raise ValueError('Native callback ordering mismatch')
            return adjusted, native

    def regulated_step(plant, command, *args, **kwargs):
        nonlocal latched_floor
        native = np.asarray(kwargs['hand_target_rad'])
        if not np.array_equal(native, pending['nominal']):
            raise ValueError('Post callback native request changed')
        final = pending['regulator'].apply(pending['now'], native,
                                           pending['encoders'], pending['gate'])
        unmodified_final = final.copy()
        floor_active = bool(closing_floor_enabled and pending['gate'] is not None and pending['now'] >= pending['gate'] + 1.30 - 1e-8)
        if floor_active:
            if latched_floor is None:
                latched_floor = final[fingers].copy()
            final = final.copy()
            final[fingers] = np.maximum(final[fingers], latched_floor)
        assert np.all(final[fingers] >= unmodified_final[fingers])
        floor_rows.append(dict(time_s=float(pending['now']), gate_s=pending['gate'], active=floor_active, unmodified_requested=unmodified_final.tolist(), final_requested=final.tolist(), latched_floor=None if latched_floor is None else latched_floor.tolist()))
        if not np.array_equal(final[[4,5,6,11,12,13]], native[[4,5,6,11,12,13]]):
            raise ValueError('Thumb requests changed')
        native_rows.append(np.r_[pending['now'],np.nan if pending['gate'] is None else pending['gate'],float(pending['regulator'].rows[-1]['blend']),pending['encoders'],pending['raw'],final])
        kwargs['hand_target_rad'] = final
        return original_step(plant, command, *args, **kwargs)

    def observed(plant):
        values = original_observe(plant)
        observations.append((plant.timestamp_s, *(np.asarray(v).copy() for v in values)))
        return values

    for name in ['EVAL_STATES' , 'EVAL_CONTACTS', 'EVAL_SELF_CONTACTS', 'EVAL_ARM_TARGETS']:
        getattr(native_runner, name).clear()
    NativeCupGrip.targets = nominal
    HandRoutePlant.step = regulated_step
    HandRoutePlant.observe = observed
    try:
        model_comparison.main([
            '--source', str(config['source']),
            '--amp-source', str(config['amp_source']),
            '--profile', str(config['profile']),
            '--output', str(directory), '--method', 'current',
            '--contact-tau', '.0025',
        ], residual_callback=FixedNativeBoundary())
        report = json.loads((directory/'report.json').read_text())
        if output is not None:
            np.savez_compressed(directory/'native-compliance.npz',samples=np.asarray(native_rows),fingers=fingers,unchanged_thumbs=thumbs,columns=np.array(['time','visual_gate','active','native_encoders14','nominal_requested14','final_requested14']))
            np.savez_compressed(directory/'robot-observation.npz',timestamp_s=np.asarray([r[0]for r in observations]),joint_position=np.asarray([r[1]for r in observations]),joint_velocity=np.asarray([r[2]for r in observations]),imu_quat_wxyz=np.asarray([r[3]for r in observations]),imu_gyro=np.asarray([r[4]for r in observations]),joint_names=np.asarray(__import__('g1_receiver._vendor.g1_runtime.pool.wbc',fromlist=['BODY_NAMES']).BODY_NAMES))
            (directory/'late-closing-floor.json').write_text(json.dumps(dict(enabled=closing_floor_enabled, delay_from_visual_gate_s=1.30, floor_rows=floor_rows), indent=2)+'\n')
            (directory/'observation-diagnostics.json').write_text(json.dumps(observation_diagnostics, indent=2)+'\n')
            (directory/'fixed-native-regulation.json').write_text(json.dumps(instances[0].rows, indent=2)+'\n')
        return report
    finally:
        NativeCupGrip.targets = original
        HandRoutePlant.step = original_step
        HandRoutePlant.observe = original_observe
        if temporary is not None:
            temporary.cleanup()
