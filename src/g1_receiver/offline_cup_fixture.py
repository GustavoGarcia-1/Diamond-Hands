"""Offline fixed-fixture geometry from serialized calibration, never runtime feedback."""
import numpy as np,mujoco
from .scrimmage_throw import ScrimmagePass

def projected_cup_seed(profile,flight,calibration):
    arrival=float(profile.get('offline_fixture_arrival_s',profile['play'].get('release_s',5.2-flight)+profile['play'].get('flight_s',flight)));row=min(calibration['poses'],key=lambda r:abs(r['time_s']-arrival));faces=np.array([h['face_world_m']for h in row['hands']],float);N=np.array([h['normal_world']for h in row['hands']],float)
    if faces.shape!=(2,3)or N.shape!=(2,3)or not np.isfinite([faces,N]).all()or not np.allclose(np.linalg.norm(N,axis=1),1.,atol=1e-5)or np.linalg.svd(N,compute_uv=False)[-1]<.02:raise ValueError('two finite independent calibrated unit palm normals required')
    point=faces.mean(0)+.12*N.mean(0);origin=profile['play']['quarterback_origin_m'];radii=None
    for iteration in range(60):
        plan=ScrimmagePass(tuple(origin),tuple(point),flight,align_ball_to_launch=not profile['play'].get('quarterback_level_ball',False),horizontal_distance_only=True).plan(arrival-flight);R=np.empty(9);mujoco.mju_quat2Mat(R,np.array(plan.release_quaternion_wxyz));R=R.reshape(3,3);A=R@np.diag(np.array([.142,.085,.085])**2)@R.T;vectors=(A@N.T).T;radii=np.sqrt(np.sum(N*vectors,axis=1));vectors/=radii[:,None];mean=(faces+vectors).mean(0);rhs=radii+np.sum(N*faces,axis=1)-N@mean;new=mean+N.T@np.linalg.solve(N@N.T,rhs)
        if np.linalg.norm(new-point)<1e-10:point=new;break
        point=new
    else:raise ValueError('declared ellipsoid launch orientation did not converge')
    normal=N.mean(0);normal/=np.linalg.norm(normal);gaps=np.sum((point-faces)*N,axis=1)-radii
    metadata=dict(face_world_m=faces.tolist(),normal_world=N.tolist(),support_radius_m=radii.tolist(),support_vectors_m=vectors.tolist(),seed_m=point.tolist(),plane_gap_m=gaps.tolist(),calibration_time_s=row['time_s'],method='full ellipsoid support VECTOR mean followed by BOTH palm-plane projection; offline fixture calibration only')
    if all('finger_axis_world'in h for h in row['hands']):
        F=np.array([h['finger_axis_world']for h in row['hands']]);width=np.cross(N,F);foot=point-vectors;offset=foot-faces;w=np.sum(offset*width,1);longitudinal=np.sum(offset*F,1);metadata.update(contact_footpoint_m=foot.tolist(),width_offset_m=w.tolist(),finger_axis_offset_m=longitudinal.tolist(),rough_finite_patch_screen_only=bool(np.max(np.abs(w))<=.04 and longitudinal.min()>=-.05225 and longitudinal.max()<=.02975),skin_contact_still_requires_actual_geom_and_contact_audit=True)
    return point,normal,metadata

def one_palm_seed(profile,flight,calibration,side='left',width_offset_m=0.,finger_offset_m=0.):
    """Finite centered first-palm fixture; eventual bilateral control is still scored."""
    if side not in ('left','right'):raise ValueError('explicit first palm required')
    arrival=float(profile.get('offline_fixture_arrival_s',profile['play'].get('release_s',5.2-flight)+profile['play'].get('flight_s',flight)));row=min(calibration['poses'],key=lambda r:abs(r['time_s']-arrival));h=row['hands'][0 if side=='left'else 1];face=np.array(h['face_world_m']);n=np.array(h['normal_world']);F=np.array(h['finger_axis_world']);W=np.cross(n,F);face=face+float(width_offset_m)*W+float(finger_offset_m)*F;point=face+.1*n
    for _ in range(60):
        plan=ScrimmagePass(tuple(profile['play']['quarterback_origin_m']),tuple(point),flight,align_ball_to_launch=not profile['play'].get('quarterback_level_ball',False),horizontal_distance_only=True).plan(arrival-flight);R=np.empty(9);mujoco.mju_quat2Mat(R,np.array(plan.release_quaternion_wxyz));R=R.reshape(3,3);A=R@np.diag(np.array([.142,.085,.085])**2)@R.T;vector=A@n/np.sqrt(n@A@n);new=face+vector
        if np.linalg.norm(new-point)<1e-10:point=new;break
        point=new
    else:raise ValueError('first-palm declared fixture did not converge')
    return point,n,dict(method='one finite centered first-palm ellipsoid tangent; other hand open/clear then eventual bilateral control required',first_palm=side,width_offset_m=float(width_offset_m),finger_offset_m=float(finger_offset_m),face_world_m=face.tolist(),normal_world=n.tolist(),seed_m=point.tolist(),contact_footpoint_m=(point-vector).tolist(),calibration_time_s=row['time_s'])
