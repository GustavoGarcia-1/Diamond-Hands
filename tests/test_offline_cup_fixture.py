import numpy as np,mujoco
from g1_receiver.offline_cup_fixture import projected_cup_seed
from g1_receiver.scrimmage_throw import ScrimmagePass

def test_two_distinct_calibrated_palm_planes_are_simultaneously_tangent():
 N=np.array([[.3,-.4,np.sqrt(.75)],[.2,.5,np.sqrt(.71)]])
 faces=np.array([[6.2,.04,1.],[6.22,-.10,.98]])
 cal={'poses':[{'time_s':5.2,'hands':[{'face_world_m':f.tolist(),'normal_world':n.tolist()}for f,n in zip(faces,N)]}]}
 profile={'play':{'quarterback_origin_m':[0.,-1.8288,1.4],'quarterback_level_ball':False}}
 C,normal,metadata=projected_cup_seed(profile,.8,cal)
 plan=ScrimmagePass((0.,-1.8288,1.4),tuple(C),.8,horizontal_distance_only=True).plan(4.4)
 R=np.empty(9);mujoco.mju_quat2Mat(R,np.array(plan.release_quaternion_wxyz));R=R.reshape(3,3)
 radii=np.linalg.norm((N@R)*[.142,.085,.085],axis=1)
 assert np.allclose(np.einsum('ij,ij->i',C-faces,N),radii,atol=1e-9)
 assert abs(np.linalg.norm(normal)-1.)<1e-12
 assert max(abs(np.array(metadata['plane_gap_m'])))<1e-9

def test_first_palm_fixture_stays_on_declared_finite_patch_offset():
 from g1_receiver.offline_cup_fixture import one_palm_seed
 profile={'play':{'quarterback_origin_m':[0.,-1.8288,1.4]}}
 cal={'poses':[{'time_s':5.2,'hands':[{'face_world_m':[6.,0.,1.],'normal_world':[0.,0.,1.],'finger_axis_world':[1.,0.,0.]},{'face_world_m':[6.,-.15,1.],'normal_world':[0.,0.,1.],'finger_axis_world':[1.,0.,0.]}]}]}
 center,normal,metadata=one_palm_seed(profile,.8,cal,'left',.02,-.015)
 assert np.allclose(metadata['contact_footpoint_m'],[5.985,.02,1.],atol=1e-9)
 assert center[2]>1.
 assert np.array_equal(normal,[0.,0.,1.])
