import numpy as np
from g1_receiver.ellipsoid_perception import fit_visible_ellipsoid
from g1_receiver.ellipsoid_perception import EllipsoidBallDetector
from g1_receiver.perception import CameraIntrinsics, RGBDFrame, RigidPose, ColorBallDetector


def test_partial_visible_surface_recovers_center_without_truth_input():
    rng = np.random.default_rng(173)
    for major in (np.array([1.,0.,0.]), np.array([.3,.6,.74161985])):
        major /= np.linalg.norm(major)
        tangent = np.cross(major, [0.,0.,1.])
        tangent /= np.linalg.norm(tangent)
        rotation = np.c_[major, tangent, np.cross(major, tangent)]
        directions = rng.normal(size=(2500,3))
        directions /= np.linalg.norm(directions, axis=1)[:,None]
        surface = (directions*np.array([.142,.085,.085]))@rotation.T
        center = np.array([.2,-.1,2.5])
        # Visible front cap, including surface normals facing the camera.
        normals = (directions/np.array([.142,.085,.085]))@rotation.T
        visible = np.einsum('ij,ij->i', normals, surface+center) < 0
        points = surface[visible]+center+rng.normal(0,.0002,size=(visible.sum(),3))
        fit = fit_visible_ellipsoid(points, center+np.array([.008,-.005,-.025]))
        assert fit is not None
        assert np.linalg.norm(fit.center_m-center) < .002
        assert abs(fit.major_axis@major) > .995


def test_missing_degenerate_or_invalid_surface_is_not_a_fit():
    assert fit_visible_ellipsoid(np.zeros((12,3)), np.array([0.,0.,2.])) is None
    assert fit_visible_ellipsoid(np.full((60,3),np.nan), np.array([0.,0.,2.])) is None
    assert fit_visible_ellipsoid(np.ones((60,3)), np.zeros(3)) is None
    assert fit_visible_ellipsoid(np.tile([0.,0.,2.],(60,1)), np.array([0.,0.,2.1])) is None


def test_registered_image_fit_preserves_uncertainty_floor():
    intrinsics = CameraIntrinsics(200.,200.,49.5,49.5)
    y,x = np.mgrid[:100,:100]
    rays = np.stack([(x-intrinsics.cx)/intrinsics.fx,(y-intrinsics.cy)/intrinsics.fy,np.ones_like(x)],axis=-1)
    center = np.array([.02,-.015,2.])
    axis = np.array([.8,.4,.4472135955])
    axis /= np.linalg.norm(axis)
    shape = np.eye(3)/.085**2+(1/.142**2-1/.085**2)*np.outer(axis,axis)
    aa = np.einsum('...i,ij,...j->...',rays,shape,rays)
    bb = -2*np.einsum('...i,ij,j->...',rays,shape,center)
    cc = center@shape@center-1
    discriminant = bb*bb-4*aa*cc
    visible = discriminant > 0
    depth = np.full((100,100),np.nan)
    depth[visible] = (-bb[visible]-np.sqrt(discriminant[visible]))/(2*aa[visible])
    rgb = np.zeros((100,100,3),np.uint8)
    rgb[visible] = [180,110,50]
    frame = RGBDFrame(1.,'camera',rgb,depth,intrinsics)
    pose = RigidPose(1.,'camera','field',np.eye(3),np.zeros(3),'fixed_calibration')
    old = ColorBallDetector(ball_center_offset_m=.075).detect(frame,pose)
    new = EllipsoidBallDetector(ball_center_offset_m=.075).detect(frame,pose)
    assert old is not None and new is not None
    assert np.linalg.norm(new.position_m-center) < .003
    assert np.linalg.norm(new.position_m-center) < np.linalg.norm(old.position_m-center)
    assert np.linalg.eigvalsh(new.covariance_m2-old.covariance_m2).min() >= -1e-12
