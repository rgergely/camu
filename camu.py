'''Camera utilities'''

import numpy as np
import scipy.linalg

FLIP_NO_FLIP = [+1., +1., +1., +1.]
FLIP_OPENCV_CONVENTION = [-1., -1., +1., +1.]
FLIP_NERF_CONVENTION = [-1., +1., -1., +1.]


class camera:
    '''Camera class represents a pinhole camera with the 
    corresponding parameters. It defines basic functions 
    for camera and world co-sys transforms.
    '''

    def __init__(self, left, up, forward, origin, img_w, img_h, fov, skew=0.) -> None:
        '''Camera object initializer.
        :param left: camera left vector (in world coordinates, should be of unit length!)
        :param up: camera up vector (point of interest in world cooridnates, should be of unit length!)
        :param forward: camera forward vector (in world coordinates, should be of unit length!)
        :param origin: origin of the camera (in world coordinates)
        :param img_w: width of the camera image plane (in pixels)
        :param img_h: height of the camera image plane (in pixels)
        :param fov: horizontal field of view (in radians)
        :param skew: skew along camera x-axis
        '''
        def is_unit(x): return np.isclose(scipy.linalg.norm(x), 1.)
        assert(is_unit(left) and is_unit(up) and is_unit(forward))
        assert(np.isclose(np.dot(up, forward), 0.) and np.isclose(np.dot(up, left), 0.))

        # camera co-sys (extrinsic parameters)
        self.left = np.array(left)
        self.up = np.array(up)
        self.forward = np.array(forward)
        self.origin = np.array(origin)

        self.flip_axes = np.array(FLIP_NO_FLIP)

        # camera params (intrinsic parameters)
        self.fov = fov
        self.img_w = img_w
        self.img_h = img_h
        self.skew = skew

    def flip(self, using=FLIP_NO_FLIP):
        '''Flip specified camera co-sys vectors
        :param using: axes to flip in the camera co-sys (e.g., OpenCV convention flips x, y axes [-1, -1, +1, +1], 
                      default is `FLIP_NO_FLIP` [+1, +1, +1, +1], which means no flip)
        '''
        self.flip_axes *= using

    def extrinsics(self) -> np.ndarray:
        '''Calculate the camera extrensics matrix [R t] (world to camera transform)
        :return: camera extrensics matrix (4x4)
        '''
        return create_extrinsics(self.left, self.up, self.forward, self.origin, self.flip_axes)

    def intrinsics(self) -> np.ndarray:
        '''Calculate the camera extrensics matrix K (camera to image transform)
        :return: camera intrinsics matrix (4x4)
        '''
        return create_intrinsics(self.fov, self.img_w, self.img_h, self.skew)

    def pose(self) -> np.ndarray:
        '''Calculate the camera pose matrix [R' C] (camera to world transform)
        :return: camera pose matrix (4x4)
        '''
        return create_pose(self.left, self.up, self.forward, self.origin, self.flip_axes)

    def projection(self) -> np.ndarray:
        '''Calculate the camera projection matrix (world to image transform)
        :return: camera projection matrix (4x4)
        '''
        return self.intrinsics() @ self.extrinsics()

    def rays(self):
        '''Create a block of rays at world space from one camera
        :return: array of ray origins and ray directions
        '''
        tx = np.linspace(0.5, self.img_w - 0.5, self.img_w)
        ty = np.linspace(0.5, self.img_h - 0.5, self.img_h)
        pixels_x, pixels_y = np.meshgrid(tx, ty)

        p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1) # W, H, 3
        p = np.matmul(scipy.linalg.inv(self.K)[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3

        pose = self.c2w
        rays_v = p / scipy.linalg.norm(p, ord=2, axis=-1, keepdims=True)  # W, H, 3
        rays_v = np.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = np.broadcast_to(pose[None, None, :3, 3], rays_v.shape)  # W, H, 3

        return rays_o, rays_v

    # convenience definitions
    w2c = property(extrinsics)
    c2w = property(pose)
    K = property(intrinsics)
    Rt = property(extrinsics)
    P = property(projection)


def create(left, up, forward, origin, img_w, img_h, fov, skew=0.) -> camera:
    '''Create a camera instance
    :param left: camera left vector (in world coordinates, should be of unit length!)
    :param up: camera up vector (point of interest in world cooridnates, should be of unit length!)
    :param forward: camera forward vector (in world coordinates, should be of unit length!)
    :param origin: origin of the camera (in world coordinates)
    :param img_w: width of the camera image plane (in pixels)
    :param img_h: height of the camera image plane (in pixels)
    :param fov: horizontal field of view (in radians)
    :param skew: skew along camera x-axis
    :return: camera instance
    '''
    return camera(left, up, forward, origin, img_w, img_h, fov, skew)


def looking_at(target, origin, world_up, fov, img_w, img_h, skew=0.) -> camera:
    '''Create a camera instance looking at the specified target
    :param target: target of the camera (point of interest in world cooridnates)
    :param origin: origin of the camera (in world coordinates)
    :param world_up: world up vector (in world coordinates)
    :param fov: horizontal field of view (in radians)
    :param img_w: width of the camera image plane (in pixels)
    :param img_h: height of the camera image plane (in pixels)
    :param skew: skew along camera x-axis
    :return: camera instance
    '''
    def normalize(x): return x / scipy.linalg.norm(x)

    # camera co-sys
    forward = normalize(np.asarray(target) - np.asarray(origin))
    left = normalize(np.cross(world_up, forward))
    up = np.cross(forward, left)

    return camera(left, up, forward, origin, img_w, img_h, fov, skew)


def from_intrinsics_and_extrinsics(intrinsics, extrinsics) -> camera:
    '''Create a camera instance based on camera intrinsics and extrinsics
    :param intrinsics: camera intrinsics matrix (4x4)
    :param extrinsics: camera extrinsics matrix (4x4)
    :return: camera instance
    '''
    img_w, img_h, fov, skew = params_from_intrinsics(intrinsics)
    left, up, forward, origin = params_from_extrinsics(extrinsics)

    return camera(left, up, forward, origin, img_w, img_h, fov, skew)


def from_intrinsics_and_pose(intrinsics, pose) -> camera:
    '''Create a camera instance based on camera intrinsics and pose
    :param intrinsics: camera intrinsics matrix (4x4)
    :param pose: camera pose matrix (4x4)
    :return: camera instance
    '''
    img_w, img_h, fov, skew = params_from_intrinsics(intrinsics)
    left, up, forward, origin = params_from_pose(pose)

    return camera(left, up, forward, origin, img_w, img_h, fov, skew)


def from_projection(projection) -> camera:
    '''Create a camera instance based on camera projection matrix
    :param projection: camera projection matrix (4x4)
    :return: camera instance
    '''
    intrinsics, extrinsics = factorize_projection_matrix(projection)

    return from_intrinsics_and_extrinsics(intrinsics, extrinsics)


def params_from_intrinsics(intrinsics):
    '''Calculate camera intrinsic parameters from camera intrinsics
    :param intrinsics: camera intrinsics matrix (4x4)
    :return: camera intrinsic parameters (image plane width in pixels, image plane height in pixels, 
             horizontal field of view [fov] in radians, skew)
    '''
    intrinsics = np.asarray(intrinsics)

    f_x = intrinsics[0, 0]

    c_x = intrinsics[0, 2]
    c_y = intrinsics[1, 2]

    img_w = int(c_x * 2.)
    img_h = int(c_y * 2.)
    fov = np.arctan(c_x / f_x) * 2.
    skew = intrinsics[0, 1]

    return img_w, img_h, fov, skew


def params_from_extrinsics(extrinsics):
    '''Calculate extrinsic parameters from camera extrinsics
    (https://ksimek.github.io/2012/08/22/extrinsic/)
    :param extrinsics: camera extrinsics matrix, i.e., world to camera transform (4x4)
    :return: camera extrinsic parameters, i.e., camera co-sys (left, up, forward, origin)
    '''
    extrinsics = np.array(extrinsics)

    left = extrinsics[0, :3]
    up = extrinsics[1, :3]
    forward = extrinsics[2, :3]
    origin = -np.transpose(extrinsics[:3, :3]) @ extrinsics[:3, 3]

    return left, up, forward, origin


def params_from_pose(pose):
    '''Calculate extrinsic parameters from camera pose
    :param pose: camera pose matrix, i.e., camera to world transform (4x4)
    :return: camera extrinsic parameters, i.e., camera co-sys (left, up, forward, origin)
    '''
    pose = np.array(pose)

    left = pose[:3, 0]
    up = pose[:3, 1]
    forward = pose[:3, 2]
    origin = pose[:3, 3]

    return left, up, forward, origin


def factorize_projection_matrix(P):
    '''Factorize camera projection matrix into intrinsics and extrinsics matrices
    (https://cgcooke.github.io/Blog/computer%20vision/linear%20algebra/2020/03/13/RQ-Decomposition-In-Practice.html)
    (https://ksimek.github.io/2012/08/14/decompose/)
    :param P: camera projection matrix (4x4)
    :return: camera intrinsics matrix (4x4), camera extrinsics matrix (4x4)
    '''
    M = P[0:3, 0:3]
    K, R = scipy.linalg.rq(M)
    T = np.diag(np.sign(np.diag(K)))

    if scipy.linalg.det(T) < 0:
        T[1, 1] *= -1

    K = np.dot(K, T)
    R = np.dot(T, R)

    # camera origin in world coordinates
    C = -np.dot(scipy.linalg.inv(M), P[:3, 3])

    # world origin in camera coordinates
    t = -np.dot(R, C)

    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t

    return intrinsics, extrinsics


def to_pose(extrinsics):
    '''Convert camera extrinsics matrix to camera pose matrix
    :param extrinsics: camera extrinsics matrix (world to camera transform) (4x4)
    :return: camera pose matrix (camera to world transform) (4x4)
    '''
    return scipy.linalg.inv(extrinsics)


def to_extrinsics(pose):
    '''Convert camera pose matrix to camera extrinsics matrix
    :param pose: camera pose matrix (camera to world transform) (4x4)
    :return: camera extrinsics matrix (world to camera transform) (4x4)
    '''
    return scipy.linalg.inv(pose)


def create_intrinsics(fov, img_w, img_h, skew=0.):
    '''Create camera intrinsics matrix
    :param fov: horizontal field of view (in radians)
    :param img_w: width of the camera image plane (in pixels)
    :param img_h: height of the camera image plane (in pixels)
    :param skew: skew along camera x-axis
    :return: camera intrinsics matrix (4x4)
    '''
    c_x = img_w / 2.
    c_y = img_h / 2.

    f_x = c_x / np.tan(fov / 2.)
    f_y = f_x

    s = skew

    return np.array([[f_x,  s, c_x, 0.],
                     [0., f_y, c_y, 0.],
                     [0.,  0.,  1., 0.],
                     [0.,  0.,  0., 1.]])


def create_extrinsics(left, up, forward, origin, flip=FLIP_NO_FLIP):
    '''Create camera extrinsics matrix
    :param left: camera left vector (in world coordinates, should be of unit length!)
    :param up: camera up vector (point of interest in world cooridnates, should be of unit length!)
    :param forward: camera forward vector (in world coordinates, should be of unit length!)
    :param origin: origin of the camera (in world coordinates)
    :param flip: axes to flip in the camera co-sys (e.g., OpenCV convention flips x, y axes, 
                 default is `FLIP_NO_FLIP`, which means no flip)
    :return: camera extrinsics matrix (4x4)
    '''
    pose = create_pose(left, up, forward, origin, flip)

    # camera extrinsics, i.e., world to camera transform
    extrinsics = to_extrinsics(pose)

    return extrinsics


def create_pose(left, up, forward, origin, flip=FLIP_NO_FLIP):
    '''Create camera pose matrix
    :param left: camera left vector (in world coordinates, should be of unit length!)
    :param up: camera up vector (point of interest in world cooridnates, should be of unit length!)
    :param forward: camera forward vector (in world coordinates, should be of unit length!)
    :param origin: origin of the camera (in world coordinates)
    :param flip: axes to flip in the camera co-sys (e.g., OpenCV convention flips x, y axes, 
                 default is `FLIP_NO_FLIP`, which means no flip)
    :return: camera pose matrix (4x4)
    '''
    RC = np.column_stack((left, up, forward, origin))
    flip_transform = np.diag(flip)

    # camera pose, i.e., camera to world transform
    pose = np.vstack((RC, [0., 0., 0., 1.])) @ flip_transform

    return pose
