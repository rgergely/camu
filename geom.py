'''Geometry Utilities'''

import numpy as np


def pol2cart(vector) -> np.ndarray:
    '''Convert 3D polar coordinates to Cartesian coordinates
    :param rho: is the radius
    :param theta: is the rotation angle in the x-y plane
    :param phi: is the rotation angle to z
    :return: vector with cartesian coordinates
    '''
    rho, theta, phi = vector
    x = rho * np.cos(theta) * np.sin(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(phi)
    return np.array([x, y, z])


def cart2pol(vector) -> np.ndarray:
    '''Convert Cartesian coordinates to 3D polar coordinates
    :param x: is the first Cartesian coordinate
    :param y: is the second Cartesian coordinate
    :param z: is the third Cartesian coordinate
    :return: vector with polar coordinates
    '''
    x, y, z = vector
    rho = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / rho)
    return np.array([rho, theta, phi])


def calculate_rotation_matrix(up=[0., 0., 1.]) -> np.ndarray:
    '''Calculate rotation matrix that transforms points into a coordinate system
    whose z-axis points to `up` direction
    :param up: up direction (z-axis) of the target co-sys (3x1 vector)
    :return: rotation matrix (3x3)
    '''
    def normalize(x): return x / np.linalg.norm(x)

    z = normalize(np.asarray(up))
    x = [1., 0., 0.] if not np.isclose(
        np.abs(np.dot([1., 0., 0.], z)), 1.) else [0., 1., 0.]
    y = normalize(np.cross(z, x))
    x = np.cross(y, z)
    R = np.column_stack((x, y, z))

    return R


def generate_spherical_sector(center=[0., 0., 0.], radius=1., up=[0., 1., 0.], sector=[-1., 1.], bounce_amplitude=0.) -> np.ndarray:
    '''Generate a uniformly random sample on a spherical sector
    :param center: sphere center (3x1 vector)
    :param radius: sphere radius
    :param up: sphere up direction (3x1 vector)
    :param sector: sector margins along the up axis between [-1., 1.] (default: [-1., 1.] which is a full sphere)
    :param bounce_amplitude: variance in the radius
    :return: uniformly random position on the spherical sector
    '''
    low, high = sector
    R = calculate_rotation_matrix(up)

    while True:
        rho = radius + np.random.uniform(-bounce_amplitude, bounce_amplitude)
        theta = np.random.uniform(0, np.pi * 2.)
        phi = np.arccos(np.random.uniform(low, high))
        dir = pol2cart((rho, theta, phi))
        sample = center + np.dot(R, dir)

        yield sample


def generate_spherical_spiral(center=[0., 0., 0.], radius=1., up=[0., 1., 0.], steps=200, begin=0, turns=1, sector=[-0.85, 0.85], bounce_amplitude=0., bounce_frequency=0) -> np.ndarray:
    '''Generate consecutive samples on a spherical spiral 
    :param center: sphere center (3x1 vector)
    :param radius: sphere radius
    :param up: sphere up direction (3x1 vector)
    :param steps: number of steps to take
    :param begin: index of the step to begin with (defult is zero)
    :param turns: number of full turns around the up axis
    :param sector: sector margins along the up axis between [-1., 1.]
    :return: uniformly random position on the spherical sector
    '''
    low, high = sector
    R = calculate_rotation_matrix(up)

    dtheta = np.pi * 2 * turns / steps
    dphi = (np.arccos(low) - np.arccos(high)) / steps

    rho = radius
    theta = dtheta * begin
    phi = np.arccos(high) + dphi * begin

    for i in range(steps):
        dir = pol2cart([rho, theta, phi])
        sample = center + np.dot(R, dir)

        yield sample

        rho = radius + bounce_amplitude * \
            np.sin(np.pi * 2 * bounce_frequency * turns * i / steps)
        theta += dtheta
        phi += dphi
