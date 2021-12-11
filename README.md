# Camera utilities

This project contains utilities for generating, converting and managing camera poses in Python. It includes a set of functions for easy generation and manipulation of camera poses and trajectories. 

The `camu` module defines the `camera` class with a rich set of initialization and conversion functions.

The `geom` module defines functions for generating 3D poses on a spherical sector or trajectory. It also collects useful coordinate transformations. 

Usage example:
```
if mode == 'test':
    position_generator = geom.generate_spherical_spiral(
        world_origin, radius, world_up, image_count, 0, turns, sector, bounce_amplitude, bounce_frequency)
else:
    # sample spherical sector with uniform distribution
    position_generator = geom.generate_spherical_sector(
        world_origin, radius, world_up, sector, bounce_amplitude)
    
for i in range(image_count):
    camera_origin = next(position_generator)
    camera = camu.looking_at(
        world_origin, camera_origin, world_up, np.deg2rad(fov_x), image_width, image_height)
    camera.flip(using=camu.FLIP_OPENCV_CONVENTION) # flip camera up and left vectors
```
