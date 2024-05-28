import math
import numpy as np
from typing import Any
from dtcc_model import Surface, MultiSurface, PointCloud, Field
from dtcc_viewer import Scene, Window


def generate_tregenza_sky_model():
    # Define the division parameters for the Tregenza sky model
    tregenza_divisions = [
        (1, 30.0),
        (8, 15.0),
        (16, 12.0),
        (24, 10.0),
        (32, 8.0),
        (40, 6.0),
        (48, 5.0),
        (8, 3.0),
        (1, 0.0),
    ]

    multi_surface = MultiSurface()

    zenith = np.array([0.0, 0.0, 1.0])
    for division in tregenza_divisions:
        num_patches, altitude = division
        altitude_rad = math.radians(altitude)
        ring_radius = math.sin(altitude_rad)
        ring_height = math.cos(altitude_rad)
        for i in range(num_patches):
            azimuth_rad = (2 * math.pi / num_patches) * i
            next_azimuth_rad = (2 * math.pi / num_patches) * (i + 1)

            vertices = np.array(
                [
                    [
                        ring_radius * math.cos(azimuth_rad),
                        ring_radius * math.sin(azimuth_rad),
                        ring_height,
                    ],
                    [
                        ring_radius * math.cos(next_azimuth_rad),
                        ring_radius * math.sin(next_azimuth_rad),
                        ring_height,
                    ],
                    [0.0, 0.0, 1.0],  # Connect to zenith
                ]
            )

            surface = Surface(vertices=vertices)
            surface.calculate_normal()
            multi_surface.surfaces.append(surface)

    return multi_surface


def sph2cart(r, theta, phi):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def generate_tregenza_sky_model_2(subdee: bool = False):

    altitude_step = 12
    tregenza_divisions = [
        (30, 0.0),
        (30, 12.0),
        (24, 24.0),
        (24, 36.0),
        (18, 48.0),
        (12, 60.0),
        (6, 72.0),
        (1, 84.0),
    ]
    if subdee:
        altitude_step = 6
        tregenza_divisions = [
            (60, 0.0),
            (60, 6.0),
            (60, 12.0),
            (60, 18.0),
            (48, 24.0),
            (48, 30.0),
            (48, 36.0),
            (48, 42.0),
            (36, 48.0),
            (36, 54.0),
            (24, 60.0),
            (24, 66.0),
            (12, 72.0),
            (12, 78.0),
            (4, 84.0),
        ]

    surfaces = []

    for num_patches, altitude in tregenza_divisions:
        altitude_next = altitude + altitude_step if altitude != 84 else 90
        azimuth_step = 360 / num_patches

        if altitude != 84:
            for i in range(num_patches):
                azimuth = i * azimuth_step
                azimuth_next = (i + 1) * azimuth_step

                # Create vertices for each patch
                v1 = sph2cart(1, 90 - altitude, azimuth)
                v2 = sph2cart(1, 90 - altitude, azimuth_next)
                v3 = sph2cart(1, 90 - altitude_next, azimuth_next)
                v4 = sph2cart(1, 90 - altitude_next, azimuth)

                # Create the surface
                vertices = np.array([v1, v2, v3, v4])
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)
        elif not subdee:
            # Create 1 top patch surface
            vertices = []
            for i in range(6):
                vertices.append(sph2cart(1, 90 - altitude, i * 60))
            vertices = np.array(vertices)
            surface = Surface(vertices=vertices)
            surface.calculate_normal()
            surfaces.append(surface)
        elif subdee:
            # Create 4 top patch surfaces
            for i in range(4):
                vertices = []
                vertices.append(sph2cart(1, 90 - altitude, i * 90))
                vertices.append(sph2cart(1, 90 - altitude, i * 90 + 30))
                vertices.append(sph2cart(1, 90 - altitude, i * 90 + 60))
                vertices.append(sph2cart(1, 90 - altitude, i * 90 + 90))
                vertices.append(np.array([0.0, 0.0, 1.0]))
                vertices = np.array(vertices)
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)

        multi_surface = MultiSurface(surfaces=surfaces)

    return multi_surface


if __name__ == "__main__":

    # Create the Tregenza sky model
    dome = generate_tregenza_sky_model_2(subdee=False)
    print("Quad count: " + str(len(dome)))
    window = Window(1200, 800)
    scene = Scene()
    field = Field(name="id", values=np.random.rand(len(dome.surfaces)), dim=1)
    dome.add_field(field)
    scene.add_multisurface("Ms Dome", dome)
    window.render(scene)
