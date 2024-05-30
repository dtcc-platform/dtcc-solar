import math
import numpy as np
from typing import Any
from dtcc_model import Surface, MultiSurface, PointCloud, Field
from dtcc_viewer import Scene, Window


# Function to convert degrees to radians
def deg2rad(degrees):
    return degrees * math.pi / 180.0


# Function to calculate the area of a spherical patch
def spherical_patch_area(r, a1, a2, e1, e2):
    # Convert angles from degrees to radians
    a1_rad = deg2rad(a1)
    a2_rad = deg2rad(a2)
    e1_rad = deg2rad(e1)
    e2_rad = deg2rad(e2)

    # Calculate the area of the spherical patch
    area = r * r * abs((a2_rad - a1_rad) * (math.sin(e2_rad) - math.sin(e1_rad)))
    return area


def calc_sphere_cap_area(elevation):
    polar_angle = math.pi / 2.0 - elevation
    r = 1.0
    a = 2 * math.pi * r * r * (1 - math.cos(polar_angle))
    return a


def sph2cart(r, elevation, azimuth):
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)
    x = r * math.cos(elevation) * math.cos(azimuth)
    y = r * math.cos(elevation) * math.sin(azimuth)
    z = r * math.sin(elevation)
    return np.array([x, y, z])


def create_tregenza_sky_model():

    elevation_step = 12
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

    surfaces = []
    areas = []

    for num_patches, elev in tregenza_divisions:
        elev_next = elev + elevation_step
        azim_step = 360 / num_patches

        if elev != 84:
            for i in range(num_patches):
                azim = i * azim_step
                azim_next = (i + 1) * azim_step

                # Create vertices for each patch
                v1 = sph2cart(1, elev, azim)
                v2 = sph2cart(1, elev, azim_next)
                v3 = sph2cart(1, elev_next, azim_next)
                v4 = sph2cart(1, elev_next, azim)

                # Create the surface
                vertices = np.array([v1, v2, v3, v4])
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)

                area = spherical_patch_area(1, azim, azim_next, elev, elev_next)
                areas.append(area)
        else:
            # Create 1 top patch surface
            vertices = []
            for i in range(6):
                vertices.append(sph2cart(1, elev, i * 60))
            vertices = np.array(vertices)
            surface = Surface(vertices=vertices)
            surface.calculate_normal()
            surfaces.append(surface)

            elev1 = deg2rad(elev)
            area = calc_sphere_cap_area(elev1)
            areas.append(area)

        multi_surface = MultiSurface(surfaces=surfaces)

    areas = np.array(areas)

    tot_area = np.sum(areas)
    print("Total area: " + str(tot_area))
    print("Min max area: " + str(np.min(areas)) + " " + str(np.max(areas)))
    return multi_surface, areas


def create_reinhart_sky_model():

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
    areas = []

    for num_patches, elev in tregenza_divisions:
        elev_next = elev + altitude_step
        azim_step = 360 / num_patches

        if elev != 84:
            for i in range(num_patches):
                azim = i * azim_step
                azim_next = (i + 1) * azim_step

                # Create vertices for each patch
                v1 = sph2cart(1, elev, azim)
                v2 = sph2cart(1, elev, azim_next)
                v3 = sph2cart(1, elev_next, azim_next)
                v4 = sph2cart(1, elev_next, azim)

                # Create the surface
                vertices = np.array([v1, v2, v3, v4])
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)

                area = spherical_patch_area(1, azim, azim_next, elev, elev_next)
                areas.append(area)
        else:
            # Create 4 top patch surfaces
            for i in range(4):
                vertices = []
                vertices.append(sph2cart(1, elev, i * 90))
                vertices.append(sph2cart(1, elev, i * 90 + 30))
                vertices.append(sph2cart(1, elev, i * 90 + 60))
                vertices.append(sph2cart(1, elev, i * 90 + 90))
                vertices.append(np.array([0.0, 0.0, 1.0]))
                vertices = np.array(vertices)
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)

                area = 0.25 * calc_sphere_cap_area(deg2rad(elev))
                areas.append(area)

        multi_surface = MultiSurface(surfaces=surfaces)

    areas = np.array(areas)

    tot_area = np.sum(areas)
    print("Total area: " + str(tot_area))
    print("Min max area: " + str(np.min(areas)) + " " + str(np.max(areas)))

    return multi_surface, areas


def create_reinhart_sky_model_fine():

    elev_step = 3
    tregenza_divisions = [
        (120, 0.0),
        (120, 3.0),
        (120, 6.0),
        (120, 9.0),
        (120, 12.0),
        (120, 15.0),
        (120, 18.0),
        (120, 21.0),
        (96, 24.0),
        (96, 27.0),
        (96, 30.0),
        (96, 33.0),
        (96, 36.0),
        (96, 39.0),
        (96, 42.0),
        (96, 45.0),
        (72, 48.0),
        (72, 51.0),
        (72, 54.0),
        (72, 57.0),
        (48, 60.0),
        (48, 63.0),
        (48, 66.0),
        (48, 69.0),
        (24, 72.0),
        (24, 75.0),
        (24, 78.0),
        (24, 81.0),
        (1, 84.0),
    ]

    surfaces = []
    areas = []

    for num_patches, elev in tregenza_divisions:
        elev_next = elev + elev_step
        azim_step = 360 / num_patches

        if elev != 84:
            for i in range(num_patches):
                azim = i * azim_step
                azim_next = (i + 1) * azim_step

                # Create vertices for each patch
                v1 = sph2cart(1, elev, azim)
                v2 = sph2cart(1, elev, azim_next)
                v3 = sph2cart(1, elev_next, azim_next)
                v4 = sph2cart(1, elev_next, azim)

                # Create the surface
                vertices = np.array([v1, v2, v3, v4])
                surface = Surface(vertices=vertices)
                surface.calculate_normal()
                surfaces.append(surface)

                area = spherical_patch_area(1, azim, azim_next, elev, elev_next)
                areas.append(area)
        else:
            # Create 1 top patch surfaces
            vertices = []
            for i in range(4):
                vertices.append(sph2cart(1, elev, i * 90))
                vertices.append(sph2cart(1, elev, i * 90 + 30))
                vertices.append(sph2cart(1, elev, i * 90 + 60))
            vertices = np.array(vertices)
            surface = Surface(vertices=vertices)
            surface.calculate_normal()
            surfaces.append(surface)

            area = calc_sphere_cap_area(deg2rad(elev))
            areas.append(area)

        multi_surface = MultiSurface(surfaces=surfaces)

    areas = np.array(areas)

    tot_area = np.sum(areas)
    print("Total area: " + str(tot_area))
    print("Min max area: " + str(np.min(areas)) + " " + str(np.max(areas)))

    return multi_surface, areas


if __name__ == "__main__":

    # Create the Tregenza sky model
    tregenza_1, areas_1 = create_tregenza_sky_model()
    reinhart_2, areas_2 = create_reinhart_sky_model()
    reinhart_3, areas_3 = create_reinhart_sky_model_fine()

    print("Tregenza quad count: " + str(len(tregenza_1.surfaces)))
    print("Reinhart 1 quad count: " + str(len(reinhart_2.surfaces)))
    print("Reinhart 2 quad count: " + str(len(reinhart_3.surfaces)))

    window = Window(1200, 800)
    scene = Scene()

    field = Field(name="area", values=areas_1, dim=1)
    tregenza_1.add_field(field)

    field = Field(name="area", values=areas_2, dim=1)
    reinhart_2.add_field(field)

    field = Field(name="area", values=areas_3, dim=1)
    reinhart_3.add_field(field)

    scene.add_multisurface("Tregenza", tregenza_1)
    scene.add_multisurface("Reinhart", reinhart_2)
    scene.add_multisurface("Reinhart", reinhart_3)

    window.render(scene)
