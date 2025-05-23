import math
import numpy as np

from dtcc_core.model import Mesh, LineString, MultiLineString
from dtcc_viewer import Scene, Window
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SunCollection
from dtcc_solar.perez import (
    get_perez_coeffs,
    compute_epsilon,
    perez_relative_luminance,
    compute_norm_factor,
)


class Skydome:

    mesh: Mesh
    mls: MultiLineString
    sky_vector_matrix: np.ndarray

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ray_dirs = []
        self.ray_areas = []
        self.solid_angles = []
        self.quad_midpoints = []

    def create_tregenza_mesh(self):
        elevs = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]
        patch_counts = [30, 30, 24, 24, 18, 12, 6, 1]
        elev_step = math.radians(12.0)
        bands = 8
        dome_area = self.calc_hemisphere_area()

        for i in range(bands):
            patch_count = patch_counts[i]
            elev = math.radians(elevs[i])
            elev_next = elev + elev_step
            azim_step = 2 * math.pi / patch_count

            if elev != math.radians(84.0):
                for j in range(patch_count):
                    azim = j * azim_step
                    azim_next = (j + 1) * azim_step
                    mid_elev = (elev + elev_next) / 2.0
                    mid_azim = (azim + azim_next) / 2.0

                    self.create_mesh_quad(azim, azim_next, elev, elev_next)
                    ray_dir = self.spherical_to_cartesian(1.0, mid_elev, mid_azim)
                    self.ray_dirs.append(ray_dir)

                    solid_angle = self.solid_angle(elev, elev_next, azim, azim_next)
                    self.solid_angles.append(solid_angle)

                    patch_area = self.calc_sphere_patch_area(
                        1.0, elev, elev_next, azim, azim_next
                    )
                    self.ray_areas.append(patch_area / dome_area)
            else:
                self.create_tregenza_zenith_patch(elev)

        mesh_vertices = np.array(self.vertices)
        mesh_faces = np.array(self.faces)
        self.mesh = Mesh(vertices=mesh_vertices, faces=mesh_faces)

        print("Tregenza skydome mesh created.")

    def create_tregenza_zenith_patch(self, elevation):
        v_count = len(self.vertices)
        dome_area = self.calc_hemisphere_area()

        ray_dir = [0.0, 0.0, 1.0]
        self.ray_dirs.append(ray_dir)

        zenith_angle = np.pi / 2.0 - elevation
        solid_angle = self.calc_top_patch_solid_angle(elevation)
        self.solid_angles.append(solid_angle)

        cap_area = self.calc_sphere_cap_area(elevation)
        self.ray_areas.append(cap_area / dome_area)

        for i in range(6):
            azimuth_step = 2 * math.pi / 6
            azim = i * azimuth_step
            pt = self.spherical_to_cartesian(1.0, elevation, azim)
            self.vertices.append(pt)

            idx1 = v_count + i
            idx2 = v_count + (i + 1) % 6
            idx3 = v_count + 6  # Zenith point

            self.faces.append([idx1, idx2, idx3])

        self.vertices.append([0.0, 0.0, 1.0])  # Zenith point

    def calc_hemisphere_area(self):
        r = 1.0
        return 2.0 * math.pi * r * r

    def calc_sphere_cap_area(self, elevation):
        polar_angle = math.pi / 2.0 - elevation
        r = 1.0
        return 2.0 * math.pi * r * r * (1 - math.cos(polar_angle))

    def create_mesh_quad(self, azim, next_azim, elev, next_elev):
        ray_length = 1.0

        pt1 = self.spherical_to_cartesian(ray_length, elev, azim)
        pt2 = self.spherical_to_cartesian(ray_length, next_elev, azim)
        pt3 = self.spherical_to_cartesian(ray_length, elev, next_azim)
        pt4 = self.spherical_to_cartesian(ray_length, next_elev, next_azim)

        self.vertices.extend([pt1, pt2, pt3, pt4])

        v_count = len(self.vertices)

        idx0 = v_count - 4
        idx1 = v_count - 3
        idx2 = v_count - 2
        idx3 = v_count - 1

        self.faces.append([idx0, idx1, idx2])
        self.faces.append([idx1, idx3, idx2])

        mp_x = (pt1[0] + pt2[0] + pt3[0] + pt4[0]) / 4.0
        mp_y = (pt1[1] + pt2[1] + pt3[1] + pt4[1]) / 4.0
        mp_z = (pt1[2] + pt2[2] + pt3[2] + pt4[2]) / 4.0

        self.quad_midpoints.append([mp_x, mp_y, mp_z])

    def calc_sphere_patch_area(self, r, elev1, elev2, azim1, azim2):
        return r * r * abs((azim2 - azim1) * (math.sin(elev2) - math.sin(elev1)))

    def spherical_to_cartesian(self, radius, elev, azim):
        x = radius * math.cos(elev) * math.cos(azim)
        y = radius * math.cos(elev) * math.sin(azim)
        z = radius * math.sin(elev)
        return [x, y, z]

    def create_dirvector_mls(self, line_s, line_dir):
        line_strings = []
        line_s = np.array(line_s)
        line_dir = np.array(line_dir)

        for start, direction in zip(line_s, line_dir):
            end = start + direction / np.linalg.norm(direction)
            vertices = []
            vertices.append(start)
            vertices.append(end)
            vertices = np.array(vertices)
            ls = LineString(vertices=vertices)
            line_strings.append(ls)

        self.mls = MultiLineString(linestrings=line_strings)

    def view(self):
        self.create_dirvector_mls(self.quad_midpoints, self.ray_dirs)
        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Skydome", self.mesh)
        scene.add_multilinestring("Skydome rays", self.mls)
        window.render(scene)

    def solid_angle(self, elev1, elev2, azim1, azim2):
        """
        Compute solid angle (steradians) of a spherical rectangle patch on a unit hemisphere,
        using elevation and azimuth bounds in radians.

        Parameters:
            elev1, elev2 : float
                Elevation bounds (in degrees), from horizon (0) up to zenith (π/2)
            azim1, azim2 : float
                Azimuth bounds (in degrgees), range [0, 2π]

        Returns:
            float : solid angle in steradians
        """
        elev1_rad = math.radians(elev1)
        elev2_rad = math.radians(elev2)
        azim1_rad = math.radians(azim1)
        azim2_rad = math.radians(azim2)

        return (azim2_rad - azim1_rad) * (np.sin(elev2_rad) - np.sin(elev1_rad))

    def calc_top_patch_solid_angle(self, elevation):
        zenith = math.pi / 2.0 - math.radians(elevation)
        solid_angle = 2 * np.pi * (1 - np.cos(zenith))
        return solid_angle

    def calc_sky_vector_matrix(self, sunpath: Sunpath):
        print("Cumulating radiation...")

        sun_vecs = sunpath.sunc.sun_vecs
        dni = sunpath.sunc.dni
        dhi = sunpath.sunc.dhi
        zenith = sunpath.sunc.zeniths
        solid_angles = self.solid_angles

        sky_vec_mat = np.zeros([len(sun_vecs), len(self.ray_dirs)])

        for i in range(len(sun_vecs)):
            # For each sun position compuite the radiation on each patch
            epsilon = compute_epsilon(dni[i], dhi[i], zenith[i])

            [a, b, c, d, e] = get_perez_coeffs(epsilon)

            fs = []
            patch_zeniths = []
            # Compute the radiation for each ray
            for j in range(len(self.ray_dirs)):
                ray_dir = self.ray_dirs[j]
                sun_patch_dot = np.dot(sun_vecs[i], ray_dir)
                sun_patch_angel = math.acos(sun_patch_dot)

                f = perez_relative_luminance(zenith[i], sun_patch_angel, a, b, c, d, e)
                fs.append(f)

                zvec = np.array([0.0, 0.0, 1.0])
                patch_zenith = math.acos(np.dot(zvec, ray_dir))
                patch_zeniths.append(patch_zenith)

            norm_factor = compute_norm_factor(dhi[i], fs, patch_zeniths, solid_angles)
            sky_vec_mat[i, :] = norm_factor * np.array(fs) * np.array(solid_angles)

        self.sky_vector_matrix = sky_vec_mat
