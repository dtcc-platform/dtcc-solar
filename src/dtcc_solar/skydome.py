import math
import numpy as np

from dtcc_core.model import Mesh, LineString, MultiLineString
from dtcc_viewer import Scene, Window
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SunCollection
from dtcc_solar.perez import (
    get_perez_coeffs,
    compute_epsilon,
    per_rel_luminance,
    calc_norm_factor,
)
from dtcc_solar.logging import info, debug, warning, error


class Skydome:

    mesh: Mesh
    mls: MultiLineString
    sky_vector_matrix: np.ndarray
    per_rel_lumiance: np.ndarray

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ray_dirs = []
        self.patch_zeniths = []
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
                    self.patch_zeniths.append((math.pi / 2.0) - mid_elev)

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

        # Should be ≈ 6.283 (2π)
        tot_solid_angle = np.sum(self.solid_angles)
        info("Total solid angle: " + str(tot_solid_angle) + ", expected: 6.283")
        info("Tregenza skydome mesh created.")

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

    def view(self, data=None):
        self.create_dirvector_mls(self.quad_midpoints, self.ray_dirs)
        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Skydome", self.mesh, data=data)
        scene.add_multilinestring("Skydome rays", self.mls)
        window.render(scene)

    def solid_angle(self, elev1, elev2, azim1, azim2):
        """
        Compute solid angle (steradians) of a spherical rectangle patch on a unit hemisphere,
        using elevation and azimuth bounds in **radians**.

        Parameters:
            elev1, elev2 : float
                Elevation bounds (in radians), from horizon (0) up to zenith (π/2)
            azim1, azim2 : float
                Azimuth bounds (in degrgees), range [0, 2π]

        Returns:
            float : solid angle in steradians
        """

        return (azim2 - azim1) * (np.sin(elev2) - np.sin(elev1))

    def calc_top_patch_solid_angle(self, elev):
        zenith = math.pi / 2.0 - elev
        solid_angle = 2 * np.pi * (1 - np.cos(zenith))
        return solid_angle

    def calc_sky_vector_matrix(self, sunpath: Sunpath):
        print("Cumulating radiation...")

        sun_vecs = sunpath.sunc.sun_vecs
        dni = sunpath.sunc.dni
        dhi = sunpath.sunc.dhi
        sun_zenith = sunpath.sunc.zeniths
        solid_angles = np.array(self.solid_angles)

        sky_vec_mat = np.zeros([len(self.ray_dirs), len(sun_vecs)])
        pre_rel_lum = np.zeros([len(self.ray_dirs), len(sun_vecs)])

        zvec = np.array([0.0, 0.0, 1.0])

        for i in range(len(sun_vecs)):
            # For each sun position compuite the radiation on each patch
            if dhi[i] > 0.0 and sun_zenith[i] < np.pi / 2.0:
                epsilon = compute_epsilon(dni[i], dhi[i], sun_zenith[i])
                [a, b, c, d, e] = get_perez_coeffs(epsilon)

                fs = []
                patch_zeniths = []
                # Compute the radiation for each ray (i.e. each sky patch)
                for j in range(len(self.ray_dirs)):
                    ray_dir = np.array(self.ray_dirs[j])
                    sun_patch_dot = np.dot(sun_vecs[i], ray_dir)
                    sun_patch_angle = math.acos(sun_patch_dot)

                    f = per_rel_luminance(sun_zenith[i], sun_patch_angle, a, b, c, d, e)

                    fs.append(f)

                    patch_zenith_dot = np.dot(zvec, ray_dir)
                    patch_zenith = math.acos(patch_zenith_dot)
                    patch_zeniths.append(patch_zenith)

                patch_zeniths = np.array(patch_zeniths)
                fs = np.array(fs)

                norm_factor = calc_norm_factor(dhi[i], fs, patch_zeniths, solid_angles)
                print(f"Sun {i}, normalization factor: {norm_factor:.2f}")
                pre_rel_lum[:, i] = fs
                sky_vec_mat[:, i] = norm_factor * fs * solid_angles

                # Check total diffuse energy matches
                energy = np.sum(sky_vec_mat[:, i])
                print(f"Sun {i}, diffuse energy: {energy:.2f}, expected: {dhi[i]:.2f}")

                self.validate_sky_normalization(fs, patch_zeniths, solid_angles)

            else:
                # If no diffuse radiation, set to zero
                pre_rel_lum[:, i] = 0.0
                sky_vec_mat[:, i] = 0.0

        # Store as class attributes
        self.per_rel_lumiance = pre_rel_lum
        self.sky_vector_matrix = sky_vec_mat

    def validate_sky_normalization(
        self, f: np.ndarray, zenith_angles: np.ndarray, solid_angles: np.ndarray
    ):
        """
        Validate that the Perez relative luminance distribution is correctly normalized.

        Parameters:
            f : np.ndarray
                Array of relative luminance values for each patch
            zenith_angles : np.ndarray
                Array of zenith angles for each patch [radians]
            solid_angles : np.ndarray
                Array of solid angles for each patch [steradians]

        Prints:
            - Normalization integral
            - Deviation from 1.0
        """
        cos_zenith = np.clip(np.cos(zenith_angles), 0.0, 1.0)
        weights = f * cos_zenith * solid_angles
        integral = np.sum(weights)

        print(f"Perez sky normalization check:")
        print(f"∑(f * cos(θ) * dΩ) = {integral:.6f}")
        print(f"Deviation from 1.0: {abs(1.0 - integral):.6f}")

        return integral
