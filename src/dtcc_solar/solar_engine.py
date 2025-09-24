import math
import numpy as np
from pprint import pp
from dtcc_solar.utils import SolarParameters, calc_face_areas, concatenate_meshes
from dtcc_solar import py_embree_solar as embree
from dtcc_solar.utils import SunCollection, OutputCollection, SkyType, plot_matrix
from dtcc_solar.utils import Rays, SunMapping, split_mesh_by_face_mask
from dtcc_solar.skydome import Skydome
from dtcc_solar.sunpath import Sunpath
from dtcc_core.model import Mesh, PointCloud
from dtcc_core.model import Bounds
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer import Window, Scene
from dtcc_solar.viewer import Viewer
from dtcc_solar.perez import calc_2_phase_matrix, calc_3_phase_matrices
import matplotlib.pyplot as plt


class SolarEngine:
    """
    Class for performing solar analysis on 3D meshes.

    Attributes
    ----------
    mesh : Mesh
        The combined mesh for analysis including both analysis and shading meshes.
    analysis_mesh : Mesh
        The mesh that will be analyzed for solar exposure.
    shading_mesh : Mesh
        The mesh that will cast shadows but not be analyzed.
    origin : np.ndarray
        The origin point of the mesh.
    horizon_z : float
        The z-coordinate of the horizon.
    sunpath_radius : float
        The radius of the sun path.
    sun_size : float
        The size of the sun for analysis purposes.
    path_width : float
        The width of the sun path.
    bb : Bounds
        The bounding box of the mesh.
    bbx : np.ndarray
        The x-coordinates of the bounding box.
    bby : np.ndarray
        The y-coordinates of the bounding box.
    bbz : np.ndarray
        The z-coordinates of the bounding box.
    face_mask : np.ndarray
        A mask indicating which faces of the mesh are to be analyzed.
    sky : Sky
        The sky model used for analysis.
    rays : Rays
        The rays model used for ray tracing.
    """

    mesh: Mesh
    analysis_mesh: Mesh
    shading_mesh: Mesh
    origin: np.ndarray
    horizon_z: float
    sunpath_radius: float
    sun_size: float
    path_width: float
    bb: Bounds
    bbx: np.ndarray
    bby: np.ndarray
    bbz: np.ndarray
    face_mask: np.ndarray
    sky: SkyType
    rays: Rays

    def __init__(
        self,
        analysis_mesh: Mesh,
        shading_mesh: Mesh = None,
        center_mesh: bool = False,
    ):
        """
        Initialize the SolarEngine with the provided meshes and parameters.

        Parameters
        ----------
        analysis_mesh : Mesh
            The mesh that will be analyzed for solar exposure.
        shading_mesh : Mesh, optional
            The mesh that will cast shadows but not be analyzed.
        sky : Sky, optional
            The sky model used for analysis.
        rays : Rays, optional
            The rays model used for ray tracing.
        center_mesh : bool, optional
            If True, the mesh will be centered based on x and y coordinates.
        """

        self.analysis_mesh = analysis_mesh
        self.shading_mesh = shading_mesh
        (self.mesh, self.face_mask) = self._join_meshes(analysis_mesh, shading_mesh)
        self.origin = np.array([0, 0, 0])
        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.path_width = 0
        self._preprocess_mesh(center_mesh)

        info("-----------------------------------------------------")
        info("Solar engine created:")
        info(f"  Analysis mesh has {len(self.analysis_mesh.faces)} faces.")
        if self.shading_mesh is not None:
            info(f"  Shading mesh has {len(self.shading_mesh.faces)} faces.")
            info(f"  Combined mesh has {len(self.mesh.faces)} faces.")
        info(f"  Mesh moved to center: {center_mesh}")
        info("-----------------------------------------------------")

    def _join_meshes(self, analysis_mesh: Mesh, shading_mesh: Mesh = None):
        """
        Join the analysis mesh and shading mesh into a single mesh.

        Parameters
        ----------
        analysis_mesh : Mesh
            The mesh that will be analyzed for solar exposure.
        shading_mesh : Mesh, optional
            The mesh that will cast shadows but not be analyzed.

        Returns
        -------
        tuple
            A tuple containing the combined mesh and a face mask indicating
            which faces of the mesh are to be analyzed.
        """

        if shading_mesh is None:
            return analysis_mesh, np.ones(len(analysis_mesh.faces), dtype=bool)

        mesh = concatenate_meshes([analysis_mesh, shading_mesh])
        face_mask = np.ones(len(analysis_mesh.faces), dtype=bool)
        face_mask = np.append(face_mask, np.zeros(len(shading_mesh.faces), dtype=bool))

        return mesh, face_mask

    def _preprocess_mesh(self, move_to_center: bool):
        """
        Preprocess the mesh, including calculating bounds and optionally moving the mesh to the center.

        Parameters
        ----------
        move_to_center : bool
            If True, the mesh will be centered based on x and y coordinates.
        """
        self._calc_bounds()
        # Center mesh based on x and y coordinates only
        center_bb = np.array([np.mean(self.bbx), np.mean(self.bby), np.mean(self.bbz)])
        centerVec = self.origin - center_bb

        # Move the mesh to the centre of the model
        if move_to_center:
            self.mesh.vertices += centerVec
            info(f"Mesh has been moved to origin.")

        # Update bounding box after the mesh has been moved
        self._calc_bounds()

        # Assumption: The horizon is the avrage z height in the mesh
        self.horizon_z = 0.0  # np.average(self.mesh.vertices[:, 2])

        # Calculating sunpath radius
        dx = self.bb.width
        dy = self.bb.height
        dz = self.zmax - self.zmin
        self.sunpath_radius = 0.5 * math.sqrt(dx**2 + dy**2 + dz**2)

        # Hard coded size proportions
        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40
        self.tolerance = self.sunpath_radius / 1.0e7

    def _calc_bounds(self):
        """Calculate the bounding box of the mesh and set related attributes."""
        self.xmin = self.mesh.vertices[:, 0].min()
        self.xmax = self.mesh.vertices[:, 0].max()
        self.ymin = self.mesh.vertices[:, 1].min()
        self.ymax = self.mesh.vertices[:, 1].max()
        self.zmin = self.mesh.vertices[:, 2].min()
        self.zmax = self.mesh.vertices[:, 2].max()

        self.bbx = np.array([self.xmin, self.xmax])
        self.bby = np.array([self.ymin, self.ymax])
        self.bbz = np.array([self.zmin, self.zmax])

        self.bb = Bounds(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

    def run_2_phase_analysis(self, sunp: Sunpath, skyd: Skydome, p: SolarParameters):

        (sky_res, sun_res) = calc_2_phase_matrix(sunp, skyd, p.sun_mapping)

        matrix = sun_res.matrix + sky_res.matrix

        ray_dirs = np.array(skyd.ray_dirs)
        solid_angles = np.array(skyd.solid_angles)

        info("-----------------------------------------------------")
        info(f"Creating Embree instance and running analysis...")
        info("-----------------------------------------------------")

        self.embree = embree.PyEmbreeSolar(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_mask,
            ray_dirs,
            solid_angles,
        )

        self.embree.run_2_phase_analysis(matrix)

        vis_mat = self.embree.get_visibility_matrix_tot()
        prj_mat = self.embree.get_projection_matrix_tot()
        irr_mat = self.embree.get_irradiance_matrix_tot()

        vis_vec = np.sum(vis_mat, axis=1)
        prj_vec = np.sum(prj_mat, axis=1)
        irr_vec = np.sum(irr_mat, axis=1) * 0.001  # Convert to kWh/m2

        dict_data = {
            "irradiance [kWh/m2]": irr_vec,
            "visibility": vis_vec,
            "projection": prj_vec,
        }

        face_normals = self.embree.get_face_normals()

        self.check_2_phase_energy_balance(skyd, matrix, irr_vec, vis_mat, face_normals)

        # self.plot_matrices(vis_mat, matrix)

        sun_pc = PointCloud(points=sunp.sunc.positions)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Mesh", self.mesh, data=dict_data)
        scene.add_pointcloud("Suns", sun_pc, 0.1)
        window.render(scene)

    def run_3_phase_analysis(self, sunp: Sunpath, skyd: Skydome, p: SolarParameters):

        sky_res, sun_res = calc_3_phase_matrices(sunp, skyd)

        sky_matrix = sky_res.matrix
        sun_matrix = sun_res.matrix

        sky_ray_dirs = np.array(skyd.ray_dirs)
        sun_ray_dirs = np.array(sunp.sunc.sun_vecs)
        sky_solid_angles = np.array(skyd.solid_angles)
        sun_solid_angles = np.ones(sunp.sunc.count)

        info("-----------------------------------------------------")
        info(f"Creating Embree instance and running analysis...")
        info("-----------------------------------------------------")

        self.embree = embree.PyEmbreeSolar(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_mask,
            sky_ray_dirs,
            sky_solid_angles,
            sun_ray_dirs,
            sun_solid_angles,
        )

        self.embree.run_3_phase_analysis(sky_matrix, sun_matrix)

        sky_vis_mat = self.embree.get_visibility_matrix_sky()
        sky_proj_mat = self.embree.get_projection_matrix_sky()
        sky_irr_mat = self.embree.get_irradiance_matrix_sky()

        sun_vis_mat = self.embree.get_visibility_matrix_sun()
        sun_proj_mat = self.embree.get_projection_matrix_sun()
        sun_irr_mat = self.embree.get_irradiance_matrix_sun()

        sky_vis = np.sum(sky_vis_mat, axis=1)
        sky_pro = np.sum(sky_proj_mat, axis=1)
        sky_irr = np.sum(sky_irr_mat, axis=1) * 0.001  # Convert to kWh/m2

        sun_vis = np.sum(sun_vis_mat, axis=1)
        sun_pro = np.sum(sun_proj_mat, axis=1)
        sun_irr = np.sum(sun_irr_mat, axis=1) * 0.001  # Convert to kWh/m2

        tot_irr = sky_irr + sun_irr

        dict_data = {
            "sky irradiance [kWh/m2]": sky_irr,
            "sky visibility": sky_vis,
            "sky projection": sky_pro,
            "sun irradiance [kWh/m2]": sun_irr,
            "sun visibility": sun_vis,
            "sun projection": sun_pro,
            "total irradiance [kWh/m2]": tot_irr,
        }

        face_normals = self.embree.get_face_normals()

        # self.check_3_phase_energy_balance(
        #    skyd,
        #    sky_matrix,
        #    sky_vis,
        #    sky_irr,
        #    sunp,
        #    sun_matrix,
        #    sun_vis,
        #    sun_irr,
        #    face_normals,
        # )

        # self.plot_matrices(sky_vis_mat, sky_matrix)
        # self.plot_matrices(sun_vis_mat, sun_matrix)

        sun_pc = PointCloud(points=sunp.sunc.positions)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Mesh", self.mesh, data=dict_data)
        scene.add_pointcloud("Suns", sun_pc, 0.1)
        window.render(scene)

    def check_2_phase_energy_balance(
        self,
        skydome: Skydome,
        tot_mat: np.ndarray,
        irr_vec: np.ndarray,
        vis_mat: np.ndarray,
        face_normals: np.ndarray,
    ):
        """
        Check energy balance by comparing total sun + sky matrix energy with received irradiance.
        Uses the most upward-facing and fully visible face(s) as reference.
        """

        # Sum total sun + sky contribution per face shape:(N_sky_patches, timesteps)
        total_energy = np.sum(tot_mat, axis=1)

        # Project energy onto horizontal plane (sky dome side)
        proj_total_energy = total_energy * np.cos(skydome.patch_zeniths)

        sum_energy = np.sum(proj_total_energy) * 0.001  # Convert to kWh/m2

        # Identify fully visible faces
        all_visible = np.all(vis_mat == 1, axis=1)

        # Include only fully visible faces that are not shading faces
        mask = all_visible & self.face_mask

        # print("Sum all visible: " + str(np.sum(all_visible)))
        # print("Sum face_mask: " + str(np.sum(self.face_mask)))
        # print("Sum mask: " + str(np.sum(mask)))

        # Find dot product with up vector
        up_vec = np.array([0.0, 0.0, 1.0])
        cos_angles = np.dot(face_normals, up_vec)

        cos_angles[~mask] = -np.inf  # Exclude non-fully-visible faces

        # Among fully visible faces, find the one(s) with the highest upward alignment
        max_dot = np.max(cos_angles)

        # Find indices of faces with this maximum upward alignment
        valid_indices = np.where(cos_angles == max_dot)[0]

        irr_values = irr_vec[valid_indices]

        res = np.max(irr_values)

        # Max irradiance on any face (for context)
        max_irr = np.max(irr_vec)

        mesh_in, mesh_out = split_mesh_by_face_mask(self.mesh, all_visible)

        # Report
        info("-----------------------------------------------------")
        info("Energy balance results:")
        info(f"  Total sun + sky horizontal plane irradiance : {sum_energy:.2f} kWh/m²")
        info(f"  Irradiance on most horizontal visible faces : {res:.2f} kWh/m²")
        info(f"  Max irradiance (any face): {max_irr:.2f} Wh/m²")
        info(f"  Found {len(valid_indices)} close to upward-facing visible face(s)")
        info("-----------------------------------------------------")

        return mesh_in, mesh_out

    def check_3_phase_energy_balance(
        self,
        skydome: Skydome,
        sky_matrix: np.ndarray,
        sky_vis: np.ndarray,
        sky_irr: np.ndarray,
        sunpath: Sunpath,
        sun_matrix: np.ndarray,
        sun_vis: np.ndarray,
        sun_irr: np.ndarray,
        face_normals: np.ndarray,
    ):
        """
        Check energy balance by comparing sun + sky matrix energy with received irradiance.
        Uses the most upward-facing and fully visible face(s) as reference.
        """

        patch_zeniths = np.array(skydome.patch_zeniths)
        sun_zeniths = np.array(sunpath.sunc.zeniths)

        # Sum sky and sun projected on a horizontal plane W/m²
        sky_energy = np.sum(sky_matrix, axis=1) * patch_zeniths
        sun_energy = np.sum(sun_matrix, axis=1) * sun_zeniths
        sum_energy = np.sum(sky_energy + sun_energy)

        # Identify fully visible faces (for both sky and sun)
        fully_visible_sky = np.all(sky_vis == 1, axis=1)
        fully_visible_sun = np.all(sun_vis == 1, axis=1)
        fully_visible_mask = fully_visible_sky & fully_visible_sun

        # Normalise face normals
        normed_normals = face_normals / np.linalg.norm(
            face_normals, axis=1, keepdims=True
        )

        # Dot with vertical (z-up) vector
        up_vector = np.array([0.0, 0.0, 1.0])
        cos_up = np.dot(normed_normals, up_vector)

        # Exclude non-fully-visible faces
        cos_up[~fully_visible_mask] = -np.inf
        max_up = np.max(cos_up)

        # Get the face(s) most aligned with upward vector
        valid_indices = np.where(cos_up == max_up)[0]

        # Compute irradiance on those faces
        sky_face_irr = np.sum(sky_irr[valid_indices, :], axis=1)
        sun_face_irr = np.sum(sun_irr[valid_indices, :], axis=1)

        # For horizontal faces calculate mean irradiance
        hor_faces_mean_irr = np.mean(sky_face_irr + sun_face_irr)

        # Max irradiance (any face, all directions)
        all_faces_max_irr = np.max(np.sum(sky_irr, axis=1) + np.sum(sun_irr, axis=1))

        # Report
        info("-----------------------------------------------------")
        info("Energy balance results:")
        info(
            f"  Projected total sun + sky irradiance (horizontal): {sum_energy:.2f} Wh/m²"
        )
        info(
            f"  Irradiance on most upward-facing visible faces    : {hor_faces_mean_irr:.2f} Wh/m²"
        )
        info(
            f"  Maximum irradiance on any face                    : {all_faces_max_irr:.2f} Wh/m²"
        )
        info(f"  Found {len(valid_indices)} upward-facing visible face(s)")
        info("-----------------------------------------------------")

        return valid_indices

    def plot_matrices(self, vis_mat, tot_mat):

        plot_matrix(vis_mat, cmap="gray")
        plot_matrix(tot_mat, cmap="jet")
