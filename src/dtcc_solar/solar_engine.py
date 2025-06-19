import math
import numpy as np
from dtcc_solar.utils import SolarParameters, calc_face_areas, concatenate_meshes
from dtcc_solar import py_embree_solar as embree
from dtcc_solar.utils import SunCollection, OutputCollection, Sky
from dtcc_solar.utils import Rays, SunMatrixType
from dtcc_solar.skydome import Skydome
from dtcc_solar.sunpath import Sunpath
from dtcc_core.model import Mesh, PointCloud
from dtcc_core.model import Bounds
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer import Window, Scene
from dtcc_solar.viewer import Viewer
from pprint import pp


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
    sky: Sky
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

        info("Solar engine created")

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
        center_bb = np.array(
            [np.average(self.bbx), np.average(self.bby), np.average(self.bbz)]
        )
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
        self.sunpath_radius = math.sqrt(
            math.pow(dx / 2, 2) + math.pow(dy / 2, 2) + math.pow(dz / 2, 2)
        )

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

    def run_2_phase_analysis(
        self,
        sunp: Sunpath,
        skydome: Skydome,
        tot_matrix: np.ndarray,
    ):

        ray_dirs = np.array(skydome.ray_dirs)
        solid_angles = np.array(skydome.solid_angles)

        # for i in range(len(self.face_mask)):
        #    print(f"Face {i + 1}/{len(self.face_mask)}: {self.face_mask[i]}")

        self.embree = embree.PyEmbreeSolar(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_mask,
            ray_dirs,
            solid_angles,
        )

        face_areas = calc_face_areas(self.mesh)

        info(f"Embree instance created successfully.")
        info(f"Running analysis...")

        self.embree.run_2_phase_analysis(tot_matrix)

        vis_mat = self.embree.get_visibility_results()
        proj_mat = self.embree.get_projection_results()
        irr_mat = self.embree.get_irradiance_results()

        irr = np.sum(irr_mat, axis=1) * 0.001  # Convert to kWh/m2
        vis = np.sum(vis_mat, axis=1)
        pro = np.sum(proj_mat, axis=1)

        energy = irr * face_areas  # Convert to kWh

        dict_data = {
            "irradiance [kWh/m2]": irr,
            "energy [kWh]": energy,
            "visibility": vis,
            "projection": pro,
        }

        face_normals = self.embree.get_face_normals()

        check_energy_balance(skydome, tot_matrix, irr_mat, vis_mat, face_normals)

        sun_pc = PointCloud(points=sunp.sunc.positions)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Mesh", self.mesh, data=dict_data)
        scene.add_pointcloud("Suns", sun_pc, 0.1)
        window.render(scene)


def check_energy_balance(
    skydome: Skydome,
    tot_matrix: np.ndarray,
    irradiance: np.ndarray,
    visibility: np.ndarray,
    face_normals: np.ndarray,
):
    """
    Check energy balance by comparing total sun + sky matrix energy with received irradiance.
    Uses the most upward-facing and fully visible face(s) as reference.
    """

    # Sum total sun + sky contribution per face (shape: N_faces,)
    total_energy = np.sum(tot_matrix, axis=1)

    # Project energy onto horizontal plane (sky dome side)
    proj_total_energy = total_energy * np.cos(skydome.patch_zeniths)

    sum_energy = np.sum(proj_total_energy)

    # Identify fully visible faces
    fully_visible_mask = np.all(visibility == 1, axis=1)

    # Normalize normals
    normed = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Find dot product with up vector
    up_vec = np.array([0.0, 0.0, 1.0])
    cos_angles = np.dot(normed, up_vec)

    # Among fully visible faces, find the one(s) with the highest upward alignment
    cos_angles[~fully_visible_mask] = -np.inf  # Disqualify non-visible faces
    max_dot = np.max(cos_angles)

    # Find indices of faces with this maximum upward alignment
    valid_indices = np.where(cos_angles == max_dot)[0]

    # Compute mean irradiance on selected face(s)
    max_face_irradiance = np.sum(irradiance[valid_indices, :], axis=1)
    res = np.mean(max_face_irradiance)

    # Max irradiance on any face (for context)
    max_irr = np.max(np.sum(irradiance, axis=1))

    # Report
    info(f"Total sun + sky irradiance on horizontal plane   : {sum_energy:.2f} Wh/m²")
    info(f"Irradiance on the most horizontal visible faces  : {res:.2f} Wh/m²")
    info(f"Max irradiance (any face): {max_irr:.2f} Wh/m²")
    info(f"Found {len(valid_indices)} close to upward-facing visible face(s)")

    return valid_indices
