import math
import numpy as np
from dtcc_solar.utils import SolarParameters, calc_face_mid_points, concatenate_meshes
from dtcc_solar import py_embree_solar as embree
from dtcc_solar.sundome import SunDome
from dtcc_solar.utils import SunCollection, OutputCollection, SunApprox, Sky
from dtcc_solar.utils import Rays
from dtcc_solar.sunpath import Sunpath
from dtcc_model import Mesh, PointCloud
from dtcc_model.geometry import Bounds
from dtcc_solar.logging import info, debug, warning, error
from dtcc_viewer import Window, Scene
from dtcc_solar.sungroups import SunGroups
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
        sky: Sky = None,
        rays: Rays = None,
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
        self.sky = sky
        self.rays = rays
        (self.mesh, self.face_mask) = self._join_meshes(analysis_mesh, shading_mesh)
        self.origin = np.array([0, 0, 0])
        self.horizon_z = 0
        self.sunpath_radius = 0
        self.sun_size = 0
        self.dome_radius = 0
        self.path_width = 0
        self._preprocess_mesh(center_mesh)

        if sky is None:
            self.sky = Sky.EqualArea320

        if rays is None:
            self.rays = Rays.Bundle8

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

    def get_skydome_ray_count(self):
        """
        Get the count of skydome rays.

        Returns
        -------
        int
            The count of skydome rays.
        """
        return self.embree.get_skydome_ray_count()

    def run_analysis(self, p: SolarParameters, sunp: Sunpath, outc: OutputCollection):
        """
        Run the solar analysis with the provided parameters and sunpath.

        Parameters
        ----------
        p : SolarParameters
            Parameters for the solar analysis.
        sunp : Sunpath
            Sunpath data for the analysis.
        outc : OutputCollection
            Output collection for storing the results.
        """
        # Creating instance of embree solar
        info(f"Creating embree instance")
        self.embree = embree.PyEmbreeSolar(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_mask,
            int(self.sky),
        )
        info(f"Running analysis...")

        # Run raytracing
        if p.sun_analysis:
            if p.sun_approx == SunApprox.quad and sunp.sundome is not None:
                self._sun_quad_raycasting(sunp.sundome, sunp.sunc)
            elif p.sun_approx == SunApprox.group and sunp.sungroups is not None:
                self._sun_group_raycasting(sunp.sungroups, sunp.sunc)
            elif p.sun_approx == SunApprox.none:
                self._sun_raycasting(sunp.sunc)

        if p.sky_analysis:
            self._sky_raycasting()

        # Calculate irradiance base on raytracing results and weather data
        if p.sun_approx == SunApprox.none:
            self.embree.calc_irradiance(sunp.sunc.dni, sunp.sunc.dhi)
        elif p.sun_approx == SunApprox.group:
            indices = sunp.sungroups.sun_indices
            self.embree.calc_irradiance_group(sunp.sunc.dni, sunp.sunc.dhi, indices)
        elif p.sun_approx == SunApprox.quad:
            active_quads = sunp.sundome.active_quads
            indices = [quad.sun_indices for quad in active_quads]
            self.embree.calc_irradiance_group(sunp.sunc.dni, sunp.sunc.dhi, indices)

        # Save results to output collection
        if p.sun_analysis:
            n_suns = sunp.sunc.count
            outc.dni = self.embree.get_results_dni()
            outc.face_sun_angles = self.embree.get_accumulated_angles() / n_suns
            outc.sun_hours = n_suns - self.embree.get_accumulated_occlusion()
            outc.shadow_hours = self.embree.get_accumulated_occlusion()

        if p.sky_analysis:
            outc.dhi = self.embree.get_results_dhi()
            outc.facehit_sky = self.embree.get_face_skyhit_results()
            outc.sky_view_factor = self.embree.get_sky_view_factor_results()

        outc.process_results(p.sun_analysis, p.sky_analysis, self.face_mask)

    def _sun_group_raycasting(self, sungroups: SunGroups, sunc: SunCollection):
        """
        Perform sun group raycasting analysis.

        Parameters
        ----------
        sungroups : SunGroups
            Sun groups data for the analysis.
        sunc : SunCollection
            Sun collection data for the analysis.
        """
        sun_vecs = sungroups.list_centers
        info(f"Anlysing {sunc.count} suns in {len(sun_vecs)} sun groups.")
        if self._sun_raycasting_embree(sun_vecs):
            info(f"Raytracing sun groups completed successfully.")

    def _sun_quad_raycasting(self, sundome: SunDome, sunc: SunCollection):
        """
        Perform sun quad raycasting analysis.

        Parameters
        ----------
        sundome : SunDome
            Sundome data for the analysis.
        sunc : SunCollection
            Sun collection data for the analysis.
        """
        sundome.match_suns_and_quads(sunc)
        sun_vecs = sundome.get_active_quad_centers()
        info(f"Anlysing {sunc.count} suns in {len(sun_vecs)} quad groups.")
        if self._sun_raycasting_embree(sun_vecs):
            info(f"Raytracing sun groups completed successfully.")

    def _sun_raycasting(self, sunc: SunCollection):
        """
        Perform sun raycasting analysis.

        Parameters
        ----------
        sunc : SunCollection
            Sun collection data for the analysis.
        """
        info(f"Running sun raycasting")
        if self._sun_raycasting_embree(sunc.sun_vecs):
            info(f"Running sun raycasting completed successfully.")

    def _sky_raycasting(self):
        """Perform sky raycasting analysis."""
        info(f"Running sky raycasting")
        if self._sky_raycasting_embree():
            info(f"Running sky raycasting completed succefully.")

    def _sun_raycasting_embree(self, sun_vecs) -> bool:
        """
        Perform sun raycasting using the Embree library.

        Parameters
        ----------
        sun_vecs : list
            List of sun vectors for the analysis.

        Returns
        -------
        bool
            True if the raycasting was successful, False otherwise.
        """
        success = False
        if self.rays == Rays.Bundle1:
            success = self.embree.sun_raytrace_occ1(sun_vecs)
        elif self.rays == Rays.Bundle4:
            success = self.embree.sun_raytrace_occ4(sun_vecs)
        elif self.rays == Rays.Bundle8:
            success = self.embree.sun_raytrace_occ8(sun_vecs)
        elif self.rays == Rays.Bundle16:
            success = self.embree.sun_raytrace_occ16(sun_vecs)

        if not success:
            error("Something went wrong with sun raycasting in embree.")

        return success

    def _sky_raycasting_embree(self) -> bool:
        """
        Perform sky raycasting using the Embree library.

        Returns
        -------
        bool
            True if the raycasting was successful, False otherwise.
        """
        success = False
        if self.rays == Rays.Bundle1:
            success = self.embree.sky_raytrace_occ1()
        elif self.rays == Rays.Bundle4:
            success = self.embree.sky_raytrace_occ4()
        elif self.rays == Rays.Bundle8:
            success = self.embree.sky_raytrace_occ8()
        elif self.rays == Rays.Bundle16:
            success = self.embree.sky_raytrace_occ16()

        if not success:
            error("Something went wrong with sky raycasting in embree.")

        return success

    def view_results(
        self,
        p: SolarParameters,
        sunpath: Sunpath,
        outc: OutputCollection,
    ):
        """
        View the results of the solar analysis.

        Parameters
        ----------
        p : SolarParameters
            Parameters for the solar analysis.
        sunpath : Sunpath
            Sunpath data for the analysis.
        outc : OutputCollection
            Output collection containing the results.
        """
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath, p)
        viewer.add_mesh("Analysed mesh", mesh=self.analysis_mesh, data=outc.data_1)
        if self.shading_mesh is not None:
            viewer.add_mesh("Shading mesh", mesh=self.shading_mesh, data=outc.data_2)
        viewer.show()
