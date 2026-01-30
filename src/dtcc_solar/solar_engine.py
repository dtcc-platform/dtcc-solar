import math
import numpy as np
from pprint import pp
from time import time

try:  # Attempt to import the compiled BVH-based solar bindings
    from dtcc_solar import py_solar as solar_module
except ImportError as exc:  # pragma: no cover - platform dependent
    solar_module = None
    _SOLAR_IMPORT_ERROR = exc
else:
    _SOLAR_IMPORT_ERROR = None

from dtcc_solar.utils import SolarParameters, concatenate_meshes
from dtcc_solar.utils import OutputCollection, SkyType
from dtcc_solar.utils import Rays, split_mesh_by_face_mask, AnalysisType
from dtcc_solar.skydome import Skydome
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.perez import (
    calc_2_phase_matrices,
    calc_3_phase_matrices,
    calc_3_phase_vector,
    calc_2_phase_vector,
)
from dtcc_core.model import Mesh, Bounds


def _require_solar():
    """Return the solar bindings or raise a clear error when unavailable."""
    if solar_module is None:  # pragma: no cover - executed only when bindings missing
        msg = (
            "Solar ray-tracing bindings (py_solar) are not available. "
            "This typically happens when the extension is not built. "
            "Build dtcc-solar from source. Original import error: "
            f"{_SOLAR_IMPORT_ERROR}"
        )
        raise RuntimeError(msg) from _SOLAR_IMPORT_ERROR
    return solar_module


def _mesh_to_lists(mesh: Mesh | None):
    """Convert Mesh arrays into plain nested lists for pybind11 (or empty if None)."""
    if mesh is None:
        return [], []
    return (
        np.asarray(mesh.vertices, dtype=np.float32).tolist(),
        np.asarray(mesh.faces, dtype=np.int32).tolist(),
    )


class SolarEngine:
    """
    Class for performing solar analysis on 3D meshes.

    - analysis_mesh: faces where results are computed and returned
    - shading_mesh: optional shadow casters (not included in result arrays)
    - scene_mesh: (analysis + shading) used only for bounds / display if shading exists
    """

    def __init__(
        self,
        analysis_mesh: Mesh,
        shading_mesh: Mesh = None,
        center_mesh: bool = False,
    ):
        self.analysis_mesh = analysis_mesh
        self.shading_mesh = shading_mesh

        # For bounds / display we may still want a combined mesh,
        # but we do NOT create face masks anymore.
        if shading_mesh is None:
            self.scene_mesh = analysis_mesh
        else:
            self.scene_mesh = concatenate_meshes([analysis_mesh, shading_mesh])

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
            info(f"  Scene mesh has {len(self.scene_mesh.faces)} faces.")
        info(f"  Mesh moved to center: {center_mesh}")
        info("-----------------------------------------------------")

    def _preprocess_mesh(self, move_to_center: bool):
        """
        Preprocess the scene mesh (analysis + shading if present) for bounds/radius.
        """
        self._calc_bounds()

        # Center mesh based on x and y coordinates only
        center_bb = np.array([np.mean(self.bbx), np.mean(self.bby), np.mean(self.bbz)])
        centerVec = self.origin - center_bb

        if move_to_center:
            # Move the scene mesh (and underlying analysis/shading if they share arrays)
            self.scene_mesh.vertices += centerVec
            info("Scene mesh has been moved to origin.")

        # Recompute after potential move
        self._calc_bounds()

        self.horizon_z = 0.0  # could be np.average(scene_mesh.vertices[:, 2])

        dx = self.bb.width
        dy = self.bb.height
        dz = self.zmax - self.zmin
        self.sunpath_radius = 0.5 * math.sqrt(dx**2 + dy**2 + dz**2)

        self.sun_size = self.sunpath_radius / 90.0
        self.dome_radius = self.sunpath_radius / 40
        self.tolerance = self.sunpath_radius / 1.0e7

    def _calc_bounds(self):
        v = self.scene_mesh.vertices
        self.xmin = v[:, 0].min()
        self.xmax = v[:, 0].max()
        self.ymin = v[:, 1].min()
        self.ymax = v[:, 1].max()
        self.zmin = v[:, 2].min()
        self.zmax = v[:, 2].max()

        self.bbx = np.array([self.xmin, self.xmax])
        self.bby = np.array([self.ymin, self.ymax])
        self.bbz = np.array([self.zmin, self.zmax])

        self.bb = Bounds(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

    def run_analysis(
        self, sunp: Sunpath, skyd: Skydome, p: SolarParameters
    ) -> OutputCollection:
        output = None

        if p.analysis_type == AnalysisType.TWO_PHASE_1D:
            output = self.run_2_phase_analysis_1D(sunp, skyd, p)
        elif p.analysis_type == AnalysisType.TWO_PHASE_2D:
            output = self.run_2_phase_analysis_2D(sunp, skyd, p)
        elif p.analysis_type == AnalysisType.THREE_PHASE_1D:
            output = self.run_3_phase_analysis_1D(sunp, skyd, p)
        elif p.analysis_type == AnalysisType.THREE_PHASE_2D:
            output = self.run_3_phase_analysis_2D(sunp, skyd, p)

        if output is not None:
            output.info_print()

        return output

    # -----------------------
    # 2-PHASE
    # -----------------------

    def _make_cpp_solar_2phase(self, ray_dirs: np.ndarray, solid_angles: np.ndarray):
        solar_mod = _require_solar()

        aV, aF = _mesh_to_lists(self.analysis_mesh)
        sV, sF = _mesh_to_lists(self.shading_mesh)

        rd = np.asarray(ray_dirs, dtype=np.float32).tolist()
        sa = np.asarray(solid_angles, dtype=np.float32).tolist()

        return solar_mod.PySolar(aV, aF, sV, sF, rd, sa)

    def run_2_phase_analysis_1D(
        self, sunpath: Sunpath, skydome: Skydome, p: SolarParameters
    ) -> OutputCollection:
        ss_vector, skyres, sunres = calc_2_phase_vector(sunpath, skydome, p)

        ray_dirs = np.asarray(skydome.ray_dirs, dtype=np.float32)
        solid_angles = np.asarray(skydome.solid_angles, dtype=np.float32)

        info("-----------------------------------------------------")
        info("Creating solar instance and running analysis...")
        info("-----------------------------------------------------")

        # Call the C++ solar constructor
        self.solar = self._make_cpp_solar_2phase(ray_dirs, solid_angles)

        # Run the analysis
        self.solar.run_2_phase_analysis_vec(ss_vector)

        # Retrieve results
        irr_vec = self.solar.get_irradiance_vector()
        runtime = self.solar.get_runtime()

        irr_vec = irr_vec * 0.001  # -> kWh/m2

        outc = OutputCollection(
            analysis_mesh=self.analysis_mesh,  # results correspond to analysis mesh only
            shading_mesh=self.shading_mesh,  # still included for context / rendering
            sky_results=skyres,
            sun_results=sunres,
            total_irradiance=irr_vec,
            runtime=runtime,
        )
        return outc

    def run_2_phase_analysis_2D(
        self, sunpath: Sunpath, skydome: Skydome, p: SolarParameters
    ) -> OutputCollection:
        skyres, sunres = calc_2_phase_matrices(sunpath, skydome, p)
        ss_matrix = sunres.matrix + skyres.matrix

        ray_dirs = np.asarray(skydome.ray_dirs, dtype=np.float32)
        solid_angles = np.asarray(skydome.solid_angles, dtype=np.float32)

        info("-----------------------------------------------------")
        info("Creating solar instance and running analysis...")
        info("-----------------------------------------------------")

        # Call the C++ solar constructor
        self.solar = self._make_cpp_solar_2phase(ray_dirs, solid_angles)

        # Run the analysis
        self.solar.run_2_phase_analysis_mat(ss_matrix)

        # Retrieve results
        start = time()
        irr_vec = self.solar.get_irradiance_matrix_flat()
        runtime = self.solar.get_runtime()
        irr_vec = irr_vec * 0.001  # -> kWh/m2
        end = time()
        info(f"Retrieving irradiance matrix flat took {end - start} seconds.")

        outc = OutputCollection(
            analysis_mesh=self.analysis_mesh,
            shading_mesh=self.shading_mesh,
            sky_results=skyres,
            sun_results=sunres,
            total_irradiance=irr_vec,
            runtime=runtime,
        )
        return outc

    # -----------------------
    # 3-PHASE
    # -----------------------

    def _make_cpp_solar_3phase(
        self,
        sky_ray_dirs: np.ndarray,
        sky_solid_angles: np.ndarray,
        sun_ray_dirs: np.ndarray,
        sun_solid_angles: np.ndarray,
    ):
        solar_mod = _require_solar()

        aV, aF = _mesh_to_lists(self.analysis_mesh)
        sV, sF = _mesh_to_lists(self.shading_mesh)

        rd_sky = np.asarray(sky_ray_dirs, dtype=np.float32).tolist()
        sa_sky = np.asarray(sky_solid_angles, dtype=np.float32).tolist()
        rd_sun = np.asarray(sun_ray_dirs, dtype=np.float32).tolist()
        sa_sun = np.asarray(sun_solid_angles, dtype=np.float32).tolist()

        return solar_mod.PySolar(aV, aF, sV, sF, rd_sky, sa_sky, rd_sun, sa_sun)

    def run_3_phase_analysis_1D(
        self, sunpath: Sunpath, skydome: Skydome, p: SolarParameters
    ) -> OutputCollection:
        sky_vec, sun_vec, skyres, sunres = calc_3_phase_vector(sunpath, skydome, p)

        sky_rd = np.asarray(skydome.ray_dirs, dtype=np.float32)
        sun_rd = np.asarray(sunpath.sunc.sun_vecs, dtype=np.float32)

        sky_sa = np.asarray(skydome.solid_angles, dtype=np.float32)
        sun_sa = np.ones(sunpath.sunc.count, dtype=np.float32)

        info("-----------------------------------------------------")
        info("Creating solar instance and running analysis...")
        info("-----------------------------------------------------")

        # Call the C++ solar constructor
        self.solar = self._make_cpp_solar_3phase(sky_rd, sky_sa, sun_rd, sun_sa)

        # Run the analysis
        self.solar.run_3_phase_analysis_vec(sky_vec, sun_vec)

        # Retrieve results
        sky_irr = self.solar.get_irradiance_vector_sky() * 0.001
        sun_irr = self.solar.get_irradiance_vector_sun() * 0.001
        tot_irr = sky_irr + sun_irr

        sun_hours = self.solar.get_sun_hours()
        svf = self.solar.get_sky_view_factor()
        runtime = self.solar.get_runtime()

        outc = OutputCollection(
            analysis_mesh=self.analysis_mesh,
            shading_mesh=self.shading_mesh,
            sky_results=skyres,
            sun_results=sunres,
            total_irradiance=tot_irr,
            sky_irradiance=sky_irr,
            sun_irradiance=sun_irr,
            runtime=runtime,
            sun_hours=sun_hours,
            sky_view_factor=svf,
        )
        return outc

    def run_3_phase_analysis_2D(
        self, sunpath: Sunpath, skydome: Skydome, p: SolarParameters
    ) -> OutputCollection:
        sky_res, sun_res = calc_3_phase_matrices(sunpath, skydome, p)

        sky_matrix = sky_res.matrix
        sun_matrix = sun_res.matrix

        sky_rd = np.asarray(skydome.ray_dirs, dtype=np.float32)
        sun_rd = np.asarray(sunpath.sunc.sun_vecs, dtype=np.float32)

        sky_sa = np.asarray(skydome.solid_angles, dtype=np.float32)
        sun_sa = np.ones(sunpath.sunc.count, dtype=np.float32)

        info("-----------------------------------------------------")
        info("Creating solar instance and running analysis...")
        info("-----------------------------------------------------")

        # Call the C++ solar constructor
        self.solar = self._make_cpp_solar_3phase(sky_rd, sky_sa, sun_rd, sun_sa)

        # Run the analysis
        self.solar.run_3_phase_analysis_mat(sky_matrix, sun_matrix)

        # Retrieve results
        sky_irr = self.solar.get_irradiance_matrix_sky_flat() * 0.001
        sun_irr = self.solar.get_irradiance_matrix_sun_flat() * 0.001
        tot_irr = sky_irr + sun_irr

        sun_hours = self.solar.get_sun_hours()
        svf = self.solar.get_sky_view_factor()
        runtime = self.solar.get_runtime()

        outc = OutputCollection(
            analysis_mesh=self.analysis_mesh,
            shading_mesh=self.shading_mesh,
            sky_results=sky_res,
            sun_results=sun_res,
            total_irradiance=tot_irr,
            sky_irradiance=sky_irr,
            sun_irradiance=sun_irr,
            runtime=runtime,
            sun_hours=sun_hours,
            sky_view_factor=svf,
        )
        return outc

    # -----------------------
    # ENERGY BALANCE CHECKS
    # -----------------------

    def check_2_phase_energy_balance(
        self,
        skydome: Skydome,
        tot_mat: np.ndarray,
        irr_vec: np.ndarray,
        vis_mat: np.ndarray,
        face_normals: np.ndarray,
    ):
        """
        Same idea as before, but no face-mask filtering.
        If you still have shading faces in vis_mat/face_normals, pass only analysis-face arrays in.
        """

        total_energy = np.sum(tot_mat, axis=1)
        proj_total_energy = total_energy * np.cos(skydome.patch_zeniths)
        sum_energy = np.sum(proj_total_energy) * 0.001  # kWh/m2

        # Fully visible faces
        all_visible = np.all(vis_mat == 1, axis=1)

        up_vec = np.array([0.0, 0.0, 1.0])
        cos_angles = np.dot(face_normals, up_vec)
        cos_angles[~all_visible] = -np.inf

        max_dot = np.max(cos_angles)
        valid_indices = np.where(cos_angles == max_dot)[0]

        irr_values = irr_vec[valid_indices]
        res = np.max(irr_values)

        max_irr = np.max(irr_vec)

        # Optional debug split (still useful)
        mesh_in, mesh_out = split_mesh_by_face_mask(self.analysis_mesh, all_visible)

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
        patch_zeniths = np.array(skydome.patch_zeniths)
        sun_zeniths = np.array(sunpath.sunc.zeniths)

        sky_energy = np.sum(sky_matrix, axis=1) * patch_zeniths
        sun_energy = np.sum(sun_matrix, axis=1) * sun_zeniths
        sum_energy = np.sum(sky_energy + sun_energy)

        fully_visible_sky = np.all(sky_vis == 1, axis=1)
        fully_visible_sun = np.all(sun_vis == 1, axis=1)
        fully_visible_mask = fully_visible_sky & fully_visible_sun

        normed_normals = face_normals / np.linalg.norm(
            face_normals, axis=1, keepdims=True
        )
        up_vector = np.array([0.0, 0.0, 1.0])
        cos_up = np.dot(normed_normals, up_vector)

        cos_up[~fully_visible_mask] = -np.inf
        max_up = np.max(cos_up)
        valid_indices = np.where(cos_up == max_up)[0]

        sky_face_irr = np.sum(sky_irr[valid_indices, :], axis=1)
        sun_face_irr = np.sum(sun_irr[valid_indices, :], axis=1)
        hor_faces_mean_irr = np.mean(sky_face_irr + sun_face_irr)

        all_faces_max_irr = np.max(np.sum(sky_irr, axis=1) + np.sum(sun_irr, axis=1))

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
