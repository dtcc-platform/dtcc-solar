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

from dtcc_solar.utils import Dim, SolarParameters, concatenate_meshes
from dtcc_solar.utils import OutputCollection, SkyType
from dtcc_solar.utils import Rays, split_mesh_by_face_mask, AnalysisType
from dtcc_solar.dome import Dome
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.perez import calc_sky_sun_matrices
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


def _as_ray_dirs(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {a.shape}")
    return a


def _as_solid_angles(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).ravel()
    if a.size != n:
        raise ValueError(f"{name} must have length {n}; got {a.size}")
    return a


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

    def _make_cpp(
        self,
        sky_ray_dirs: np.ndarray,
        sky_solid_angles: np.ndarray,
        sun_ray_dirs: np.ndarray | None,
        sun_solid_angles: np.ndarray | None,
    ):
        """
        Build the C++ DtccSolar instance.

        Supports:
          - sundome mode: sun_ray_dirs = (P,3), sun_solid_angles = (P,)
          - true-sun mode: sun_ray_dirs = (T,3), sun_solid_angles = (T,) (or ones)

        Note: the *caller* must ensure sun_matrix rows == len(sun_ray_dirs) and
              active_idx indices refer to [0..len(sun_ray_dirs)-1].
        """
        solar_mod = _require_solar()

        aV, aF = _mesh_to_lists(self.analysis_mesh)
        sV, sF = _mesh_to_lists(self.shading_mesh)

        # --- Sky ---
        rd_sky_np = _as_ray_dirs(sky_ray_dirs, "sky_ray_dirs")
        sa_sky_np = _as_solid_angles(
            sky_solid_angles, rd_sky_np.shape[0], "sky_solid_angles"
        )

        # --- Sun ---
        if sun_ray_dirs is None or (
            isinstance(sun_ray_dirs, (list, tuple)) and len(sun_ray_dirs) == 0
        ):
            raise ValueError(
                "sun_ray_dirs must be provided (sundome rays or true-sun rays)."
            )

        rd_sun_np = _as_ray_dirs(sun_ray_dirs, "sun_ray_dirs")

        if sun_solid_angles is None or (
            isinstance(sun_solid_angles, (list, tuple)) and len(sun_solid_angles) == 0
        ):
            # sensible default if caller doesn't provide: all ones
            sa_sun_np = np.ones(rd_sun_np.shape[0], dtype=np.float32)
        else:
            sa_sun_np = _as_solid_angles(
                sun_solid_angles, rd_sun_np.shape[0], "sun_solid_angles"
            )

        # Convert to Python lists for pybind11
        rd_sky = rd_sky_np.tolist()
        sa_sky = sa_sky_np.tolist()
        rd_sun = rd_sun_np.tolist()
        sa_sun = sa_sun_np.tolist()

        return solar_mod.PySolar(aV, aF, sV, sF, rd_sky, sa_sky, rd_sun, sa_sun)

    def run_analysis(
        self,
        p: SolarParameters,
        sunpath: Sunpath,
        skydome: Dome,
        sundome: Dome = None,
    ) -> OutputCollection:

        skyres, sunres = calc_sky_sun_matrices(sunpath, skydome, sundome)
        sun_mat = np.asarray(sunres.matrix, dtype=np.float32)
        sky_mat = np.asarray(skyres.matrix, dtype=np.float32)
        idx = np.asarray(sunres.active_idx, dtype=np.int32)

        # --- Sky rays (always from skydome) ---
        skydome_rd = np.asarray(skydome.ray_dirs, dtype=np.float32)
        skydome_sa = np.asarray(skydome.solid_angles, dtype=np.float32)

        if sundome is None:
            # Natural-sun mode: one ray per timestep
            sun_dirs = np.asarray(sunpath.sunc.sun_vecs, dtype=np.float32)
            sun_sa = np.ones(len(sun_dirs), dtype=np.float32)  # Use unit solid angles
            info(f"Using natural-sun rays")
        else:
            # Sundome mode
            sun_dirs = np.asarray(sundome.ray_dirs, dtype=np.float32)
            sun_sa = np.asarray(sundome.solid_angles, dtype=np.float32)
            info(f"Using sundome rays")

        info("-----------------------------------------------------")
        info("Creating solar instance and running analysis...")
        info("-----------------------------------------------------")

        # Call the C++ solar constructor
        self.solar = self._make_cpp(skydome_rd, skydome_sa, sun_dirs, sun_sa)

        # Run the analysis
        self.solar.analyse(sky_mat, sun_mat, idx, p.is1D, p.compute_sh, p.compute_svf)

        if p.is1D:
            irr_vec = self.solar.get_irradiance_vector()
            irr_vec_sky = self.solar.get_irradiance_vector_sky()
            irr_vec_sun = self.solar.get_irradiance_vector_sun()
        else:
            irr_vec = self.solar.get_irradiance_matrix_flat()
            irr_vec_sky = self.solar.get_irradiance_matrix_sky_flat()
            irr_vec_sun = self.solar.get_irradiance_matrix_sun_flat()

        irr_vec = irr_vec * 0.001  # -> kWh/m2
        irr_vec_sky = irr_vec_sky * 0.001  # -> kWh/m2
        irr_vec_sun = irr_vec_sun * 0.001  # -> kWh/m2

        runtime = self.solar.get_runtime()
        sun_hours = self.solar.get_sun_visible_rays()
        svf = self.solar.get_sky_view_factor()

        outc = OutputCollection(
            analysis_mesh=self.analysis_mesh,
            shading_mesh=self.shading_mesh,
            sky_results=skyres,
            sun_results=sunres,
            sky_view_factor=svf,
            sun_hours=sun_hours,
            total_irradiance=irr_vec,
            sky_irradiance=irr_vec_sky,
            sun_irradiance=irr_vec_sun,
            runtime=runtime,
        )
        return outc

    # -----------------------
    # ENERGY BALANCE CHECKS
    # -----------------------

    def check_2_phase_energy_balance(
        self,
        skydome: Dome,
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
        skydome: Dome,
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
