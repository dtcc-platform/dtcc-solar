import os
from pathlib import Path

# import dtcc
from dtcc_core import io as io
from dtcc_core.io import load_city
from dtcc_core.model import PointCloud, Mesh
from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer, SkydomeViewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.reinhart4 import ReinhartM4
from dtcc_solar.perez import *
from dtcc_solar.radiance import calc_radiance_matrices
from dtcc_solar.synthetic_data import synthetic_epw_df, df_to_epw

import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlretrieve


def _candidate_data_roots():
    """Yield plausible roots for the ``data`` folder in priority order."""

    roots = []

    env_root = os.environ.get("DTCC_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())

    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir, *script_dir.parents[:4]]:
        roots.append(parent / "data")

    roots.append(Path.cwd() / "data")

    seen = set()
    ordered = []
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _find_data_path(relative: Path, expect_directory: bool | None = None) -> Path:
    roots = _candidate_data_roots()
    for root in roots:
        candidate = root / relative
        if expect_directory is None and candidate.exists():
            return candidate
        if expect_directory is True and candidate.is_dir():
            return candidate
        if expect_directory is False and candidate.is_file():
            return candidate

    kind = "directory" if expect_directory else "file"
    searched = ", ".join(str(root / relative) for root in roots)
    raise FileNotFoundError(
        f"Could not locate {kind} '{relative}'. Set DTCC_DATA_ROOT to the project "
        f"data folder or place the resources manually. Searched: {searched}"
    )


def data_file(*parts: str) -> Path:
    return _find_data_path(Path(*parts), expect_directory=False)


def data_dir(*parts: str) -> Path:
    return _find_data_path(Path(*parts), expect_directory=True)


def only_perez_test():
    path_lnd = data_file("weather", "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw")

    p = SolarParameters(
        weather_file=str(path_lnd),
        analysis_type=AnalysisType.TWO_PHASE,
        start=pd.Timestamp("2019-01-01 12:00:00"),
        end=pd.Timestamp("2019-12-02 12:00:00"),
    )

    # skydome = Reinhart()
    skydome = Tregenza()
    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    (sky_res, sun_res) = calc_2_phase_matrices(sunpath, skydome, p.sun_mapping)

    face_data_dict = {
        "relative lumiance": sky_res.relative_luminance,
        "sky matrix": sky_res.matrix,
        "sun matrix": sun_res.matrix,
        "total matrix": sky_res.matrix + sun_res.matrix,
        "solid angles": sky_res.solid_angles,
        "ksis": sky_res.ksis,
        "gammas": sky_res.gammas,
    }

    sun_pc = PointCloud(points=sunpath.sunc.positions)
    viewer = SkydomeViewer(skydome, face_data_dict, sun_pc)


def synthetic_data_test():
    df, header = synthetic_epw_df()
    export_path = data_dir("weather") / "synthetic.epw"
    # df_to_epw(df, header, export_path)
    # print("Synthetic EPW written with shape:", df.shape)

    # Print first few rows
    print(df.head())


def radiance_test():

    # Set path for weather file
    path = data_file("weather", "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw")
    # Set path for radiance installation
    rad_path = "/usr/local/radiance/bin/"

    rad_sky, rad_sun, rad_tot = calc_radiance_matrices(
        str(path), sky_type=SkyType.REINHART_578, rad_path=rad_path
    )

    p = SolarParameters(weather_file=str(path))

    skydome = ReinhartM2()
    sunpath = Sunpath(p, include_night=True)

    (sky_res, sun_res) = calc_2_phase_matrices(sunpath, skydome, p)

    dtcc_sky = sky_res.matrix
    dtcc_sun = sun_res.matrix
    dtcc_total = dtcc_sky + dtcc_sun

    # Comparing the data
    rad_sky_tot = np.sum(rad_sky)
    rad_sun_tot = np.sum(rad_sun)
    rad_tot = np.sum(rad_tot)

    dtcc_sky_tot = np.sum(dtcc_sky)
    dtcc_sun_tot = np.sum(dtcc_sun)
    dtcc_total = dtcc_sky_tot + dtcc_sun_tot

    sky_diff = math.fabs(rad_sky_tot - dtcc_sky_tot)
    sun_diff = math.fabs(rad_sun_tot - dtcc_sun_tot)
    total_diff = math.fabs(rad_tot - dtcc_total)

    info("-----------------------------------------------------")
    info(f"sky error:  {100 * sky_diff / rad_sky_tot} %")
    info(f"sun error:  {100 * sun_diff / rad_sun_tot} %")
    info(f"tot error:  {100 * total_diff / rad_tot} %")
    info("-----------------------------------------------------")

    rad_sky_patch = np.sum(rad_sky, axis=1)
    dtcc_sky_patch = np.sum(dtcc_sky, axis=1)

    rad_sun_patch = np.sum(rad_sun, axis=1)
    dtcc_sun_patch = np.sum(dtcc_sun, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Sky
    axes[0].plot(rad_sky_patch, label="Radiance sky")
    axes[0].plot(dtcc_sky_patch, label="DTCC sky")
    axes[0].set_title("Sky")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sun
    axes[1].plot(rad_sun_patch, label="Radiance sun")
    axes[1].plot(dtcc_sun_patch, label="DTCC sun")
    axes[1].set_title("Sun")
    axes[1].set_xlabel("Patch index")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyse_mesh_1():
    # filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    # filename = data_file("models", "City136kSoft.stl")
    filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))
    engine = SolarEngine(mesh)

    weather_dir = data_dir("weather")
    sth_epw = weather_dir / "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Stockholm
    p = SolarParameters(
        weather_file=str(sth_epw),
        analysis_type=AnalysisType.TWO_PHASE,
        sun_mapping=SunMapping.SMOOTH_SMEAR,
        start=pd.Timestamp("2019-07-15 16:00:00"),
        end=pd.Timestamp("2019-07-15 17:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(sunpath, skydome, p)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sunpath, p)


def analyse_mesh_2():

    filename = data_file("validation", "boxes_sharp_f5248.obj")
    mesh = io.load_mesh(str(filename))
    weather_dir = data_dir("weather")
    gbg_epw = weather_dir / "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"

    # Gothenburg
    p = SolarParameters(
        weather_file=str(gbg_epw),
        analysis_type=AnalysisType.TWO_PHASE,
        sun_mapping=SunMapping.RADIANCE,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    (analysis_mesh, shading_mesh) = split_mesh_by_vertical_faces(mesh)

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(sunpath, skydome, p)
    viewer = Viewer(output, skydome, sunpath, p)


def analyse_mesh_3():
    # filename = "...../../data/validation/boxes_sharp_f5248.obj"
    filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))

    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.3, 0.9], [0.3, 0.9])
    engine = SolarEngine(analysis_mesh, shading_mesh)

    weather_dir = data_dir("weather")
    lnd_epw = weather_dir / "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    # London
    p = SolarParameters(
        weather_file=str(lnd_epw),
        analysis_type=AnalysisType.TWO_PHASE,
        sun_mapping=SunMapping.NONE,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM4()
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(sunpath, skydome, p)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sunpath, p)


def analyse_mesh_4():
    print("-------- Solar Mesh Analysis Started -------")
    filename = data_file("validation", "boxes_sharp_f5248.obj")
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    weather_dir = data_dir("weather")
    sth_epw = weather_dir / "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    mesh = io.load_mesh(str(filename))
    engine = SolarEngine(mesh)

    # Stockholm
    p = SolarParameters(
        weather_file=str(sth_epw),
        analysis_type=AnalysisType.THREE_PHASE,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(sunpath, skydome, p)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sunpath, p)


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")
    info("#################### DTCC-SOLAR #####################")

    # only_perez_test()
    # radiance_test()
    # synthetic_data_test()
    analyse_mesh_1()
    # analyse_mesh_2()
    # analyse_mesh_3()
    # analyse_mesh_4()
