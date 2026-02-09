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
from dtcc_solar.reinhart6 import ReinhartM6
from dtcc_solar.reinhart8 import ReinhartM8
from dtcc_solar.reinhartMF import ReinhartMF
from dtcc_solar.perez import *
from dtcc_solar.radiance import calc_radiance_matrices
from dtcc_solar.synthetic_data import synthetic_epw_df, df_to_epw

import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlretrieve
from time import time
from pprint import pprint


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
        dim=Dim.ONE_D,
        start=pd.Timestamp("2019-01-01 12:00:00"),
        end=pd.Timestamp("2019-12-02 12:00:00"),
    )

    skydome = Tregenza()
    sundome = Tregenza()
    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    (sky_res, sun_res) = calc_sky_sun_matrices(sunpath, skydome, sundome, p)

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
    filename = "../../../data/validation/boxes_soft_f5248.obj"
    mesh = io.load_mesh(str(filename))
    engine = SolarEngine(mesh)
    df, header = synthetic_epw_df()
    export_path = data_dir("weather") / "synthetic.epw"
    df_to_epw(df, header, export_path)
    info("Synthetic EPW written with shape:", df.shape)

    weather_dir = data_dir("weather")
    synt_epw = weather_dir / "synthetic.epw"

    # Stockholm
    p = SolarParameters(
        weather_file=str(synt_epw),
        is1D=True,
        compute_sh=True,
        compute_svf=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    sunpath = Sunpath(p, engine.sunpath_radius)

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sundome = ReinhartM2()
    output = engine.run_analysis(p, sunpath, skydome, sundome)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sundome, sunpath, p)


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
    sundome = ReinhartM2()
    sunpath = Sunpath(p, include_night=True)

    (sky_res, sun_res) = calc_sky_sun_matrices(sunpath, skydome, sundome)

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
    filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    # filename = data_file("models", "City136kSoft.stl")
    # filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))
    engine = SolarEngine(mesh)

    weather_dir = data_dir("weather")
    sth_epw = weather_dir / "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Stockholm
    p = SolarParameters(
        weather_file=str(sth_epw),
        is1D=True,
        compute_sh=True,
        compute_svf=True,
        start=pd.Timestamp("2019-07-15 00:00:00"),
        end=pd.Timestamp("2019-07-15 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sundome = ReinhartM2()
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(p, sunpath, skydome, sundome)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sundome, sunpath, p)


def analyse_mesh_2():
    filename = data_file("validation", "boxes_sharp_f5248.obj")
    mesh = io.load_mesh(str(filename))
    weather_dir = data_dir("weather")
    gbg_epw = weather_dir / "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"

    # Gothenburg
    p = SolarParameters(
        weather_file=str(gbg_epw),
        is1D=True,
        compute_sh=True,
        compute_svf=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    (analysis_mesh, shading_mesh, mask) = split_mesh_by_vertical_faces(mesh)

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sundome = ReinhartM2()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    output = engine.run_analysis(p, sunpath, skydome, sundome)
    viewer = Viewer(output, skydome, sundome, sunpath, p)


def analyse_mesh_3():
    # filename = "...../../data/validation/boxes_sharp_f5248.obj"
    filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))
    lengths, face_counts = subdivision_lengths_for_targets(mesh, [1e4])
    mesh = subdivide_mesh(mesh, lengths[0])
    start_time = time()
    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.3, 0.9], [0.3, 0.9])
    engine = SolarEngine(analysis_mesh, shading_mesh)
    print("Face count mesh: ", len(analysis_mesh.faces))
    weather_dir = data_dir("weather")
    lnd_epw = weather_dir / "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    # London
    p = SolarParameters(
        weather_file=str(lnd_epw),
        is1D=False,
        compute_sh=False,
        compute_svf=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    sunpath = Sunpath(p, engine.sunpath_radius)
    skydome = ReinhartMF(2)
    sundome = ReinhartMF(2)
    output = engine.run_analysis(p, sunpath, skydome, sundome)
    end_time = time()
    print("Analysis time (s): ", end_time - start_time)
    export_path = data_dir("validation") / "export_test.json"
    export_to_json(output, p, export_path)
    viewer = Viewer(output, skydome, sundome, sunpath, p)


def analyse_time_test():
    filename = data_file("validation", "boxes_soft_f5248.obj")
    base_mesh = io.load_mesh(str(filename))
    sub_dom = [0.3, 0.9]
    (a_mesh, s_mesh) = split_mesh_with_domain(base_mesh, sub_dom, sub_dom)
    # Choose targets (log or linear)
    targets = np.linspace(1e4, 1e5, num=6, dtype=int)
    lengths, face_counts = subdivision_lengths_for_targets(a_mesh, targets)

    pprint({"Lengths": lengths})
    pprint({"Face counts": face_counts})

    dims = [True, False]

    # Store results per type
    results = {
        t: {
            "faces": [],
            "python_total_time": [],
            "cpp_total_time": [],
            "multiplication_time": [],
            "raytracing_time": [],
        }
        for t in dims
    }

    weather_dir = data_dir("weather")
    lnd_epw = weather_dir / "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    for dim in dims:
        for i, length in enumerate(lengths):

            (a_mesh, s_mesh) = split_mesh_with_domain(base_mesh, sub_dom, sub_dom)
            a_mesh = subdivide_mesh(a_mesh, length)
            f_count = len(a_mesh.faces)

            print(f"# Target: {targets[i]}, length {length}, count {f_count} #")

            start_time = time()
            engine = SolarEngine(a_mesh, s_mesh)

            p = SolarParameters(
                weather_file=str(lnd_epw),
                is1D=dim,
                compute_sh=True,
                compute_svf=True,
                start=pd.Timestamp("2019-01-01 00:00:00"),
                end=pd.Timestamp("2019-12-31 23:00:00"),
            )

            skydome = ReinhartM2()
            sundome = ReinhartM4()
            sunpath = Sunpath(p, engine.sunpath_radius)
            output = engine.run_analysis(p, sunpath, skydome, sundome)

            elapsed = time() - start_time

            results[dim]["faces"].append(f_count)
            results[dim]["python_total_time"].append(elapsed)
            results[dim]["cpp_total_time"].append(output.runtime[2])
            results[dim]["multiplication_time"].append(output.runtime[1])
            results[dim]["raytracing_time"].append(output.runtime[0])

    # ---- Plot ----
    plot_timings_vs_faces(results, dims)
    return results


def analyse_convergence():
    filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))
    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.3, 0.9], [0.3, 0.9])
    engine = SolarEngine(analysis_mesh, shading_mesh)

    lengths, face_counts = subdivision_lengths_for_targets(analysis_mesh, [1e4])
    analysis_mesh = subdivide_mesh(analysis_mesh, lengths[0])

    sundomes = {}
    sundomes["NaturalSuns"] = None
    sundomes["Tregenza"] = Tregenza()
    sundomes["ReinhartM2"] = ReinhartM2()
    sundomes["ReinhartM4"] = ReinhartM4()
    sundomes["ReinhartM6"] = ReinhartM6()
    sundomes["ReinhartM8"] = ReinhartM8()
    sundomes["ReinhartM10"] = ReinhartMF(10)
    sundomes["ReinhartM12"] = ReinhartMF(12)

    weather_dir = data_dir("weather")
    lnd_epw = weather_dir / "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    results_sh = {}
    results_irr = {}

    for key, sundome in sundomes.items():
        engine = SolarEngine(analysis_mesh, shading_mesh)

        p = SolarParameters(
            weather_file=str(lnd_epw),
            is1D=True,
            compute_sh=True,
            compute_svf=True,
            start=pd.Timestamp("2019-01-01 00:00:00"),
            end=pd.Timestamp("2019-12-31 23:00:00"),
        )

        skydome = ReinhartM2()
        sunpath = Sunpath(p, engine.sunpath_radius)
        output = engine.run_analysis(p, sunpath, skydome, sundome)
        results_sh.setdefault(key, {})["sun_hours"] = output.sun_hours
        results_irr.setdefault(key, {})["irradiance"] = output.sun_irradiance

    # ---- Sort + Plot ----
    sorted_sh, order, sort_key = sort_faces_by_key_value(
        results_sh, field="sun_hours", baseline="NaturalSuns"
    )
    sorted_irr, order, sort_key = sort_faces_by_key_value(
        results_irr, field="irradiance", baseline="NaturalSuns"
    )
    plot_values_per_face(sorted_sh, field="sun_hours", title="Sun hours per face")

    plot_deltas(
        sorted_sh,
        baseline="NaturalSuns",
        key="sun_hours",
        title="Sun hours convergence",
        step=1,
        ylabel="Sun hours (h)",
    )

    plot_values_per_face(sorted_irr, field="irradiance", title="Irradiance per face")

    plot_deltas(
        sorted_irr,
        baseline="NaturalSuns",
        key="irradiance",
        title="Irradiance convergence",
        step=1,
        ylabel="Irradiance (kW/m²)",
    )


def analyse_all_modes():
    print("-------- Solar Mesh Analysis Started -------")
    filename = data_file("validation", "boxes_soft_f5248.obj")
    mesh = io.load_mesh(str(filename))
    lengths, face_counts = subdivision_lengths_for_targets(mesh, [1e4])
    mesh = subdivide_mesh(mesh, lengths[0])
    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.3, 0.9], [0.3, 0.9])
    engine = SolarEngine(analysis_mesh, shading_mesh)

    list_is_1D = [True, False]
    dicretisation = [[2, None], [2, 2], [2, 2], [2, 4], [2, 4]]
    list_compute_sh = [True, False, True, True, True]
    list_compute_svf = [True, True, True, True, True]

    weather_dir = data_dir("weather")
    lnd_epw = weather_dir / "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    p = SolarParameters(
        weather_file=str(lnd_epw),
        start=pd.Timestamp("2019-07-01 00:00:00"),
        end=pd.Timestamp("2019-07-31 23:00:00"),
    )

    results = {}
    order = None  # will be set from the first run
    counter = 0

    for is1D in list_is_1D:
        for i in range(len(list_compute_sh)):
            compute_sh = list_compute_sh[i]
            compute_svf = list_compute_svf[i]

            p.is1D = is1D
            p.compute_sh = compute_sh
            p.compute_svf = compute_svf

            skydome = ReinhartMF(dicretisation[i][0])
            sundome = (
                None if dicretisation[i][1] is None else ReinhartMF(dicretisation[i][1])
            )

            sunpath = Sunpath(p, engine.sunpath_radius)
            output = engine.run_analysis(p, sunpath, skydome, sundome)

            y = np.asarray(output.total_irradiance, dtype=float).ravel()

            # --- first run defines the face order ---
            if order is None:
                order = np.argsort(y)  # ascending (small -> large)
                # if you want descending instead: order = np.argsort(y)[::-1]

            # --- apply the same order to every run ---
            y_sorted = y[order]

            log = output.analysis_log
            key = f"{log} compute_sh={compute_sh} is1D={is1D} {counter}"
            results[key] = y_sorted
            counter += 1

    plot_results(results)
    return results, order


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")
    info("#################### DTCC-SOLAR #####################")

    # only_perez_test()
    # radiance_test()
    # synthetic_data_test()
    # analyse_mesh_1()
    # analyse_mesh_2()
    analyse_mesh_3()
    # analyse_time_test()
    # analyse_convergence()
    # analyse_all_modes()
