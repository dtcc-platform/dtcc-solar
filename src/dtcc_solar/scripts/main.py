import os
import dtcc
from dtcc_core import io as io

from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.reinhart4 import ReinhartM4
from dtcc_core.io import load_city
from dtcc_core.model import PointCloud
from dtcc_solar.perez import *
from dtcc_solar.city_utils import *
from dtcc_solar.radiance import calc_radiance_matrices
from dtcc_viewer import Window, Scene
from dtcc_core.model import PointCloud, Mesh
from pprint import pp

import matplotlib.pyplot as plt


import numpy as np
from urllib.request import urlretrieve


def only_perez_test():
    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    p = SolarParameters(
        weather_file=path_lnd,
        analysis_type=AnalysisType.TWO_PHASE,
        start=pd.Timestamp("2019-01-01 12:00:00"),
        end=pd.Timestamp("2019-12-02 12:00:00"),
    )

    # skydome = Reinhart()
    skydome = Tregenza()
    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    (sky_res, sun_res) = calc_2_phase_matrix(sunpath, skydome, p.sun_mapping)

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

    skydome.view(name="Skydome", data_dict=face_data_dict, sun_pos_pc=sun_pc)


def embree_perez_test():
    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    # filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    filename = "../../../data/models/CitySurfaceS.stl"

    mesh = io.load_mesh(filename)

    p = SolarParameters(
        weather_file=path_lnd,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # skydome = Tregenza()
    skydome = ReinhartM2()
    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    # Setup model, run analysis and view results
    engine = SolarEngine(mesh)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def radiance_test():

    # Set path for weather file
    path = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    # Set path for radiance installation
    rad_path = "/usr/local/radiance/bin/"

    rad_sky, rad_sun, rad_tot = calc_radiance_matrices(
        path, sky_type=SkyType.REINHART_578, rad_path=rad_path
    )

    p = SolarParameters(weather_file=path, sun_mapping=SunMapping.RADIANCE)

    skydome = ReinhartM2()
    sunpath = Sunpath(p, include_night=True)

    (sky_res, sun_res) = calc_2_phase_matrix(sunpath, skydome, p.sun_mapping)

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


def skydome_m4_test():
    skydome = ReinhartM4()

    data_dict = {
        "Random data1": np.random.rand(skydome.patch_counter),
        "Random data2": np.random.rand(skydome.patch_counter),
    }

    skydome.view(name="Reinhart4", data_dict=data_dict, sun_pos_pc=None)

    pass


# -------- 2 phase analysis --------


def analyse_mesh_1():
    # filename = "../../../data/validation/boxes_sharp_f5248.obj"
    filename = "../../../data/validation/boxes_soft_f5248.obj"
    # filename = "../../../data/models/City136kSoft.stl"
    mesh = io.load_mesh(filename)
    engine = SolarEngine(mesh)

    path = "../../../data/weather/"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Stockholm
    p = SolarParameters(
        weather_file=path + sth_epw,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM4()
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def analyse_mesh_2():

    filename = "../../../data/validation/boxes_sharp_f5248.obj"
    mesh = io.load_mesh(filename)
    path = "../../../data/weather/"
    gbg_epw = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"

    # Gothenburg
    p = SolarParameters(
        weather_file=path + gbg_epw,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    (analysis_mesh, shading_mesh) = split_mesh_by_vertical_faces(mesh)

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def analyse_mesh_3():
    # filename = "...../../data/validation/boxes_sharp_f5248.obj"
    filename = "../../../data/validation/boxes_soft_f5248.obj"
    mesh = io.load_mesh(filename)

    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.2, 0.8], [0.2, 0.8])
    engine = SolarEngine(analysis_mesh, shading_mesh)

    path = "../../../data/weather/"
    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    # London
    p = SolarParameters(
        weather_file=path + lnd_epw,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


# -------- 3 phase analysis --------


def analyse_mesh_4():
    print("-------- Solar Mesh Analysis Started -------")
    filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    path = "../../../data/weather/"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    mesh = io.load_mesh(filename)
    engine = SolarEngine(mesh)

    # Stockholm
    p = SolarParameters(
        weather_file=path + sth_epw,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Setup model, run analysis and view results
    skydome = ReinhartM2()
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_3_phase_analysis(sunpath, skydome, p)


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")
    info("#################### DTCC-SOLAR #####################")

    # perez_test()
    # embree_perez_test()
    # radiance_test()
    # skydome_m4_test()
    analyse_mesh_1()
    # analyse_mesh_2()
    # analyse_mesh_3()
    # analyse_mesh_4()
