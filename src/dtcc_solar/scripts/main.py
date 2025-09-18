import os
import dtcc
from dtcc_core import io as io

from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart import Reinhart
from dtcc_core.io import load_city
from dtcc_core.model import PointCloud
from dtcc_solar.perez import *
from dtcc_solar.city_utils import *
from dtcc_solar.radiance import load_radiance, epw_to_wea
from pprint import pp

import matplotlib.pyplot as plt


import numpy as np
from urllib.request import urlretrieve


def perez_test():
    print("-------- Skydome Test -------")

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

    (sky_res, sun_res) = calc_2_phase_matrix(sunpath, skydome, p.sun_sky_mapping)

    face_data_dict = {
        "relative lumiance": sky_res.relative_luminance,
        "relative lum norm": sky_res.relative_lum_norm,
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
    print("-------- Skydome Test -------")

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
    skydome = Reinhart()
    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    # Setup model, run analysis and view results
    engine = SolarEngine(mesh)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def sunpath_test():
    print("-------- Skydome Test -------")
    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    p = SolarParameters(
        weather_file=path_lnd,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    sunpath_radius = 50
    sunpath = Sunpath(p, sunpath_radius, interpolate_df=True)

    df = sunpath.df
    df_original = sunpath.df_original

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["dni"], label="interp DNI")
    plt.plot(df_original.index, df_original["dni"], label="real DNI")
    plt.xlabel("Time")
    plt.ylabel("Wh/m²")
    plt.title("Irradiance over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(pd.Timestamp("2019-05-01"), pd.Timestamp("2019-05-30"))
    plt.show()


# -------- 2 phase analysis --------


def analyse_mesh_1():
    print("-------- Solar Mesh Analysis Started -------")
    filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    mesh = io.load_mesh(filename)
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
    skydome = Reinhart()
    engine = SolarEngine(mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def analyse_mesh_2():
    print("-------- Solar Mesh Analysis Started -------")

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
    skydome = Reinhart()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


def analyse_mesh_3():
    print("-------- Solar Mesh Analysis Started -------")

    # filename = "...../../data/validation/boxes_sharp_f5248.obj"
    filename = "../../../data/validation/boxes_soft_f5248.obj"
    mesh = io.load_mesh(filename)

    path = "../../../data/weather/"
    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    # London
    p = SolarParameters(
        weather_file=path + lnd_epw,
        sun_path_type=SunPathType.NORMAL,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.2, 0.8], [0.2, 0.8])

    # Setup model, run analysis and view results
    skydome = Reinhart()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_2_phase_analysis(sunpath, skydome, p)


# -------- 3 phase analysis --------


def analyse_mesh_4():
    print("-------- Solar Mesh Analysis Started -------")
    filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    mesh = io.load_mesh(filename)
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
    skydome = Reinhart()
    engine = SolarEngine(mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    engine.run_3_phase_analysis(sunpath, skydome, p)


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")

    # perez_test()
    # embree_perez_test()
    # sunpath_test()
    # analyse_mesh_1()
    # analyse_mesh_2()
    # analyse_mesh_3()
    analyse_mesh_4()
