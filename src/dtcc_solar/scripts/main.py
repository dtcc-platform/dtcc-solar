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

    lat_lnd = 51.5
    long_lnd = 0.12

    p = SolarParameters(
        file_name="",
        weather_file=path_lnd,
        latitude=lat_lnd,
        longitude=long_lnd,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-06-01 12:00:00"),
        end=pd.Timestamp("2019-06-01 13:00:00"),
    )

    # wea_file = epw_to_wea(p.weather_file)
    # load_radiance(wea_file)

    skydome = Reinhart()

    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)
    sky_results = calc_sky_matrix(sunpath, skydome)

    # sun_results = calc_sun_matrix(sunpath, skydome)
    # sun_results = calc_sun_mat_flat_smear(sunpath, skydome, da=15)
    sun_results = calc_sun_mat_smooth_smear(sunpath, skydome, da=20)

    sun_pc = PointCloud(points=sunpath.sunc.positions)

    face_data_dict = {
        "relative lumiance": sky_results.relative_luminance,
        "relative lum norm": sky_results.relative_lum_norm,
        "sky matrix": sky_results.sky_matrix,
        "sun matrix": sun_results.sun_matrix,
        "total matrix": sky_results.sky_matrix + sun_results.sun_matrix,
        "solid angles": sky_results.solid_angles,
        "ksis": sky_results.ksis,
        "gammas": sky_results.gammas,
    }

    # Compare total measured radiation with the sum of radiation on skydomes
    calc_tot_error(sky_results, skydome, sun_results, sunpath)

    skydome.view(name="Skydome", data_dict=face_data_dict, sun_pos_pc=sun_pc)


def embree_perez_test():
    print("-------- Skydome Test -------")

    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    # filename = "../../../data/validation/boxes_sharp_f5248.obj"
    # filename = "../../../data/validation/boxes_soft_f5248.obj"
    filename = "../../../data/models/CitySurfaceS.stl"

    lat_lnd = 51.5
    long_lnd = 5.5e-2

    mesh = io.load_mesh(filename)

    p = SolarParameters(
        file_name="",
        weather_file=path_lnd,
        latitude=lat_lnd,
        longitude=long_lnd,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # skydome = Tregenza()
    skydome = Reinhart()

    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)

    sky = calc_sky_matrix(sunpath, skydome)
    sun = calc_sun_mat_smooth_smear(sunpath, skydome, da=20)
    tot = sky.sky_matrix + sun.sun_matrix

    face_data_dict = {
        "sky matrix": sky.sky_matrix,
        "sun matrix": sun.sun_matrix,
        "total matrix": tot,
        "solid angles": sky.solid_angles,
        "ksis": sky.ksis,
        "gammas": sky.gammas,
    }

    sun_pc = PointCloud(points=sunpath.sunc.positions)
    skydome.view("skydome", data_dict=face_data_dict, sun_pos_pc=sun_pc)

    # Setup model, run analysis and view results
    engine = SolarEngine(mesh)

    engine.run_2_phase_analysis(sunpath, skydome, tot)


def sunpath_test():
    print("-------- Skydome Test -------")
    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    # filename = "../../../data/validation/boxes_sharp_f5248.obj"
    filename = "../../../data/validation/boxes_soft_f5248.obj"
    # filename = "../../../data/models/CitySurfaceS.stl"

    lat_lnd = 51.5
    long_lnd = 5.5e-2

    p = SolarParameters(
        file_name="",
        weather_file=path_lnd,
        latitude=lat_lnd,
        longitude=long_lnd,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    sunpath_radius = 50
    sunpath = Sunpath(p, sunpath_radius)

    df = sunpath.df
    df_itp = sunpath.df_itp

    plt.figure(figsize=(12, 4))
    plt.plot(df_itp.index, df_itp["dni"], label="interp DNI")
    plt.plot(df.index, df["dni"], label="real DNI")
    plt.xlabel("Time")
    plt.ylabel("Wh/m²")
    plt.title("Irradiance over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(pd.Timestamp("2019-05-01"), pd.Timestamp("2019-05-30"))
    plt.show()


def analyse_mesh_1(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)

    # Setup model, run analysis and view results
    skydome = Reinhart()
    engine = SolarEngine(mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    matrix = calc_sky_sun_matrix(sunpath, skydome, da=20)
    engine.run_2_phase_analysis(sunpath, skydome, matrix)


def analyse_mesh_2(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)
    (analysis_mesh, shading_mesh) = split_mesh_by_vertical_faces(mesh)
    # analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    # Setup model, run analysis and view results
    skydome = Reinhart()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    matrix = calc_sky_sun_matrix(sunpath, skydome, da=20)
    engine.run_2_phase_analysis(sunpath, skydome, matrix)


def analyse_mesh_3(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)
    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.2, 0.8], [0.2, 0.8])
    # analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    # Setup model, run analysis and view results
    skydome = Reinhart()
    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    matrix = calc_sky_sun_matrix(sunpath, skydome, da=20)
    engine.run_2_phase_analysis(sunpath, skydome, matrix)


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    path = "../../../data/weather/"

    box_sharp_f26 = "../../../data/validation/box_sharp_f26.obj"
    box_sharp_f1664 = "../../../data/validation/box_sharp_f1664.obj"
    box_soft_f1664 = "../../../data/validation/box_soft_f1664.obj"
    boxes_sharp_f5248 = "../../../data/validation/boxes_sharp_f5248.obj"
    boxes_soft_f5248 = "../../../data/validation/boxes_soft_f5248.obj"

    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    gbg_epw = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Gothenburg
    p_1 = SolarParameters(
        file_name=boxes_sharp_f5248,
        weather_file=path + gbg_epw,
        latitude=57.66,
        longitude=12.28,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Stockholm
    p_2 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + sth_epw,
        latitude=59.33,
        longitude=18.063,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Rio de Janeiro
    p_3 = SolarParameters(
        file_name=inputfile_L,
        latitude=-22.90,
        longitude=-43.19,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # perez_test()
    # embree_perez_test()
    # sunpath_test()
    # analyse_mesh_1(p_1)
    analyse_mesh_2(p_2)
    # analyse_mesh_3(p_1)
