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

import numpy as np
from urllib.request import urlretrieve


def perez_test():
    print("-------- Skydome Test -------")

    path_lnd = "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    long_lnd = 0.12
    lat_lnd = 51.5

    p = SolarParameters(
        file_name="",
        weather_file=path_lnd,
        longitude=long_lnd,
        latitude=lat_lnd,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # wea_file = epw_to_wea(p.weather_file)
    # load_radiance(wea_file)

    skydome = Reinhart()
    skydome.create_mesh()

    sunpath_radius = 1.5
    sunpath = Sunpath(p, sunpath_radius)
    sky_results = calc_sky_matrix(sunpath, skydome)

    # sun_results = calc_sun_matrix(sunpath, skydome)
    # sun_results = calc_sun_mat_flat_smear(sunpath, skydome, da=15)
    sun_results = calc_sun_mat_smooth_smear(sunpath, skydome, da=20)

    sun_pc = PointCloud(points=sunpath.sunc.positions)

    rel_lum = sky_results.relative_luminance
    rel_lum = skydome.map_data_to_faces(rel_lum)

    sky_mat = sky_results.sky_matrix
    sky_mat = skydome.map_data_to_faces(sky_mat)

    solid_angles = sky_results.solid_angles
    sa = skydome.map_data_to_faces(solid_angles)

    ksis = sky_results.ksis
    ksis = skydome.map_data_to_faces(ksis)

    gammas = sky_results.gammas
    gammas = skydome.map_data_to_faces(gammas)

    sun_mat = sun_results.sun_matrix
    sun_mat = skydome.map_data_to_faces(sun_mat)

    dict_data = {
        "relative lumiance": rel_lum,
        "sky matrix": sky_mat,
        "sun matrix": sun_mat,
        "solid angles": sa,
        "ksis": ksis,
        "gammas": gammas,
    }

    skydome.view(name="Skydome", data=dict_data, sun_pos_pc=sun_pc)


def analyse_mesh_1(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)

    # Setup model, run analysis and view results
    engine = SolarEngine(mesh, sky=Sky.Reinhart580, rays=Rays.Bundle8)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)

    filename = os.path.splitext(p.file_name)[0]
    filename = filename + "_results.json"
    export_results_to_json(len(mesh.faces), p, outputc, filename)


def analyse_mesh_2(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)
    (analysis_mesh, shading_mesh) = split_mesh_by_vertical_faces(mesh)
    analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    # Setup model, run analysis and view results
    engine = SolarEngine(analysis_mesh, shading_mesh, sky=Sky.Tregenza145)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)


def analyse_mesh_3(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = io.load_mesh(p.file_name)
    (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.2, 0.8], [0.2, 0.8])
    analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    # Setup model, run analysis and view results
    engine = SolarEngine(analysis_mesh, shading_mesh, sky=Sky.Reinhart580)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)


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
        longitude=11.97,
        latitude=57.71,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Stockholm
    p_2 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + sth_epw,
        longitude=18.063,
        latitude=59.33,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    # Rio de Janeiro
    p_3 = SolarParameters(
        file_name=inputfile_L,
        longitude=-43.19,
        latitude=-22.90,
        sun_analysis=True,
        sky_analysis=True,
        start=pd.Timestamp("2019-01-01 00:00:00"),
        end=pd.Timestamp("2019-12-31 23:00:00"),
    )

    perez_test()
    # analyse_mesh_1(p_1)
    # analyse_mesh_2(p_1)
    # analyse_mesh_3(p_1)
