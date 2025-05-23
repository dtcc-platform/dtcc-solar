import os
import dtcc
from dtcc_core import io as io

from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.skydome import Skydome

# from dtcc_model import Mesh, PointCloud
from dtcc_solar.city import *
from pprint import pp

import numpy as np
from urllib.request import urlretrieve


def skydome_test():
    print("-------- Skydome Test -------")

    path = "../../../data/weather/SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"

    p = SolarParameters(
        file_name="",
        weather_file=path,
        start_date="2019-06-01 00:00:00",
        end_date="2019-06-30 23:00:00",
        longitude=11.97,
        latitude=57.71,
        data_source=DataSource.epw,
        sun_analysis=True,
        sky_analysis=True,
    )

    skydome = Skydome()
    skydome.create_tregenza_mesh()

    sunpath_radius = 10.0
    sunpath = Sunpath(p, sunpath_radius)
    skydome.calc_sky_vector_matrix(sunpath)

    sky_vector_matrix = skydome.sky_vector_matrix
    pp(sky_vector_matrix)

    skydome.view()


def analyse_city(solar_parameters: SolarParameters):
    print("-------- Solar City Analysis Started -------")

    url = (
        "https://3d.bk.tudelft.nl/opendata/cityjson/3dcities/v2.0/DenHaag_01.city.json"
    )
    urlretrieve(url=url, filename="city.json")
    city = dtcc.load_city("city.json")
    bld_meshes = get_building_meshes(city)
    bld_mesh = dtcc.builder.meshing.merge_meshes(bld_meshes)
    ter_mesh = city.terrain.mesh
    check_mesh(bld_mesh)
    check_mesh(ter_mesh)

    move_vec = bld_mesh.vertices.mean()
    bld_mesh.vertices = bld_mesh.vertices - move_vec
    bld_mesh.view()

    ter_mesh.vertices = ter_mesh.vertices - move_vec
    ter_mesh.view()

    p = solar_parameters

    engine = SolarEngine(
        bld_mesh, sky=Sky.Reinhart580, rays=Rays.Bundle8, center_mesh=True
    )
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)


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

    lnd_clm = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    gbg_clm = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.clm"
    gbg_epw = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"
    sth_clm = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.clm"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Gothenburg
    p_1 = SolarParameters(
        file_name=boxes_sharp_f5248,
        weather_file=path + gbg_epw,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 23:00:00",
        longitude=11.97,
        latitude=57.71,
        data_source=DataSource.epw,
        sun_analysis=True,
        sky_analysis=True,
    )

    # Stockholm
    p_2 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + sth_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=18.063,
        latitude=59.33,
        data_source=DataSource.clm,
        sun_analysis=True,
        sky_analysis=True,
    )

    # Rio de Janeiro
    p_3 = SolarParameters(
        file_name=inputfile_L,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=-43.19,
        latitude=-22.90,
        data_source=DataSource.meteo,
        sun_analysis=True,
        sky_analysis=True,
    )

    analyse_mesh_1(p_1)
    # analyse_mesh_2(p_1)
    # analyse_mesh_3(p_1)
    # analyse_city(p_1)
    # skydome_test()
