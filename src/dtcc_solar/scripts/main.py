import os
import time
import dtcc_io

from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_model import Mesh, PointCloud
from dtcc_io import meshes
from dtcc_solar.city import *
from pprint import pp


def analyse_city(solar_parameters: SolarParameters):
    print("-------- Solar City Analysis Started -------")

    p = solar_parameters
    p.file_name = "../../../data/models/denhaag.city.json"
    city = dtcc_io.load_cityjson(p.file_name)

    bld_mesh, parts = generate_building_mesh_2(city, subdee_length=3.5)
    ter_mesh = get_terrain_mesh(city)
    # terrain_mesh = reduce_mesh(terrain_mesh, 0.95)

    check_mesh(bld_mesh)
    check_mesh(ter_mesh)

    engine = SolarEngine(bld_mesh, ter_mesh, Sky.Tregenza145, Rays.Bundle8)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)

    p.export = False
    if p.export:
        filename = "../../../data/output/test.json"
        export_mesh_to_json(
            bld_mesh,
            parts,
            outputc.data_1["total irradiation (kWh/m2)"],
            outputc.data_1["face sun angles (rad)"],
            outputc.data_1["sun hours [h]"],
            filename,
        )


def analyse_mesh_1(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)

    # Setup model, run analysis and view results
    engine = SolarEngine(mesh, sky=Sky.Reinhart580, rays=Rays.Bundle8)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()
    engine.run_analysis(p, sunpath, outputc)
    engine.view_results(p, sunpath, outputc)

    filename = "../../../data/validation/boxes_soft_f5248_results.json"
    export_results_to_json(len(mesh.faces), p, outputc, filename)


def analyse_mesh_2(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)
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
    mesh = meshes.load_mesh(p.file_name)
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
        end_date="2019-12-31 00:00:00",
        longitude=11.97,
        latitude=57.71,
        data_source=DataSource.epw,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.group,
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
        sun_approx=SunApprox.quad,
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
        sun_approx=SunApprox.group,
    )

    analyse_mesh_1(p_1)
    # analyse_mesh_2(p_1)
    # analyse_mesh_3(p_1)
    # analyse_city(p_1)
