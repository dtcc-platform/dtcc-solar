import os
import time
import dtcc_io

from dtcc_solar.utils import *
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_viewer.opengl.parts import Parts
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_model import Mesh, PointCloud
from dtcc_io import meshes
from pprint import pp


def analyse_city(solar_parameters: SolarParameters):
    print("-------- Solar City Analysis Started -------")

    p = solar_parameters
    p.file_name = "../../../data/models/lozenets_citygml2cityjson_lod1_replaced.json"
    city = dtcc_io.load_cityjson(p.file_name)

    building_mesh, parts = generate_building_mesh(city)
    terrain_mesh = get_terrain_mesh(city)
    terrain_mesh = reduce_mesh(terrain_mesh, 0.95)

    # Analyse the meshes for duplicates and small faces
    check_mesh(building_mesh)
    check_mesh(terrain_mesh)

    engine = SolarEngine(building_mesh, terrain_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()

    engine.run_analysis(p, sunpath, outputc)

    if p.display:
        outputc.process_results(engine.face_mask)
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath, p)
        viewer.add_mesh("Analysed mesh", mesh=building_mesh, data=outputc.data_dict_1)
        viewer.add_mesh("Shading mesh", mesh=terrain_mesh, data=outputc.data_dict_2)
        viewer.show()


def analyse_mesh(solar_parameters: SolarParameters):

    print("-------- Solar Mesh Analysis Started -------")

    p = solar_parameters
    mesh1 = meshes.load_mesh(p.file_name)

    # (analysis_mesh, shading_mesh) = split_mesh_with_domain(mesh, [0.4, 0.6], [0.4, 0.6])
    # analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    (analysis_mesh, shading_mesh) = split_mesh_by_vertical_faces(mesh1)
    analysis_mesh = subdivide_mesh(analysis_mesh, 3.5)

    engine = SolarEngine(analysis_mesh, shading_mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()

    engine.run_analysis(p, sunpath, outputc)

    if p.display:
        outputc.process_results(engine.face_mask)
        print(outputc.face_sun_angles)
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath, p)
        viewer.add_mesh("Analysed mesh", mesh=analysis_mesh, data=outputc.data_dict_1)
        viewer.add_mesh("Shading mesh", mesh=shading_mesh, data=outputc.data_dict_2)
        viewer.show()


if __name__ == "__main__":
    os.system("clear")
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    path = "../../../data/weather/"

    lnd_clm = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    gbg_clm = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.clm"
    gbg_epw = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"
    sth_clm = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.clm"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Gothenburg
    p_1 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + gbg_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=11.97,
        latitude=57.71,
        data_source=DataSource.smhi,
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

    analyse_mesh(p_1)
    # analyse_city(p_1)
