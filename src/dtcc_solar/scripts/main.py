import numpy as np
import pandas as pd
import time
import os
import trimesh
import argparse
import sys

from dtcc_solar.sunpath import Sunpath, Sun
from dtcc_solar.viewer import Viewer
from dtcc_solar.utils import ColorBy, AnalysisType, SolarParameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.results import Results
from dtcc_solar.utils import concatenate_meshes, print_list
from dtcc_solar.colors import *
from dtcc_viewer import MeshShading
from dtcc_solar.sundome import SunDome
from dtcc_solar.skydome import SkyDome
from dtcc_solar.logging import set_log_level, info, debug, warning, error

import dtcc_solar.data_smhi as smhi
import dtcc_solar.data_meteo as meteo
import dtcc_solar.data_epw as epw
import dtcc_solar.data_clm as clm

from dtcc_model import Mesh
from dtcc_io import meshes
from pprint import pp


def export(p: SolarParameters, city_results: Results, exportpath: str):
    if p.color_by == ColorBy.face_sun_angle:
        print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_sun_angle_shadows:
        print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_irradiance_dn:
        print_list(city_results.get_face_irradiance(), exportpath)
    elif p.color_by == ColorBy.face_shadows:
        print_list(city_results.get_face_in_sun(), exportpath)


###############################################################################################################################


def run_script(solar_parameters: SolarParameters):
    os.system("clear")
    print("-------- Solar Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)
    solar_engine = SolarEngine(mesh)
    sunpath = Sunpath(p.latitude, p.longitude, solar_engine.sunpath_radius)
    skydome = SkyDome(solar_engine.dome_radius)
    sundome = SunDome(sunpath, solar_engine.horizon_z, 150, 20)
    suns = sunpath.create_suns(p)

    results = Results(suns, len(mesh.faces))

    # Match suns and quads
    solar_engine.match_suns_and_quads(suns, sundome)
    sunquad_mesh = sundome.get_sub_sundome_mesh(solar_engine.tolerance)

    # Execute analysis
    if p.a_type == AnalysisType.sun_raycasting:
        solar_engine.sun_raycasting(suns, results)
    elif p.a_type == AnalysisType.sky_raycasting:
        solar_engine.sky_raycasting(suns, results, skydome)
    elif p.a_type == AnalysisType.com_raycasting:
        solar_engine.sun_raycasting(suns, results)
        solar_engine.sky_raycasting(suns, results, skydome)
    elif p.a_type == AnalysisType.sun_precasting:
        solar_engine.sun_precasting(suns, results, sundome)
    elif p.a_type == AnalysisType.com_precasting:
        solar_engine.sun_precasting(suns, results, sundome)
        solar_engine.sky_raycasting(suns, results, skydome)

    results.calc_accumulated_results()
    results.calc_average_results()

    if p.prepare_display:
        viewer = Viewer()
        viewer.build_sunpath_diagram(suns, solar_engine, sunpath, sundome)
        colors = color_city_mesh(results.res_acum, p.color_by)
        viewer.add_mesh("City mesh", mesh=solar_engine.dmesh, colors=colors)

        if p.display:
            viewer.show()

    return True


if __name__ == "__main__":
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    weather_file_clm = (
        "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    )

    # Instant solar anaysis
    p_1 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sun_raycasting,
        start_date="2019-03-30 09:00:00",
        end_date="2019-03-30 09:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_sun_angle_shadows,
    )

    # Instant sky analysis
    p_2 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sky_raycasting,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_di,
    )

    # Instant combined analysis
    p_3 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.com_raycasting,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_tot,
    )

    # Iterative solar analysis
    p_4 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sun_raycasting,
        start_date="2019-06-01 11:00:00",
        end_date="2019-06-01 15:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_dn,
    )

    # Iterative sky analysis
    p_5 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sky_raycasting,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_di,
    )

    # Iterative combined analysis
    p_6 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.com_raycasting,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_tot,
    )

    run_script(p_3)
