import numpy as np
import pandas as pd
import time
import os
import trimesh
import argparse
import sys

from dtcc_solar import data_io
from dtcc_solar.sunpath import Sunpath, Sun
from dtcc_solar.sunpath_vis import SunpathMesh
from dtcc_solar.viewer import Viewer
from dtcc_solar.utils import ColorBy, AnalysisType, Parameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.results import Results
from dtcc_solar import weather_data as weather
from dtcc_solar.utils import concatenate_meshes
from dtcc_solar.colors import *

import dtcc_solar.smhi_data as smhi
import dtcc_solar.meteo_data as meteo
import dtcc_solar.epw_data as epw
import dtcc_solar.clm_data as clm

from dtcc_model import Mesh
from dtcc_io import meshes


from pprint import pp


def register_args(args):
    default_path = (
        "/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl"
    )
    parser = argparse.ArgumentParser(
        description="Parameters to run city solar analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--analysis",
        type=int,
        metavar="",
        default=AnalysisType.sun_raycasting,
        help=" sun_raycasting = 1, sky_raycasting = 2, com_raycasting = 3, dome_raycasting = 4",
    )
    parser.add_argument(
        "-lat",
        "--latitude",
        type=float,
        metavar="",
        default=51.5,
        help="Latitude for location of analysis",
    )
    parser.add_argument(
        "-lon",
        "--longitude",
        type=float,
        metavar="",
        default=-0.12,
        help="Longitude for location of analysis",
    )
    parser.add_argument(
        "-f",
        "--inputfile",
        type=str,
        metavar="",
        default=default_path,
        help="Filename (incl. path) for city mesh (*.stl, *.vtu)",
    )
    parser.add_argument(
        "-sd",
        "--start_date",
        type=str,
        metavar="",
        default="2019-03-30 07:00:00",
        help="Start date for iterative analysis",
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        metavar="",
        default="2019-03-30 21:00:00",
        help="End date for iterative analysis",
    )
    parser.add_argument(
        "-disp",
        "--display",
        type=int,
        metavar="",
        default=1,
        help="Display results with pyglet",
    )
    parser.add_argument(
        "-pd",
        "--prep_disp",
        type=int,
        metavar="",
        default=1,
        help="Preproscess colors and other graphic for display",
    )
    parser.add_argument(
        "-ds",
        "--data_source",
        type=int,
        metavar="",
        default=DataSource.clm,
        help="Enum for data source. 1 = SMHI, 2 = Open Meteo, 3 = Clm file, 4 = Epw file",
    )
    parser.add_argument(
        "-c",
        "--colorby",
        type=int,
        metavar="",
        default=ColorBy.face_sun_angle_shadows,
        help="Colo_by: face_sun_angle =1, face_sun_angle_shadows = 2, shadows = 3 , irradiance_direct_normal = 4, irradiance_direct_horizontal = 5, irradiance_diffuse = 6, irradiance_all = 6",
    )
    parser.add_argument(
        "-e", "--export", type=bool, metavar="", default=True, help="Export data"
    )
    parser.add_argument(
        "-ep",
        "--exportpath",
        type=str,
        metavar="",
        default="./data/dataExport.txt",
        help="Path for data export of type *.txt",
    )
    parser.add_argument(
        "-wf",
        "--w_file",
        type=str,
        metavar="",
        default="./data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm",
        help="Weather data file to be uploaded by the user",
    )
    new_args = parser.parse_args(args)
    return new_args


def print_args(args):
    for arg in vars(args):
        print(arg, "\t", getattr(args, arg))
    print("----------------------------------------")


def export(p: Parameters, city_results: Results, exportpath: str):
    if p.color_by == ColorBy.face_sun_angle:
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_sun_angle_shadows:
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_irradiance_dn:
        data_io.print_list(city_results.get_face_irradiance(), exportpath)
    elif p.color_by == ColorBy.face_shadows:
        data_io.print_list(city_results.get_face_in_sun(), exportpath)


###############################################################################################################################


def run_script(command_line_args):
    clock1 = time.perf_counter()
    os.system("clear")
    print("-------- Solar Analysis Started -------")

    args = register_args(command_line_args)

    # Convert command line input to enums and data formated for the analysis

    p = Parameters(
        args.inputfile,
        args.w_file,
        args.analysis,
        args.latitude,
        args.longitude,
        args.prep_disp,
        args.display,
        args.data_source,
        args.colorby,
        args.export,
        args.start_date,
        args.end_date,
    )

    print_args(args)

    # city_mesh = trimesh.load_mesh(p.file_name)

    dmesh = meshes.load_mesh(p.file_name)
    solar_engine = SolarEngine(dmesh)
    sunpath = Sunpath(p.latitude, p.longitude, solar_engine.sunpath_radius)
    suns = sunpath.create_suns(p)
    suns = weather.append_weather_data(p, suns)
    results = Results(suns, len(dmesh.faces))

    # Execute analysis
    if p.a_type == AnalysisType.sun_raycasting:
        solar_engine.sun_raycasting(suns, results)
    elif p.a_type == AnalysisType.sky_raycasting:
        solar_engine.sky_raycasting(suns, results)
    elif p.a_type == AnalysisType.com_raycasting:
        solar_engine.sun_raycasting(suns, results)
        solar_engine.sky_raycasting(suns, results)

    results.calc_accumulated_results()
    results.calc_average_results()

    if p.prepare_display:
        viewer_gl = Viewer()

        # Create sunpath so that the solar postion are given a context in the 3D visualisation
        sunpath_mesh = SunpathMesh(solar_engine.sunpath_radius)
        sunpath_mesh.create_sunpath_diagram_gl(suns, sunpath, solar_engine)

        # Color city mesh and add to viewer
        colors = color_city_mesh(results.res_acum, p.color_by)

        # Get analemmas, day paths, and pc for sun positions
        analemmas = sunpath_mesh.get_analemmas_meshes()
        day_paths = sunpath_mesh.get_daypath_meshes()
        pc = sunpath_mesh.get_analemmas_pc()

        analemmas = concatenate_meshes(analemmas)
        day_paths = concatenate_meshes(day_paths)

        viewer_gl.add_mesh("Dtcc mesh", solar_engine.dmesh, colors=colors)
        viewer_gl.add_mesh("Analemmas", analemmas)
        viewer_gl.add_mesh("Day paths", day_paths)
        viewer_gl.add_pc("Sun positions", pc)

        if p.display:
            viewer_gl.show()

    clock2 = time.perf_counter()
    print("Total computation time elapsed: " + str(round(clock2 - clock1, 4)))
    print("----------------------------------------")
    return True


if __name__ == "__main__":
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    other_file_to_run = "../../../data/models/new_file.stl"

    weather_file_clm = (
        "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    )

    # Instant solar anaysis
    args_1 = [
        "--inputfile",
        inputfile_L,
        "--analysis",
        "1",
        "--start_date",
        "2019-03-30 09:00:00",
        "--end_date",
        "2019-03-30 09:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "2",
    ]

    # Instant sky analysis
    args_2 = [
        "--inputfile",
        inputfile_S,
        "--analysis",
        "2",
        "--start_date",
        "2019-03-30 12:00:00",
        "--end_date",
        "2019-03-30 12:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "6",
    ]

    # Instant combined analysis
    args_3 = [
        "--inputfile",
        inputfile_S,
        "--analysis",
        "3",
        "--start_date",
        "2015-03-30 12:00:00",
        "--end_date",
        "2015-03-30 12:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "7",
    ]

    # Iterative solar analysis
    args_4 = [
        "--inputfile",
        inputfile_L,
        "--analysis",
        "1",
        "--start_date",
        "2019-06-01 11:00:00",
        "--end_date",
        "2019-06-01 15:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "4",
    ]

    # Iterative sky analysis
    args_5 = [
        "--inputfile",
        inputfile_S,
        "--analysis",
        "2",
        "--start_date",
        "2019-03-30 06:00:00",
        "--end_date",
        "2019-03-30 21:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "6",
    ]

    # Iterative combined analysis
    args_6 = [
        "--inputfile",
        inputfile_S,
        "--analysis",
        "3",
        "--start_date",
        "2019-03-30 06:00:00",
        "--end_date",
        "2019-03-30 21:00:00",
        "--data_source",
        "3",
        "--w_file",
        weather_file_clm,
        "--colorby",
        "7",
    ]

    run_script(args_1)
