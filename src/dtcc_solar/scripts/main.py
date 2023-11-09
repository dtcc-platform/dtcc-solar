import numpy as np
import pandas as pd
import os
import time

from dtcc_solar.utils import ColorBy, OutputCollection, SolarParameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.sundome import SunDome
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.colors import color_mesh

from dtcc_model import Mesh
from dtcc_io import meshes
from pprint import pp


def run_script(solar_parameters: SolarParameters):
    os.system("clear")
    print("-------- Solar Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)
    solar_engine = SolarEngine(mesh)
    sunpath = Sunpath(p, solar_engine.sunpath_radius)
    sundome = SunDome(sunpath, 150, 20)
    outputc = OutputCollection()

    solar_engine.run_analysis(p, sunpath.sunc, outputc, sundome)

    if p.display:
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath, sundome)
        colors = color_mesh(outputc, p.color_by)
        viewer.add_mesh("City mesh", mesh=mesh, colors=colors)
        viewer.show()


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
        start_date="2019-03-30 09:00:00",
        end_date="2019-03-30 09:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_sun_angle,
        sun_analysis=True,
        sky_analysis=False,
    )

    # Instant sky analysis
    p_2 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_di,
        sun_analysis=False,
        sky_analysis=True,
    )

    # Instant combined analysis
    p_3 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_dn,
        sun_analysis=True,
        sky_analysis=True,
    )

    # Iterative solar analysis
    p_4 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        start_date="2019-06-01 11:00:00",
        end_date="2019-06-02 11:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_dn,
        sun_analysis=True,
        sky_analysis=False,
    )

    # Iterative sky analysis
    p_5 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_di,
        sun_analysis=False,
        sky_analysis=True,
    )

    # Iterative combined analysis
    p_6 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_tot,
        sun_analysis=True,
        sky_analysis=True,
    )

    run_script(p_4)
